"""Model construction: SmolVLM-256M + the four adaptation methods.

Methods
-------
ft          full finetuning of the arm's modules (the reference for Eq. 3/4)
lora        LoRA on the arm's Linear modules
dora        DoRA (use_dora=True) on the same modules
magonly     DoRA with lora_A/lora_B frozen at B=0 -> a learned per-output-channel
            gain  y = (m/||W||) * (Wx) + b.  ~out_features params per layer.
magV_doraL  DoRA on vision+text, with lora_A/lora_B frozen in the *vision* tower
            only: the direct translation of DoRA sec. 5.6 / Table 6 to a VLM.

Everything asserts. Silent mis-targeting (vision uses `out_proj`/`fc1`/`fc2`, not
`o_proj`/gate-up-down) is the single most likely way to waste a day here.
"""
from __future__ import annotations

import re

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoProcessor

try:  # transformers >= 4.46; AutoModelForVision2Seq was removed in 5.x
    from transformers import AutoModelForImageTextToText as _AutoVLM
except ImportError:  # older 4.x
    from transformers import AutoModelForVision2Seq as _AutoVLM

import config as cfg


def load_processor():
    """do_image_splitting=False pins every image to exactly 64 image tokens.

    This is the single largest VRAM / wall-clock lever (splitting can reach 1088
    tokens at longest_edge=2048). It is applied uniformly to every run, so it is
    a control rather than a confound -- but it does cap fine text detail, which
    matters for the TextVQA arm and must be stated in the report.
    """
    proc = AutoProcessor.from_pretrained(cfg.MODEL_ID, do_image_splitting=False)
    proc.image_processor.do_image_splitting = False
    return proc


def load_base_model(dtype=torch.float32):
    """Loaded in fp32; mixed precision is handled by the Trainer's fp16 AMP path.

    The T4 has no hardware bf16, and pure-fp16 weights with AdamW are the usual
    source of the NaNs the assignment warns about. fp32 weights + fp16 autocast
    costs ~1 GB and removes the problem.
    """
    try:
        model = _AutoVLM.from_pretrained(
            cfg.MODEL_ID, dtype=dtype, attn_implementation=ATTN_IMPL)
    except TypeError:  # transformers 4.x spells the argument torch_dtype
        model = _AutoVLM.from_pretrained(
            cfg.MODEL_ID, torch_dtype=dtype, attn_implementation=ATTN_IMPL)
    return model


# SDPA, not eager: the SigLIP tower attends over 1024 patches per image, and
# eager attention keeps a 1024x1024 score matrix per head, layer and image for
# backward -- ~12 GB at batch 8 with only the vision LoRA trainable, enough to
# OOM a 16 GB T4. SDPA computes the same function without materialising it.
import os as _os
ATTN_IMPL = _os.environ.get("ATTN_IMPL", "sdpa")


# --- helpers -----------------------------------------------------------------

def matched_linear_names(model, pattern: str) -> list[str]:
    rx = re.compile(pattern)
    return [n for n, m in model.named_modules()
            if isinstance(m, torch.nn.Linear) and rx.fullmatch(n)]


def count_trainable(model) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def freeze_lora_ab(model, scope_pattern: str | None = None) -> int:
    """Freeze lora_A/lora_B, optionally only inside `scope_pattern`.

    PEFT bundles lora_A, lora_B and lora_magnitude_vector under one
    `adapter_layer_names` tuple, and `set_adapter()` sets requires_grad for all
    three uniformly -- there is no config flag for magnitude-only. This must run
    *after* get_peft_model(), and nothing may call set_adapter / add_adapter /
    update_layer afterwards (update_layer ends with a set_adapter call that would
    silently re-enable them). The caller asserts the parameter count.
    """
    rx = re.compile(scope_pattern) if scope_pattern else None
    frozen = 0
    for name, param in model.named_parameters():
        if "lora_A" not in name and "lora_B" not in name:
            continue
        if rx is not None:
            owner = name.rsplit(".lora_", 1)[0].replace("base_model.model.", "")
            if not rx.fullmatch(owner):
                continue
        param.requires_grad = False
        frozen += 1
    return frozen


def _freeze_all(model):
    for p in model.parameters():
        p.requires_grad = False


# --- the four methods --------------------------------------------------------

def build_model(method: str, arm: str, model=None, verbose: bool = True):
    """Return (model, info). `info` carries everything the run record needs."""
    if arm not in cfg.ARM_REGEX:
        raise ValueError(f"arm must be one of {list(cfg.ARM_REGEX)}")
    model = model if model is not None else load_base_model()
    pattern = cfg.ARM_REGEX[arm]

    targets = matched_linear_names(model, pattern)
    expected = cfg.ARM_EXPECTED_MODULES[arm]
    assert len(targets) == expected, (
        f"arm {arm!r} matched {len(targets)} Linear modules, expected {expected}. "
        "The vision tower uses out_proj/fc1/fc2 -- check the regex.")
    if arm in ("V", "B"):
        assert any(n.endswith("out_proj") for n in targets), "vision out_proj not matched"
        assert any(n.endswith("fc1") for n in targets), "vision fc1 not matched"

    if method == "ft":
        _freeze_all(model)
        rx = re.compile(pattern)
        unfrozen = 0
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Linear) and rx.fullmatch(name):
                for p in module.parameters():
                    p.requires_grad = True
                    unfrozen += 1
        assert unfrozen > 0
        peft_model = model
        frozen_ab = 0
    else:
        lora_cfg = LoraConfig(
            r=cfg.LORA_R, lora_alpha=cfg.LORA_ALPHA, lora_dropout=cfg.LORA_DROPOUT,
            bias="none", target_modules=pattern,
            use_dora=method in ("dora", "magonly", "magV_doraL"),
            task_type="CAUSAL_LM",
        )
        peft_model = get_peft_model(model, lora_cfg)
        if method == "magonly":
            frozen_ab = freeze_lora_ab(peft_model)            # whole arm
        elif method == "magV_doraL":
            frozen_ab = freeze_lora_ab(peft_model, cfg.VISION_RE)  # vision only
        else:
            frozen_ab = 0

    info = {
        "method": method, "arm": arm,
        "n_target_modules": len(targets),
        "n_frozen_lora_ab": frozen_ab,
        "trainable_params": count_trainable(peft_model),
        "total_params": sum(p.numel() for p in peft_model.parameters()),
    }
    info["trainable_pct"] = 100.0 * info["trainable_params"] / info["total_params"]

    if method in ("magonly", "magV_doraL"):
        # With B frozen at zero the low-rank branch is identically zero forever,
        # so these layers reduce to a diagonal output gain -- not DoRA. Verify no
        # lora_A/lora_B in the frozen scope is still trainable.
        live = [n for n, p in peft_model.named_parameters()
                if p.requires_grad and ("lora_A" in n or "lora_B" in n)]
        if method == "magonly":
            assert not live, f"{len(live)} lora_A/B params still trainable"
        else:
            assert all("vision_model" not in n for n in live), "vision lora_A/B still live"

    if verbose:
        print(f"[build] {method}/{arm}: {len(targets)} modules, "
              f"{info['trainable_params']:,} trainable "
              f"({info['trainable_pct']:.3f}%), froze {frozen_ab} lora_A/B tensors")
    return peft_model, info

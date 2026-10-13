"""Verification checks from plan sec. 12. Run these BEFORE spending Colab quota.

    python tests/test_sanity.py          # all checks, ~2 min, downloads the model
    pytest tests/test_sanity.py -q

Each check targets a specific way this experiment can silently produce numbers
that look fine and mean nothing.
"""
from __future__ import annotations

import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

import config as cfg
import decompose
from model import (build_model, load_base_model, load_processor,
                   matched_linear_names, count_trainable)


def test_decomposition_math():
    """Round-trip, self-distance zero, and a hand-checked case."""
    torch.manual_seed(0)
    W = torch.randn(7, 5)
    for axis in decompose.AXES:
        m, Vhat = decompose.decompose(W, axis=axis)
        rebuilt = m.unsqueeze(1) * Vhat
        expected = W if axis == "out" else W.t()
        assert torch.allclose(rebuilt, expected, atol=1e-5), f"round-trip failed ({axis})"
        assert torch.allclose(torch.linalg.norm(Vhat, dim=1), torch.ones(len(m)), atol=1e-5)
        dM, dD = decompose.delta_m_d(W, W, axis=axis)
        assert abs(dM) < 1e-6 and abs(dD) < 1e-6, f"self-distance non-zero ({axis})"

    # Pure magnitude change: direction must be unchanged.
    dM, dD = decompose.delta_m_d(W, 2.0 * W, axis="out")
    assert dD < 1e-6, "scaling a matrix must not change direction"
    assert abs(dM - torch.linalg.norm(W, dim=1).mean().item()) < 1e-4

    # Pure direction change: a sign flip is maximal direction, zero magnitude.
    dM, dD = decompose.delta_m_d(W, -W, axis="out")
    assert abs(dM) < 1e-5 and abs(dD - 2.0) < 1e-5
    print("[ok] decomposition math")


def test_axis_shapes():
    """The magnitude vector is per OUTPUT feature, as PEFT computes it."""
    W = torch.randn(7, 5)
    assert decompose.magnitudes(W, "out").shape == (7,)
    assert decompose.magnitudes(W, "in").shape == (5,)
    print("[ok] axis shapes (out -> out_features, in -> in_features)")


def test_target_modules():
    """The vision tower uses out_proj/fc1/fc2 -- a Llama-style list silently misses it."""
    model = load_base_model()
    for arm, expected in cfg.ARM_EXPECTED_MODULES.items():
        names = matched_linear_names(model, cfg.ARM_REGEX[arm])
        assert len(names) == expected, f"arm {arm}: {len(names)} != {expected}"
    vision = matched_linear_names(model, cfg.VISION_RE)
    for suffix in ("q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"):
        assert any(n.endswith(suffix) for n in vision), f"vision {suffix} unmatched"
    assert not any(n.endswith("o_proj") for n in vision), "vision has no o_proj"
    print(f"[ok] target modules: V={len(vision)}, "
          f"L={cfg.ARM_EXPECTED_MODULES['L']}, B={cfg.ARM_EXPECTED_MODULES['B']}")
    return model


def test_param_counts():
    for method, arm in (("lora", "V"), ("dora", "V"), ("magonly", "V"), ("lora", "L")):
        _, info = build_model(method, arm, model=load_base_model(), verbose=False)
        assert info["trainable_params"] > 0
        if method == "magonly":
            # 12 layers x (4*768 attn outs + 3072 fc1 + 768 fc2) = 82,944
            assert info["trainable_params"] == 82944, info["trainable_params"]
    print("[ok] trainable parameter counts (magnitude-only V = 82,944)")


def test_dora_is_identity_at_init():
    """B is zero-init, so m = ||W||, mag_norm_scale == 1, and the delta is
    identically zero: a DoRA-wrapped model must reproduce the base model exactly."""
    processor = load_processor()
    from PIL import Image
    img = Image.new("RGB", (512, 512), (128, 128, 128))
    text = processor.apply_chat_template(
        [{"role": "user", "content": [{"type": "image"},
                                      {"type": "text", "text": "What is this?"}]}],
        add_generation_prompt=True)
    enc = processor(text=text, images=[img], return_tensors="pt")

    base = load_base_model().eval()
    with torch.no_grad():
        ref = base(**enc).logits
    wrapped, _ = build_model("dora", "V", model=base, verbose=False)
    wrapped.eval()
    with torch.no_grad():
        got = wrapped(**enc).logits
    max_diff = (ref - got).abs().max().item()
    assert max_diff < 1e-4, f"DoRA is not identity at init: max|diff| = {max_diff}"
    print(f"[ok] DoRA identity at init (max|diff| = {max_diff:.2e})")


def test_image_token_count():
    """do_image_splitting=False must pin every image to exactly 64 image tokens."""
    processor = load_processor()
    from PIL import Image
    assert processor.image_processor.do_image_splitting is False
    for size in ((512, 512), (1600, 900), (144, 200)):
        img = Image.new("RGB", size, (200, 10, 10))
        text = processor.apply_chat_template(
            [{"role": "user", "content": [{"type": "image"},
                                          {"type": "text", "text": "q?"}]}],
            add_generation_prompt=True)
        enc = processor(text=text, images=[img], return_tensors="pt")
        n_img = int((enc["input_ids"][0] == processor.tokenizer.convert_tokens_to_ids(
            "<image>")).sum())
        assert n_img == cfg.IMAGE_SEQ_LEN, f"{size}: {n_img} image tokens, expected 64"
        assert enc["pixel_values"].shape[1] == 1, "more than one image patch"
    print(f"[ok] image tokens pinned to {cfg.IMAGE_SEQ_LEN} at every input size")


def test_label_masking():
    """Loss must cover the answer only -- never the 64 image placeholders."""
    from PIL import Image
    import data as data_mod
    processor = load_processor()
    coll = data_mod.SupervisedCollator(processor)
    batch = coll([{"image": Image.new("RGB", (512, 512)),
                   "question": "Is this abnormal?", "answers": ["yes"]},
                  {"image": Image.new("RGB", (512, 512)),
                   "question": "What colour is the sign?", "answers": ["red"]}])
    labels = batch["labels"]
    assert (labels != -100).any(), "every label masked -- nothing to learn from"
    img_id = processor.tokenizer.convert_tokens_to_ids("<image>")
    assert not ((labels == img_id).any()), "image tokens leaked into the loss"
    kept = (labels != -100).sum(dim=1)
    assert (kept > 0).all() and (kept < 12).all(), f"suspicious label spans: {kept}"
    print(f"[ok] assistant-only loss masking (supervised tokens per example: {kept.tolist()})")


ALL = [test_decomposition_math, test_axis_shapes, test_target_modules,
       test_param_counts, test_dora_is_identity_at_init,
       test_image_token_count, test_label_masking]

if __name__ == "__main__":
    failed = 0
    for fn in ALL:
        try:
            fn()
        except Exception as exc:
            failed += 1
            print(f"[FAIL] {fn.__name__}: {exc}")
    print(f"\n{len(ALL) - failed}/{len(ALL)} checks passed")
    sys.exit(1 if failed else 0)

"""Weight-decomposition analysis: DoRA Eq. (2), (3), (4).

    W = m * V/||V||_c          (Eq. 2)
    dM^t = (1/k) sum_n |m^{n,t} - m^n_0|                     (Eq. 3)
    dD^t = (1/k) sum_n (1 - cos(V^{n,t}, W^n_0))             (Eq. 4)

AXIS, and why it matters
------------------------
The paper writes W in R^{d x k} with m in R^{1 x k}, i.e. one magnitude per
*column*. PEFT (and the official DoRA repo) instead compute

    torch.linalg.norm(weight, dim=1)      # weight is (out_features, in_features)

which reduces over in_features and yields one magnitude per *output* feature.
The implementation is what DoRA actually trains, so `axis="out"` is the default
and is the number reported in the paper's place. `axis="in"` is computed too, as
a robustness check -- it costs one extra reduction and pre-empts the obvious
objection that the analysis measured an axis the method never touches.

Note that V = W / ||W||_c is a positive rescaling of W along the chosen axis, so
cos(V^{n,t}, W^n_0) == cos(W^{n,t}, W^n_0); Eq. (4) needs no explicit division.
"""
from __future__ import annotations

import torch

AXES = ("out", "in")


def _vectors(W: torch.Tensor, axis: str) -> torch.Tensor:
    """Return the matrix arranged so that dim 0 indexes the n units of Eq. (2)-(4)."""
    if axis == "out":          # one unit per output feature: the rows of (out, in)
        return W
    if axis == "in":           # one unit per input feature: the columns
        return W.t()
    raise ValueError(f"axis must be one of {AXES}, got {axis!r}")


def magnitudes(W: torch.Tensor, axis: str = "out") -> torch.Tensor:
    """m: the per-unit L2 norms of W (DoRA Eq. 2)."""
    return torch.linalg.norm(_vectors(W.float(), axis), dim=1)


def decompose(W: torch.Tensor, axis: str = "out"):
    """Return (m, Vhat) with Vhat's units unit-norm, so W == m[:, None] * Vhat."""
    X = _vectors(W.float(), axis)
    m = torch.linalg.norm(X, dim=1)
    Vhat = X / m.clamp_min(torch.finfo(X.dtype).tiny).unsqueeze(1)
    return m, Vhat


def delta_m_d(W0: torch.Tensor, Wt: torch.Tensor, axis: str = "out"):
    """(dM, dD) between a pretrained weight and a finetuned/merged weight."""
    if W0.shape != Wt.shape:
        raise ValueError(f"shape mismatch: {tuple(W0.shape)} vs {tuple(Wt.shape)}")
    # float64: 1 - cos has a float32 noise floor of ~1e-7, the same order as a
    # small learning rate's early direction change. CPU cost is negligible.
    A = _vectors(W0.double(), axis)
    B = _vectors(Wt.double(), axis)
    m0 = torch.linalg.norm(A, dim=1)
    mt = torch.linalg.norm(B, dim=1)
    dM = (mt - m0).abs().mean()
    # cos is invariant to the positive rescaling that turns W into V, so Eq. (4)
    # can be evaluated on the raw rows directly.
    cos = torch.nn.functional.cosine_similarity(B, A, dim=1, eps=1e-12)
    dD = (1.0 - cos).mean()
    return dM.item(), dD.item()


# --- effective (merged) weights for PEFT adapter layers ----------------------
# The paper decomposes the *merged* LoRA/DoRA weight against W_0. We reconstruct
# it analytically rather than calling merge_adapter(), so the live model is never
# mutated mid-training.

def _lora_delta(module, adapter: str) -> torch.Tensor:
    A = module.lora_A[adapter].weight          # (r, in)
    B = module.lora_B[adapter].weight          # (out, r)
    return (B @ A) * module.scaling[adapter]


def effective_weight(module, adapter: str = "default") -> torch.Tensor:
    """W_eff for a peft LoraLayer, matching PEFT's own forward exactly.

    LoRA: W0 + s*B@A
    DoRA: (m / ||W0 + s*B@A||_row)[:, None] * (W0 + s*B@A)
    """
    W0 = module.get_base_layer().weight
    W = W0.float() + _lora_delta(module, adapter).float()
    mag_dict = getattr(module, "lora_magnitude_vector", None)
    if mag_dict and adapter in mag_dict:
        # PEFT stores the magnitude as `.weight` of a DoraLinearLayer (ModuleDict,
        # not ParameterDict). Its norm axis is dim=1 of the (out, in) weight.
        m = mag_dict[adapter].weight.float()
        norm = torch.linalg.norm(W, dim=1).clamp_min(1e-12)
        W = (m / norm).unsqueeze(1) * W
    return W


def is_adapter_layer(module) -> bool:
    return hasattr(module, "lora_A") and hasattr(module, "get_base_layer")


def snapshot_base_weights(model) -> dict[str, torch.Tensor]:
    """W_0 for every Linear we will track, captured once before training.

    Stored on CPU in float32: 282 matrices of a 256M model is well under 1 GB and
    removes any risk of reading an already-updated weight later.
    """
    base = {}
    for name, mod in model.named_modules():
        if is_adapter_layer(mod):
            base[_clean(name)] = mod.get_base_layer().weight.detach().float().cpu().clone()
        elif isinstance(mod, torch.nn.Linear) and _tracked(name):
            base[_clean(name)] = mod.weight.detach().float().cpu().clone()
    return base


def _clean(name: str) -> str:
    """Strip peft's wrapper prefixes so names match the raw checkpoint keys."""
    return name.replace("base_model.model.", "").replace("base_model.", "")


def _tracked(name: str) -> bool:
    import config as _cfg
    return any(name.replace("base_model.model.", "").startswith(p)
               for p in _cfg.MODULE_GROUPS.values()) and name.endswith(
        ("q_proj", "k_proj", "v_proj", "o_proj", "out_proj",
         "fc1", "fc2", "gate_proj", "up_proj", "down_proj", "proj"))


def current_weights(model) -> dict[str, torch.Tensor]:
    """W_t for every tracked Linear, merged analytically where an adapter is present."""
    cur = {}
    for name, mod in model.named_modules():
        if is_adapter_layer(mod):
            adapters = list(mod.lora_A.keys())
            if not adapters:
                continue
            cur[_clean(name)] = effective_weight(mod, adapters[0]).detach().cpu()
        elif isinstance(mod, torch.nn.Linear) and _tracked(name):
            cur[_clean(name)] = mod.weight.detach().float().cpu()
    return cur


def group_of(name: str) -> str:
    import config as _cfg
    for group, prefix in _cfg.MODULE_GROUPS.items():
        if name.startswith(prefix):
            return group
    return "other"


def layer_of(name: str) -> int:
    import re
    m = re.search(r"layers\.(\d+)\.", name)
    return int(m.group(1)) if m else -1


def matrix_of(name: str) -> str:
    return name.rsplit(".", 1)[-1]


def collect_stats(base: dict[str, torch.Tensor], model, step: int, frac: float):
    """One row per (matrix, axis) at this checkpoint."""
    cur = current_weights(model)
    rows = []
    for name, W0 in base.items():
        Wt = cur.get(name)
        if Wt is None:
            continue
        for axis in AXES:
            dM, dD = delta_m_d(W0, Wt, axis=axis)
            rows.append({
                "step": step, "frac": frac, "module": name,
                "group": group_of(name), "layer": layer_of(name),
                "matrix": matrix_of(name), "axis": axis,
                "delta_m": dM, "delta_d": dD,
            })
    return rows

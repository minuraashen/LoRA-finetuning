"""Central configuration: model constants, module scoping, and the tiered run grid.

Every architectural constant here was verified against SmolVLM-256M-Instruct's
config.json and the actual model.safetensors tensor names (see plan sec. 6).
"""
from dataclasses import dataclass, asdict, field
from typing import Literal, Optional

MODEL_ID = "HuggingFaceTB/SmolVLM-256M-Instruct"

# --- verified architecture facts (config.json) -------------------------------
N_VISION_LAYERS = 12      # vision_config.num_hidden_layers
N_TEXT_LAYERS = 30        # text_config.num_hidden_layers
VISION_HIDDEN = 768
TEXT_HIDDEN = 576
IMAGE_SEQ_LEN = 64        # ((512//16)**2) / scale_factor**2 = 1024/16, splitting OFF

# --- module scoping ----------------------------------------------------------
# NOTE: the vision tower uses `out_proj` / `fc1` / `fc2`, NOT `o_proj` / gate-up-down.
# A naive Llama-style target list silently skips the entire vision encoder.
# PEFT has no `idefics3` entry in its default target-module mapping, so these
# regexes are mandatory, not convenience.
VISION_RE = (r"model\.vision_model\.encoder\.layers\.\d+\."
             r"(self_attn\.(q_proj|k_proj|v_proj|out_proj)|mlp\.(fc1|fc2))")
TEXT_RE = (r"model\.text_model\.layers\.\d+\."
           r"(self_attn\.(q_proj|k_proj|v_proj|o_proj)|mlp\.(gate_proj|up_proj|down_proj))")
BOTH_RE = f"({VISION_RE})|({TEXT_RE})"

ARM_REGEX = {"V": VISION_RE, "L": TEXT_RE, "B": BOTH_RE}

# Expected number of matched Linear modules per arm -- asserted before training.
ARM_EXPECTED_MODULES = {"V": N_VISION_LAYERS * 6, "L": N_TEXT_LAYERS * 7}
ARM_EXPECTED_MODULES["B"] = ARM_EXPECTED_MODULES["V"] + ARM_EXPECTED_MODULES["L"]

# Prefixes used to bucket per-matrix stats into module groups for the analysis.
MODULE_GROUPS = {
    "vision": "model.vision_model.",
    "connector": "model.connector.",
    "text": "model.text_model.",
}

# --- fixed experimental settings (identical across every run) ----------------
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
EPOCHS = 3
PER_DEVICE_BS = 8
GRAD_ACCUM = 2            # effective batch 16
N_CHECKPOINTS = 4         # 3 intermediate + final, matching DoRA sec. 3.2
MAX_TEXT_TOKENS = 128
DEFAULT_SEED = 42
EXTRA_SEEDS = (1337, 2024)

N_TRAIN = 3000
N_EVAL = 1000
N_FORGET_PROBE = 500

# Learning rates. Full FT needs a much smaller LR than the adapters; the
# magnitude-only arm has ~83k parameters and tolerates a larger one.
DEFAULT_LR = {"ft": 1e-5, "lora": 2e-4, "dora": 2e-4, "magonly": 1e-3}
LR_SWEEP = {"ft": [5e-6, 1e-5, 3e-5],
            "lora": [1e-4, 2e-4, 5e-4],
            "dora": [1e-4, 2e-4, 5e-4],
            "magonly": [5e-4, 1e-3, 3e-3]}

Method = Literal["ft", "lora", "dora", "magonly", "magV_doraL"]
Arm = Literal["V", "L", "B"]
Dataset = Literal["pathvqa", "textvqa"]


@dataclass(frozen=True)
class RunSpec:
    method: Method
    arm: Arm
    dataset: Dataset
    seed: int = DEFAULT_SEED
    lr: Optional[float] = None
    tier: int = 0
    tag: str = ""

    @property
    def lr_resolved(self) -> float:
        if self.lr is not None:
            return self.lr
        # magV_doraL is a composite; drive it off the DoRA LR.
        return DEFAULT_LR.get(self.method, DEFAULT_LR["dora"])

    @property
    def run_id(self) -> str:
        base = f"{self.method}_{self.arm}_{self.dataset}_s{self.seed}_lr{self.lr_resolved:g}"
        return f"{base}_{self.tag}" if self.tag else base

    def to_dict(self):
        d = asdict(self)
        d["lr_resolved"] = self.lr_resolved
        d["run_id"] = self.run_id
        return d


DATASETS = ("pathvqa", "textvqa")


def build_grid(tiers=(1, 2, 3, 4)) -> list[RunSpec]:
    """The tiered run grid from plan sec. 8. Order matters: run tiers in sequence
    and stop wherever the Colab quota stops you -- the report degrades gracefully."""
    runs: list[RunSpec] = []

    if 1 in tiers:  # H1 + H3: the vision-tower replication of DoRA Figure 2
        for ds in DATASETS:
            for m in ("ft", "lora", "dora"):
                runs.append(RunSpec(m, "V", ds, tier=1))

    if 2 in tiers:  # H2: the same three methods on the language model
        for ds in DATASETS:
            for m in ("ft", "lora", "dora"):
                runs.append(RunSpec(m, "L", ds, tier=2))

    if 3 in tiers:  # accuracy claims + H5
        for ds in DATASETS:
            for m in ("ft", "lora", "dora"):
                runs.append(RunSpec(m, "B", ds, tier=3))
        for seed in EXTRA_SEEDS:   # seeds only where accuracy claims are made
            for m in ("lora", "dora"):
                runs.append(RunSpec(m, "B", "pathvqa", seed=seed, tier=3))

    if 4 in tiers:  # H4: the bonus, DoRA sec. 5.6 translated to vision
        for ds in DATASETS:
            runs.append(RunSpec("magonly", "V", ds, tier=4))
            runs.append(RunSpec("magV_doraL", "B", ds, tier=4))

    return runs

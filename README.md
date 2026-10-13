# Does DoRA's weight-decomposition claim hold in a VLM's vision encoder?

EN4554 — Deep Learning for Vision, Assignment 1, **Part 1**.
Model: `HuggingFaceTB/SmolVLM-256M-Instruct`. Compute: one free Colab T4.

## The question

DoRA ([Liu et al., arXiv 2402.09353](https://arxiv.org/abs/2402.09353), ICML 2024 oral)
is built on a measured claim: when you finetune, LoRA changes a weight's **magnitude** and
**direction** proportionally (correlation **+0.83**), while full finetuning does not
(**−0.62**); DoRA restores the FT-like pattern (**−0.31**) and with it, the authors argue,
FT-like learning capacity.

That measurement was taken on **VL-BART's query/value matrices** — a vision-*language*
model, but its **language** transformer. The paper's VLM experiments (VL-BART,
LLaVA-1.5-7B) put adapters only on language projections; both vision towers
(CLIP-ResNet101, CLIP ViT-L/336px) stay unadapted.

Meanwhile the downstream evidence from the vision side is flatly contradictory:

| | result |
|---|---|
| RepLoRA, ViT-B/16 on FGVC | DoRA **87.2** vs LoRA **84.8** (mean) |
| BananaVLM, LLaVA-1.5-7B on crop disease | LoRA **92.21%** vs DoRA **19.5%** in-domain (**+72.7 pp**) |

**Nobody has measured the magnitude/direction pattern on the vision side**, which is
exactly where DoRA's own motivation would have to hold for any of this to make sense.
This project measures it.

> A bounded search (arXiv API, the first 100 Semantic Scholar citations of DoRA, ~10
> targeted web searches) found no prior ΔM/ΔD analysis on a ViT or a VLM vision tower.
> That is "searched and did not find", **not** "does not exist" — Google Scholar's
> full-text cited-by was not reachable. This is an exploratory finding at one model size.

## Hypotheses

| | |
|---|---|
| **H1** | In SmolVLM's **vision encoder**, FT gives a negative ΔM–ΔD correlation, LoRA positive, DoRA near-FT — as the paper reports for a language transformer. |
| **H2** | The pattern **differs** between the vision encoder and the language model of the same VLM on the same data. |
| **H3** | In-domain, the vision encoder adapts mostly by **magnitude**; under domain shift it needs **direction**. |
| **H4** | *(bonus)* DoRA §5.6/Table 6's magnitude-only granularity transfers to vision: a per-output-channel gain of **82,944 params (0.03%)** keeps up in-domain but not under shift. |
| **H5** | **FT-likeness predicts accuracy** — DoRA's implicit assumption, never tested directly. |

## Datasets, and why

Contamination was checked against The Cauldron's config list (SmolVLM's training mixture),
whose README states it contains *only the train split of each sub-dataset*:

| | in The Cauldron? | role |
|---|---|---|
| **PathVQA** | **no** | domain-shift arm — the encoder *must* move. 3,000 train; 1,000 yes/no val for clean exact match against a ~50% majority floor |
| **TextVQA** | yes (train) | in-domain control; official val split is clean. 3,000 / 1,000 |
| VQAv2 | yes (train) | fixed 500-question val probe for forgetting |
| ~~VQA-RAD~~ | **yes, in full** (313 imgs / 1,793 QA = its entire train split) | **contaminated — deliberately unused** |

## Layout

```
src/config.py      run grid, module regexes, all fixed hyperparameters
src/data.py        loaders, contamination notes, assistant-only loss masking
src/model.py       the four methods; asserts every target-module count
src/decompose.py   DoRA Eq. 2/3/4, on both norm axes
src/train.py       one run -> one JSON record
src/evaluate.py    yes/no exact match, VQA soft accuracy, forgetting probe
src/analyze.py     correlations, FT-likeness, the three figures
run_grid.py        tiered driver, disconnect-safe
tests/test_sanity.py   seven checks to run before spending quota
notebook.ipynb     Colab entry point
```

## Running it

```bash
python tests/test_sanity.py                  # always first
python run_grid.py --tiers 1 --smoke         # 64/32 examples, proves the pipeline
python run_grid.py --tiers 1                 # ~2.5 h, independently submittable
python run_grid.py --tiers 2                 # then 3, then 4
python src/analyze.py                        # tables + figures
```

26 runs, ~13 h of T4 — **3–4 Colab sessions, not one sitting**. Tiers are the risk plan:
each adds one hypothesis, and any spec whose JSON already exists is skipped, so a
disconnect costs one run rather than the batch.

## Three details that silently ruin this experiment

1. **The norm axis.** PEFT computes `torch.linalg.norm(weight, dim=1)` on an
   `(out_features, in_features)` matrix → **one magnitude per output feature**. The paper's
   notation implies per *input* feature. The implementation is what DoRA actually trains,
   so it is the default here; the other axis is computed too, as a robustness check.
2. **Module names.** SmolVLM's vision tower uses `out_proj` / `fc1` / `fc2`, not
   `o_proj` / `gate_proj`. A Llama-style `target_modules` list silently skips the entire
   vision encoder — and PEFT has no `idefics3` default mapping, so the regex is mandatory.
   `tests/test_sanity.py` asserts all 72 vision Linears are matched.
3. **Magnitude-only is fragile.** PEFT bundles `lora_A`, `lora_B` and
   `lora_magnitude_vector` under one `adapter_layer_names`; `set_adapter()` re-enables all
   three. Freeze after `get_peft_model()`, never call `set_adapter` afterwards, and assert
   the parameter count immediately before `trainer.train()` — which `train.py` does.

With `lora_B` frozen at its zero init, these layers reduce to `y = (m/||W||) ⊙ (Wx) + b`,
a learned diagonal gain. That is a legitimate method close to **IA³** / **SSF** — it is
**not** DoRA, and the report says so.

## Other settings

`do_image_splitting=False` pins every image to exactly **64** tokens (splitting reaches
1088 at `longest_edge=2048`) — the largest wall-clock lever, applied uniformly so it is a
control, though it does cap fine text detail for the TextVQA arm. Weights load in fp32 with
**fp16 AMP** (the T4 has no hardware bf16); grad clipping at 1.0 guards the NaNs the
assignment warns about. r=16, α=32, 3 epochs, effective batch 16, seed 42.

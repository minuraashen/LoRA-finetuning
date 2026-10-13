"""One run = (method, arm, dataset, seed, lr) -> a single JSON record.

The JSON is the unit of durability: Colab disconnects cost one run, never the
batch, because run_grid.py skips any spec whose JSON already exists.
"""
from __future__ import annotations

import json
import os
import time

import torch
from transformers import Trainer, TrainingArguments, TrainerCallback, set_seed

import config as cfg
import data as data_mod
import decompose
import evaluate as ev
from model import build_model, load_base_model, load_processor

RESULTS_DIR = os.environ.get("RESULTS_DIR", "results")


class DecompositionCallback(TrainerCallback):
    """Records dM/dD at 3 intermediate checkpoints plus the final one, matching
    DoRA sec. 3.2 ("three intermediate steps and the final checkpoint")."""

    def __init__(self, model, base_weights, n_checkpoints=cfg.N_CHECKPOINTS):
        self.model = model
        self.base = base_weights
        self.n = n_checkpoints
        self.rows = []
        self._targets = None

    def on_train_begin(self, args, state, control, **kw):
        total = max(int(state.max_steps), 1)
        self._targets = {max(1, round(total * (i + 1) / self.n)): (i + 1) / self.n
                         for i in range(self.n)}

    def on_step_end(self, args, state, control, **kw):
        frac = self._targets.get(int(state.global_step))
        if frac is None:
            return
        self.rows.extend(decompose.collect_stats(
            self.base, self.model, int(state.global_step), frac))


def _loss_history(trainer):
    return [{"step": h["step"], "loss": h["loss"]}
            for h in trainer.state.log_history if "loss" in h]


def run(spec, processor=None, probe=None, overwrite=False, n_train=None, n_eval=None):
    out_path = os.path.join(RESULTS_DIR, f"{spec.run_id}.json")
    if os.path.exists(out_path) and not overwrite:
        print(f"[skip] {spec.run_id} (result exists)")
        return json.load(open(out_path, encoding="utf-8"))

    os.makedirs(RESULTS_DIR, exist_ok=True)
    t0 = time.time()
    set_seed(spec.seed)
    processor = processor or load_processor()

    train_ex, eval_ex, meta = data_mod.load_task(
        spec.dataset, n_train=n_train or cfg.N_TRAIN,
        n_eval=n_eval or cfg.N_EVAL, seed=spec.seed)

    model, info = build_model(spec.method, spec.arm, model=load_base_model())
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    # W_0 is captured once, before any optimizer step, for every tracked Linear.
    base_weights = decompose.snapshot_base_weights(model)
    cb = DecompositionCallback(model, base_weights)

    args = TrainingArguments(
        output_dir=os.path.join("ckpt", spec.run_id),
        per_device_train_batch_size=cfg.PER_DEVICE_BS,
        gradient_accumulation_steps=cfg.GRAD_ACCUM,
        num_train_epochs=cfg.EPOCHS,
        learning_rate=spec.lr_resolved,
        warmup_ratio=0.03,
        max_grad_norm=1.0,                 # fp16 + AdamW NaN guard
        fp16=torch.cuda.is_available(),    # T4 has no hardware bf16
        logging_steps=20,
        save_strategy="no",
        report_to=[],
        remove_unused_columns=False,
        label_names=["labels"],
        seed=spec.seed,
        dataloader_num_workers=2,
    )
    trainer = Trainer(model=model, args=args, train_dataset=train_ex,
                      data_collator=data_mod.SupervisedCollator(processor),
                      callbacks=[cb])

    # Assert the trainable set is what we think it is, immediately before training:
    # PEFT re-enables lora_A/lora_B on any set_adapter call, so this is the guard
    # for the magnitude-only arms.
    live = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert live == info["trainable_params"], (
        f"trainable params drifted: {live} vs {info['trainable_params']}")

    train_out = trainer.train()

    acc, preds = ev.evaluate_task(model, processor, eval_ex, yes_no=meta["yes_no_eval"])
    forget = ev.evaluate_forgetting(model, processor, probe) if probe else None

    record = {
        "spec": spec.to_dict(),
        "model_info": info,
        "data": meta,
        "train": {"loss_history": _loss_history(trainer),
                  "final_loss": float(train_out.training_loss),
                  "steps": int(trainer.state.global_step)},
        "accuracy": acc,
        "forgetting": forget,
        "decomposition": cb.rows,
        "sample_predictions": preds[:20],
        "wall_clock_s": round(time.time() - t0, 1),
    }
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(record, fh)
    print(f"[done] {spec.run_id}  headline={acc['headline']:.4f}  "
          f"loss={record['train']['final_loss']:.4f}  {record['wall_clock_s']:.0f}s")

    del trainer, model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    return record


@torch.no_grad()
def zero_shot_baselines(processor=None, probe=None, n_eval=None):
    """Untouched SmolVLM-256M on both eval sets -- the 'baseline' half of Step 6.5."""
    processor = processor or load_processor()
    model = load_base_model().to("cuda" if torch.cuda.is_available() else "cpu")
    out = {}
    for ds in cfg.DATASETS:
        _, eval_ex, meta = data_mod.load_task(ds, n_train=1, n_eval=n_eval or cfg.N_EVAL)
        acc, _ = ev.evaluate_task(model, processor, eval_ex, yes_no=meta["yes_no_eval"])
        out[ds] = acc
    if probe:
        out["vqav2_probe"] = ev.evaluate_forgetting(model, processor, probe)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, "baseline_zeroshot.json"), "w", encoding="utf-8") as fh:
        json.dump(out, fh)
    del model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    return out

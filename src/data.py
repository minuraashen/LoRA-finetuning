"""Datasets and collation.

Contamination status (verified against The Cauldron's config list and README,
which states "The Cauldron contains only the train split of each sub-datasets"):

  PathVQA   NOT in The Cauldron            -> clean, the domain-shift arm
  TextVQA   IS in The Cauldron (train)     -> seen; official val split is clean
  VQAv2     IS in The Cauldron (train)     -> val split clean, used as the
                                              forgetting probe
  VQA-RAD   IS in The Cauldron IN FULL     -> contaminated, deliberately unused

Subsets are drawn with a fixed seed and the selected indices are recorded in
every run's JSON, so "which 3000 examples" is reproducible and reportable, as
Step 4 of the assignment requires.
"""
from __future__ import annotations

import random

from datasets import load_dataset

import config as cfg

SPECS = {
    "pathvqa": dict(path="flaviagiammarino/path-vqa", train="train", eval="validation",
                    streaming=False, answer_key="answer"),
    "textvqa": dict(path="lmms-lab/textvqa", train="train", eval="validation",
                    streaming=True, answer_key="answers"),
}
PROBE = dict(path="lmms-lab/VQAv2", split="validation", streaming=True)

YES_NO = {"yes", "no"}


def _answers(example, key) -> list[str]:
    a = example[key]
    return [a] if isinstance(a, str) else list(a)


def _norm(s: str) -> str:
    return " ".join(str(s).strip().lower().replace(".", " ").split())


def _take(ds, n, seed, streaming):
    if streaming:
        ds = ds.shuffle(seed=seed, buffer_size=min(5000, max(n * 2, 1000)))
        return [ex for _, ex in zip(range(n), ds)], None
    idx = list(range(len(ds)))
    random.Random(seed).shuffle(idx)
    keep = sorted(idx[:n])
    # .select() is lazy; iterating it decodes each image exactly once.
    return list(ds.select(keep)), keep


def _yes_no_indices(ds, answer_key):
    """Indices of yes/no rows, read from the answer column only.

    Deliberately avoids ds.filter(), which would decode all 6k PathVQA images
    just to look at a string.
    """
    answers = ds[answer_key]
    return [i for i, a in enumerate(answers)
            if _norm(a if isinstance(a, str) else a[0]) in YES_NO]


def load_task(name: str, n_train=cfg.N_TRAIN, n_eval=cfg.N_EVAL,
              seed=cfg.DEFAULT_SEED, yes_no_eval=None):
    """Return (train_examples, eval_examples, meta).

    Each example: {"image", "question", "answers": [str, ...]}.

    For PathVQA the *training* set keeps every question type (a realistic
    finetune), while the *evaluation* set is restricted to the ~49.8% yes/no half
    so the headline metric is clean exact match against a ~50% majority floor.
    """
    spec = SPECS[name]
    yes_no_eval = (name == "pathvqa") if yes_no_eval is None else yes_no_eval

    train_raw = load_dataset(spec["path"], split=spec["train"], streaming=spec["streaming"])
    eval_raw = load_dataset(spec["path"], split=spec["eval"], streaming=spec["streaming"])

    if yes_no_eval:
        if spec["streaming"]:
            eval_raw = eval_raw.filter(
                lambda ex: _norm(_answers(ex, spec["answer_key"])[0]) in YES_NO)
        else:
            eval_raw = eval_raw.select(_yes_no_indices(eval_raw, spec["answer_key"]))

    train, train_idx = _take(train_raw, n_train, seed, spec["streaming"])
    evals, eval_idx = _take(eval_raw, n_eval, seed + 1, spec["streaming"])

    def conv(ex):
        return {"image": ex["image"].convert("RGB"),
                "question": ex["question"],
                "answers": _answers(ex, spec["answer_key"])}

    train, evals = [conv(e) for e in train], [conv(e) for e in evals]
    meta = {"dataset": name, "train_split": spec["train"], "eval_split": spec["eval"],
            "n_train": len(train), "n_eval": len(evals), "subset_seed": seed,
            "yes_no_eval": yes_no_eval, "streaming": spec["streaming"],
            # Step 4 requires stating exactly which examples were used.
            "train_indices": train_idx, "eval_indices": eval_idx}
    return train, evals, meta


def load_forgetting_probe(n=cfg.N_FORGET_PROBE, seed=cfg.DEFAULT_SEED):
    """A fixed 500-question VQAv2 validation probe, identical across all runs.

    VQAv2's *train* split is in The Cauldron; validation is not, so this measures
    general-VQA retention rather than memorisation.
    """
    ds = load_dataset(PROBE["path"], split=PROBE["split"], streaming=PROBE["streaming"])
    ds = ds.shuffle(seed=seed, buffer_size=5000)
    out = []
    for _, ex in zip(range(n), ds):
        out.append({"image": ex["image"].convert("RGB"),
                    "question": ex["question"],
                    "answers": _answers(ex, "answers")})
    return out


# --- prompt construction and collation ---------------------------------------

def build_prompt(processor, question: str) -> str:
    messages = [{"role": "user", "content": [{"type": "image"},
                                             {"type": "text", "text": question}]}]
    return processor.apply_chat_template(messages, add_generation_prompt=True)


class SupervisedCollator:
    """Builds batches with the loss masked to the assistant turn only.

    The prompt is tokenised once on its own to find its length; everything before
    the answer -- system/user text, and crucially the 64 image placeholder tokens
    -- is set to -100.
    """

    def __init__(self, processor, max_length=cfg.MAX_TEXT_TOKENS + cfg.IMAGE_SEQ_LEN + 16):
        self.p = processor
        self.max_length = max_length
        tok = processor.tokenizer
        self.pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
        self.eos = tok.eos_token

    def __call__(self, features):
        import torch
        input_ids, labels, pixel_values = [], [], []
        for f in features:
            prompt = build_prompt(self.p, f["question"])
            answer = f["answers"][0]
            enc_p = self.p(text=prompt, images=[f["image"]], return_tensors="pt")
            enc_f = self.p(text=prompt + answer + self.eos, images=[f["image"]],
                           return_tensors="pt")
            ids = enc_f["input_ids"][0][: self.max_length]
            lab = ids.clone()
            lab[: min(enc_p["input_ids"].shape[1], len(ids))] = -100
            lab[ids == self.pad_id] = -100
            input_ids.append(ids)
            labels.append(lab)
            pixel_values.append(enc_f["pixel_values"][0])

        width = max(len(x) for x in input_ids)
        batch_ids = torch.full((len(input_ids), width), self.pad_id, dtype=torch.long)
        batch_lab = torch.full((len(input_ids), width), -100, dtype=torch.long)
        attn = torch.zeros((len(input_ids), width), dtype=torch.long)
        for i, (ids, lab) in enumerate(zip(input_ids, labels)):
            batch_ids[i, : len(ids)] = ids
            batch_lab[i, : len(lab)] = lab
            attn[i, : len(ids)] = 1
        return {"input_ids": batch_ids, "attention_mask": attn,
                "labels": batch_lab, "pixel_values": torch.stack(pixel_values)}

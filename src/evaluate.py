"""Evaluation: task accuracy, the yes/no exact-match headline, and the
VQAv2 forgetting probe.

The assignment's Step 5 makes the decomposition analysis the *primary* metric;
accuracy is the second axis, needed for the baseline-vs-adapted table (Step 6.5)
and for the H5 test of whether FT-likeness predicts accuracy at all.
"""
from __future__ import annotations

import re

import torch

import config as cfg
from data import build_prompt

_ARTICLES = {"a", "an", "the"}
_PUNCT = re.compile(r"[^\w\s]")


def normalise(ans: str) -> str:
    s = _PUNCT.sub(" ", str(ans).lower())
    return " ".join(w for w in s.split() if w not in _ARTICLES)


def vqa_accuracy(pred: str, answers: list[str]) -> float:
    """Standard VQA soft accuracy: min(#humans who said it / 3, 1).

    Uses a simplified normaliser (lowercase, strip punctuation and articles)
    rather than the full official contraction/number tables -- adequate for the
    relative comparisons made here, and stated as such in the report.
    """
    p = normalise(pred)
    hits = sum(1 for a in answers if normalise(a) == p)
    return min(hits / 3.0, 1.0)


def exact_match(pred: str, answers: list[str]) -> float:
    p = normalise(pred)
    return float(any(normalise(a) == p for a in answers))


@torch.no_grad()
def generate(model, processor, examples, max_new_tokens=8, batch_size=16):
    model.eval()
    device = next(model.parameters()).device
    tok = processor.tokenizer
    saved_side, tok.padding_side = tok.padding_side, "left"
    preds = []
    try:
        for i in range(0, len(examples), batch_size):
            chunk = examples[i: i + batch_size]
            texts = [build_prompt(processor, ex["question"]) for ex in chunk]
            images = [[ex["image"]] for ex in chunk]
            enc = processor(text=texts, images=images, return_tensors="pt", padding=True)
            enc = {k: v.to(device) for k, v in enc.items()}
            out = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                                 num_beams=1, pad_token_id=tok.pad_token_id or tok.eos_token_id)
            new = out[:, enc["input_ids"].shape[1]:]
            preds.extend(t.strip() for t in tok.batch_decode(new, skip_special_tokens=True))
    finally:
        tok.padding_side = saved_side
    return preds


def score(examples, preds, yes_no: bool):
    """Returns the headline metric plus both underlying scores for transparency."""
    em = sum(exact_match(p, e["answers"]) for p, e in zip(preds, examples)) / len(examples)
    soft = sum(vqa_accuracy(p, e["answers"]) for p, e in zip(preds, examples)) / len(examples)
    out = {"exact_match": em, "vqa_soft_accuracy": soft,
           "headline": em if yes_no else soft}
    if yes_no:
        # Without the majority floor a yes/no number is uninterpretable.
        golds = [normalise(e["answers"][0]) for e in examples]
        out["majority_floor"] = max(golds.count("yes"), golds.count("no")) / len(golds)
        out["pred_yes_rate"] = sum(normalise(p) == "yes" for p in preds) / len(preds)
    return out


def evaluate_task(model, processor, examples, yes_no: bool, max_new_tokens=8):
    preds = generate(model, processor, examples, max_new_tokens=max_new_tokens)
    s = score(examples, preds, yes_no)
    s["n"] = len(examples)
    return s, preds


def evaluate_forgetting(model, processor, probe):
    preds = generate(model, processor, probe, max_new_tokens=8)
    s = score(probe, preds, yes_no=False)
    s["n"] = len(probe)
    return s

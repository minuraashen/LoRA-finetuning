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

import os
import pickle
import random

from datasets import load_dataset
from PIL import Image

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
    """Answers as a list of strings.

    PathVQA stores one string, TextVQA a list of 10 strings, and lmms-lab/VQAv2
    a list of dicts {answer, answer_confidence, answer_id}. Without unwrapping
    the dicts every VQAv2 prediction would score 0.
    """
    a = example[key]
    if isinstance(a, str):
        return [a]
    return [x["answer"] if isinstance(x, dict) else str(x) for x in a]


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

    cache_key = f"{name}_tr{n_train}_ev{n_eval}_s{seed}_yn{int(yes_no_eval)}"
    cached = _cache_load(cache_key)
    if cached is not None:
        return cached

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

    # Streaming draws (TextVQA) have no indices; their question_ids are recorded
    # instead, so runs made on different machines can be checked for using
    # the identical subset (analyze.check_subsets).
    def qids(rows):
        ids = [r.get("question_id") for r in rows]
        return None if all(i is None for i in ids) else ids

    train_qid, eval_qid = qids(train), qids(evals)

    def conv(ex):
        return {"image": _prep_image(ex["image"]),
                "question": ex["question"],
                "answers": _answers(ex, spec["answer_key"])}

    train, evals = [conv(e) for e in train], [conv(e) for e in evals]
    meta = {"dataset": name, "train_split": spec["train"], "eval_split": spec["eval"],
            "n_train": len(train), "n_eval": len(evals), "subset_seed": seed,
            "yes_no_eval": yes_no_eval, "streaming": spec["streaming"],
            # Step 4 requires stating exactly which examples were used.
            "train_indices": train_idx, "eval_indices": eval_idx,
            "train_question_ids": train_qid, "eval_question_ids": eval_qid}
    _cache_save(cache_key, (train, evals, meta))
    return train, evals, meta


# --- image preparation and the on-disk subset cache ---------------------------
# With do_image_splitting=False the processor squashes every image to one
# 512x512 tile, so nothing above 512 px on the long edge survives anyway.
# Downscaling once at load keeps 4,500 decoded images inside ~3 GB of RAM
# (raw TextVQA images would need ~9 GB) and is applied identically to every
# run, including the zero-shot baseline.
MAX_EDGE = 512
CACHE_DIR = os.environ.get("SUBSET_CACHE_DIR", os.path.join("hf_cache", "subsets"))


def _prep_image(img):
    img = img.convert("RGB")
    if max(img.size) > MAX_EDGE:
        img.thumbnail((MAX_EDGE, MAX_EDGE), Image.LANCZOS)
    return img


def _cache_load(key):
    """Streaming subsets re-download several GB per run; cache the drawn subset
    so every run of the grid sees byte-identical examples."""
    path = os.path.join(CACHE_DIR, key + ".pkl")
    if not os.path.exists(path):
        return None
    with open(path, "rb") as fh:
        return pickle.load(fh)


def _cache_save(key, obj):
    os.makedirs(CACHE_DIR, exist_ok=True)
    # Per-process temp name: two Kaggle shards may build the same subset at
    # once; os.replace then publishes one complete file atomically.
    tmp = os.path.join(CACHE_DIR, f"{key}.{os.getpid()}.tmp")
    with open(tmp, "wb") as fh:
        pickle.dump(obj, fh, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, os.path.join(CACHE_DIR, key + ".pkl"))


def load_forgetting_probe(n=cfg.N_FORGET_PROBE, seed=cfg.DEFAULT_SEED):
    """A fixed 500-question VQAv2 validation probe, identical across all runs.

    VQAv2's *train* split is in The Cauldron; validation is not, so this measures
    general-VQA retention rather than memorisation.
    """
    cache_key = f"vqav2probe_n{n}_s{seed}"
    cached = _cache_load(cache_key)
    if cached is not None:
        return cached
    ds = load_dataset(PROBE["path"], split=PROBE["split"], streaming=PROBE["streaming"])
    ds = ds.shuffle(seed=seed, buffer_size=5000)
    out = []
    for _, ex in zip(range(n), ds):
        out.append({"image": _prep_image(ex["image"]),
                    "question": ex["question"],
                    "answers": _answers(ex, "answers")})
    _cache_save(cache_key, out)
    return out


# --- prompt construction and collation ---------------------------------------

# The standard short-answer VQA instruction (also used for SmolVLM's own VQA
# evals). Without it the zero-shot model often answers in full sentences, which
# scores 0 under exact match and makes the baseline-vs-adapted gap meaningless.
# Used identically for training, evaluation and the zero-shot baseline.
SHORT_ANSWER = "\nAnswer the question using a single word or phrase."


def build_prompt(processor, question: str) -> str:
    messages = [{"role": "user", "content": [{"type": "image"},
                                             {"type": "text", "text": question + SHORT_ANSWER}]}]
    return processor.apply_chat_template(messages, add_generation_prompt=True)


def format_target(answer: str) -> str:
    """Write a dataset answer in SmolVLM's own answer style: ' Yes.', ' Pink.'.

    The datasets store answers lowercase with no punctuation ('yes'), but the
    model answers with a capital and a full stop. Measured on the untouched
    model, the lowercase form costs ~5.7 extra nats on the answer token
    (12.44 vs 6.73) -- a content-free penalty that would dominate the early
    gradient, so the vision tower would mostly learn casing rather than the
    images, contaminating the dM/dD measurement. Scoring is unaffected:
    evaluate.normalise lowercases and strips punctuation.
    """
    a = str(answer).strip()
    if not a:
        return a
    a = a[0].upper() + a[1:]
    return a if a[-1] in ".!?" else a + "."


class SupervisedCollator:
    """Builds batches with the loss masked to the assistant turn only.

    The answer is appended as " " + answer + eos: the chat template ends in
    "Assistant:" and SmolVLM's own outputs begin with a space token, so this
    matches the pretrained format. Because the space starts a new pre-token,
    the answer tokenises identically on its own, so its token count gives the
    prompt/answer boundary without running the image processor a second time.
    Everything before the answer -- user text and the 64 image placeholder
    tokens -- is set to -100.
    """

    def __init__(self, processor, max_length=cfg.MAX_TEXT_TOKENS + cfg.IMAGE_SEQ_LEN + 16):
        self.p = processor
        self.max_length = max_length
        tok = processor.tokenizer
        self.tok = tok
        self.pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
        self.eos = tok.eos_token

    def __call__(self, features):
        import torch
        input_ids, labels, pixel_values = [], [], []
        for f in features:
            prompt = build_prompt(self.p, f["question"])
            target = " " + format_target(f["answers"][0]) + self.eos
            n_target = len(self.tok(target, add_special_tokens=False)["input_ids"])
            enc_f = self.p(text=prompt + target, images=[f["image"]], return_tensors="pt")
            full = enc_f["input_ids"][0]
            prompt_len = len(full) - n_target
            ids = full[: self.max_length]
            lab = ids.clone()
            lab[: min(prompt_len, len(ids))] = -100
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

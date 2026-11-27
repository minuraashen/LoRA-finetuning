"""Tiered driver. Disconnect-safe: any spec whose JSON already exists is skipped.

    python run_grid.py --tiers 1            # ~2.5 h, independently submittable
    python run_grid.py --tiers 1 2
    python run_grid.py --smoke              # 64/32 examples, proves the pipeline
    python run_grid.py --list

Run tiers in order and stop wherever the Colab quota stops you; the report
degrades gracefully (see plan sec. 8/10).
"""
from __future__ import annotations

import argparse
import os
import sys
import time
import traceback

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

import config as cfg  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tiers", type=int, nargs="+", default=[1])
    ap.add_argument("--methods", nargs="*", default=None)
    ap.add_argument("--datasets", nargs="*", default=None)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--no-probe", action="store_true",
                    help="skip the VQAv2 forgetting probe (saves download + eval time)")
    ap.add_argument("--smoke", action="store_true",
                    help="tiny subsets: proves the pipeline end to end in minutes")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--shard", default=None, metavar="K/N",
                    help="run only shard K of N (0-based). Shards are dealt round-robin "
                         "over the FULL 26-run grid, so every machine computes the same "
                         "split no matter which --tiers it asks for")
    ap.add_argument("--max-hours", type=float, default=None,
                    help="do not START a new run after this many hours (Kaggle's 12 h cap)")
    args = ap.parse_args()

    grid = cfg.build_grid(tuple(args.tiers))
    shard_k = 0
    if args.shard:
        shard_k, shard_n = (int(x) for x in args.shard.split("/"))
        assert 0 <= shard_k < shard_n, f"bad --shard {args.shard}"
        full = cfg.build_grid((1, 2, 3, 4))
        mine = {r.run_id for i, r in enumerate(full) if i % shard_n == shard_k}
        grid = [r for r in grid if r.run_id in mine]
    if args.methods:
        grid = [r for r in grid if r.method in args.methods]
    if args.datasets:
        grid = [r for r in grid if r.dataset in args.datasets]

    if args.list:
        for r in grid:
            print(f"tier {r.tier}  {r.run_id}")
        if args.shard:
            print(f"(shard {args.shard})")
        print(f"\n{len(grid)} runs")
        return

    import train as train_mod
    from data import load_forgetting_probe
    from model import load_processor

    if args.smoke:
        # Smoke records share run_ids with the real runs; written to the real
        # folder they would make the real run "already done" and be skipped.
        train_mod.RESULTS_DIR = train_mod.RESULTS_DIR.rstrip("/\\") + "_smoke"
        print(f"[smoke] writing to {train_mod.RESULTS_DIR}")

    processor = load_processor()
    probe = None if (args.no_probe or args.smoke) else load_forgetting_probe()
    n_train, n_eval = (64, 32) if args.smoke else (None, None)

    # The baseline is needed once overall, so only shard 0 (or an unsharded run) does it.
    if not args.smoke and shard_k == 0 and not os.path.exists(
            os.path.join(train_mod.RESULTS_DIR, "baseline_zeroshot.json")):
        print("== zero-shot baselines ==")
        train_mod.zero_shot_baselines(processor=processor, probe=probe)

    t_start = time.time()
    failures, deferred = [], []
    for i, spec in enumerate(grid, 1):
        hours = (time.time() - t_start) / 3600
        if args.max_hours is not None and hours > args.max_hours and not os.path.exists(
                os.path.join(train_mod.RESULTS_DIR, f"{spec.run_id}.json")):
            deferred.append(spec.run_id)
            continue
        print(f"\n===== [{i}/{len(grid)}] tier {spec.tier}  {spec.run_id} =====")
        try:
            train_mod.run(spec, processor=processor, probe=probe,
                          overwrite=args.overwrite, n_train=n_train, n_eval=n_eval)
        except Exception:
            traceback.print_exc()
            failures.append(spec.run_id)
            print(f"[FAIL] {spec.run_id} -- continuing")

    done = len(grid) - len(failures) - len(deferred)
    print(f"\n{done}/{len(grid)} runs completed")
    if failures:
        print("failed:", ", ".join(failures))
    if deferred:
        print(f"not started (--max-hours {args.max_hours} reached), rerun this shard later:",
              ", ".join(deferred))


if __name__ == "__main__":
    main()

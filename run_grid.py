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
    args = ap.parse_args()

    grid = cfg.build_grid(tuple(args.tiers))
    if args.methods:
        grid = [r for r in grid if r.method in args.methods]
    if args.datasets:
        grid = [r for r in grid if r.dataset in args.datasets]

    if args.list:
        for r in grid:
            print(f"tier {r.tier}  {r.run_id}")
        print(f"\n{len(grid)} runs")
        return

    import train as train_mod
    from data import load_forgetting_probe
    from model import load_processor

    processor = load_processor()
    probe = None if (args.no_probe or args.smoke) else load_forgetting_probe()
    n_train, n_eval = (64, 32) if args.smoke else (None, None)

    if not args.smoke and not os.path.exists(
            os.path.join(train_mod.RESULTS_DIR, "baseline_zeroshot.json")):
        print("== zero-shot baselines ==")
        train_mod.zero_shot_baselines(processor=processor, probe=probe)

    failures = []
    for i, spec in enumerate(grid, 1):
        print(f"\n===== [{i}/{len(grid)}] tier {spec.tier}  {spec.run_id} =====")
        try:
            train_mod.run(spec, processor=processor, probe=probe,
                          overwrite=args.overwrite, n_train=n_train, n_eval=n_eval)
        except Exception:
            traceback.print_exc()
            failures.append(spec.run_id)
            print(f"[FAIL] {spec.run_id} -- continuing")

    print(f"\n{len(grid) - len(failures)}/{len(grid)} runs completed")
    if failures:
        print("failed:", ", ".join(failures))


if __name__ == "__main__":
    main()

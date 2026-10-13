"""Analysis: correlations, FT-likeness, and the three report figures.

H5's statistic is defined here *before* any data exists, deliberately:
    r(run, group)   Pearson correlation of (dD, dM) over all (layer, checkpoint)
                    points -- the paper's own statistic
    FT-likeness     -|r_method - r_FT|, where the FT reference is the run with
                    the same dataset and the same arm
then Spearman(FT-likeness, headline accuracy) across runs.
"""
from __future__ import annotations

import glob
import json
import os

import numpy as np
import pandas as pd
from scipy import stats

# DoRA paper sec. 4.1, measured on VL-BART query matrices (a LANGUAGE transformer).
PAPER_R = {"ft": -0.62, "lora": 0.83, "dora": -0.31}

RESULTS_DIR = os.environ.get("RESULTS_DIR", "results")
FIG_DIR = os.environ.get("FIG_DIR", "figures")


def load_runs(results_dir=RESULTS_DIR):
    records = []
    for path in sorted(glob.glob(os.path.join(results_dir, "*.json"))):
        if os.path.basename(path).startswith(("baseline", "h5", "summary")):
            continue
        with open(path, encoding="utf-8") as fh:
            records.append(json.load(fh))
    return records


def decomposition_frame(records) -> pd.DataFrame:
    frames = []
    for rec in records:
        df = pd.DataFrame(rec["decomposition"])
        if df.empty:
            continue
        for key in ("method", "arm", "dataset", "seed", "run_id"):
            df[key] = rec["spec"][key]
        df["headline"] = rec["accuracy"]["headline"]
        frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def correlations(df, axis="out", matrices=("q_proj", "v_proj")) -> pd.DataFrame:
    """One (dD, dM) correlation per (run, module group).

    Restricted to query/value matrices by default, which is exactly what DoRA
    Figure 2 plots; pass matrices=None to use every tracked matrix.
    """
    sub = df[df.axis == axis]
    if matrices:
        sub = sub[sub.matrix.isin(matrices)]
    rows = []
    for (run_id, group), g in sub.groupby(["run_id", "group"]):
        g = g[(g.delta_m.abs() > 0) | (g.delta_d.abs() > 0)]
        if len(g) < 3:
            continue
        r, p = stats.pearsonr(g.delta_d, g.delta_m)
        slope = np.polyfit(g.delta_d, g.delta_m, 1)[0]
        first = g.iloc[0]
        rows.append({"run_id": run_id, "group": group, "axis": axis,
                     "method": first.method, "arm": first.arm,
                     "dataset": first.dataset, "seed": first.seed,
                     "r": r, "p_value": p, "slope": slope, "n_points": len(g),
                     "mean_delta_m": g.delta_m.mean(), "mean_delta_d": g.delta_d.mean(),
                     "headline": first.headline})
    return pd.DataFrame(rows)


def ft_likeness(corr: pd.DataFrame) -> pd.DataFrame:
    """-|r_method - r_FT| against the FT run of the same dataset, arm and group."""
    if corr.empty or not (corr.method == "ft").any():
        corr = corr.copy()
        corr["r_ft"] = np.nan
        corr["ft_likeness"] = np.nan
        return corr
    ref = (corr[corr.method == "ft"]
           .set_index(["dataset", "arm", "group", "axis"])["r"].rename("r_ft"))
    ref = ref[~ref.index.duplicated()]
    out = corr.join(ref, on=["dataset", "arm", "group", "axis"])
    out["ft_likeness"] = -(out["r"] - out["r_ft"]).abs()
    return out


def h5_test(corr_with_likeness: pd.DataFrame, group="vision") -> dict:
    """Does FT-likeness predict accuracy? With ~8-12 points this is suggestive,
    not conclusive -- the report must say so, and a null result is a result."""
    sub = corr_with_likeness[(corr_with_likeness.group == group)
                             & corr_with_likeness.ft_likeness.notna()
                             & (corr_with_likeness.method != "ft")]
    if len(sub) < 4:
        return {"group": group, "n": int(len(sub)), "verdict": "insufficient runs"}
    rho, p = stats.spearmanr(sub.ft_likeness, sub.headline)
    return {"group": group, "n": int(len(sub)), "spearman_rho": float(rho),
            "p_value": float(p),
            "verdict": "suggestive" if p < 0.1 else "no detectable relationship"}


def summary_table(records) -> pd.DataFrame:
    rows = []
    for rec in records:
        s, a = rec["spec"], rec["accuracy"]
        rows.append({"run_id": s["run_id"], "method": s["method"], "arm": s["arm"],
                     "dataset": s["dataset"], "seed": s["seed"],
                     "trainable_params": rec["model_info"]["trainable_params"],
                     "trainable_pct": round(rec["model_info"]["trainable_pct"], 4),
                     "headline": a["headline"], "exact_match": a["exact_match"],
                     "vqa_soft": a["vqa_soft_accuracy"],
                     "majority_floor": a.get("majority_floor"),
                     "final_loss": rec["train"]["final_loss"],
                     "forget_vqav2": (rec["forgetting"] or {}).get("vqa_soft_accuracy"),
                     "minutes": round(rec["wall_clock_s"] / 60, 1)})
    return pd.DataFrame(rows).sort_values(["dataset", "arm", "method", "seed"])


# --- figures -----------------------------------------------------------------

def _scatter(ax, g, title, paper_r=None):
    import matplotlib.pyplot as plt
    markers = ["o", "s", "^", "D"]
    fracs = sorted(g.frac.unique())
    layers = sorted(g.layer.unique())
    cmap = plt.get_cmap("viridis")
    for i, frac in enumerate(fracs):
        for layer in layers:
            sel = g[(g.frac == frac) & (g.layer == layer)]
            if sel.empty:
                continue
            ax.scatter(sel.delta_d, sel.delta_m, s=26,
                       marker=markers[i % len(markers)],
                       color=cmap(layer / max(len(layers) - 1, 1)),
                       edgecolors="none", alpha=0.85)
    if len(g) >= 3:
        r, _ = stats.pearsonr(g.delta_d, g.delta_m)
        b, a = np.polyfit(g.delta_d, g.delta_m, 1)
        xs = np.linspace(g.delta_d.min(), g.delta_d.max(), 20)
        ax.plot(xs, b * xs + a, color="crimson", lw=1.6)
        label = "r = {:+.2f}".format(r)
        if paper_r is not None:
            label += "\n(paper, LM: {:+.2f})".format(paper_r)
        ax.text(0.04, 0.96, label, transform=ax.transAxes, va="top", fontsize=9)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("delta D")
    ax.set_ylabel("delta M")


def figure1_pattern(df, dataset="pathvqa", axis="out",
                    matrices=("q_proj", "v_proj"), out=None):
    """The DoRA Figure 2 replication: {FT, LoRA, DoRA} x {vision encoder, LM}.

    This single figure answers H1 (does the pattern hold in the vision tower)
    and H2 (do the two modules of one VLM differ).
    """
    import matplotlib.pyplot as plt
    methods, groups = ["ft", "lora", "dora"], ["vision", "text"]
    sub = df[(df.dataset == dataset) & (df.axis == axis) & df.matrix.isin(matrices)]
    fig, axes = plt.subplots(len(groups), len(methods), figsize=(12, 7))
    for i, group in enumerate(groups):
        arm = "V" if group == "vision" else "L"
        for j, method in enumerate(methods):
            g = sub[(sub.method == method) & (sub.arm == arm) & (sub.group == group)]
            ax = axes[i, j]
            if g.empty:
                ax.set_axis_off()
                ax.set_title("{} / {}: no run".format(method, group), fontsize=9)
                continue
            _scatter(ax, g, "{} - {} encoder".format(method.upper(), group),
                     paper_r=PAPER_R.get(method))
    fig.suptitle("Magnitude vs direction updates, SmolVLM-256M on {} "
                 "(axis={}, {})".format(dataset, axis, "/".join(matrices)))
    fig.tight_layout()
    out = out or os.path.join(FIG_DIR, "fig1_pattern_{}_{}.png".format(dataset, axis))
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


def figure2_domain_shift(df, axis="out", group="vision", out=None):
    """H3: is in-domain adaptation magnitude-dominated and OOD direction-dominated?"""
    import matplotlib.pyplot as plt
    sub = (df[(df.axis == axis) & (df.group == group) & (df.frac == 1.0)]
           .groupby(["dataset", "method", "arm"])[["delta_m", "delta_d"]]
           .mean().reset_index())
    fig, ax = plt.subplots(figsize=(7, 5))
    for ds, marker in zip(sorted(sub.dataset.unique()), ["o", "^"]):
        g = sub[sub.dataset == ds]
        ax.scatter(g.delta_d, g.delta_m, marker=marker, s=90, label=ds)
        for _, row in g.iterrows():
            ax.annotate("{}/{}".format(row.method, row.arm),
                        (row.delta_d, row.delta_m), fontsize=8,
                        xytext=(4, 4), textcoords="offset points")
    ax.set_xlabel("mean delta D (final checkpoint)")
    ax.set_ylabel("mean delta M (final checkpoint)")
    ax.set_title("{} encoder: in-domain (textvqa) vs domain shift (pathvqa)".format(group))
    ax.legend()
    fig.tight_layout()
    out = out or os.path.join(FIG_DIR, "fig2_domain_shift_{}.png".format(group))
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


def figure3_params_vs_accuracy(summary, baseline=None, out=None):
    """H4: does magnitude-only (~83k params) keep up, and where does it stop?"""
    import matplotlib.pyplot as plt
    datasets = sorted(summary.dataset.unique())
    fig, axes = plt.subplots(1, max(len(datasets), 2), figsize=(12, 5))
    axes = np.atleast_1d(axes)
    for ax, ds in zip(axes, datasets):
        g = summary[summary.dataset == ds]
        for method, marker in zip(sorted(g.method.unique()), ["o", "s", "^", "D", "P"]):
            h = g[g.method == method]
            ax.scatter(h.trainable_params.clip(lower=1), h.headline,
                       marker=marker, s=80, label=method)
        if baseline and ds in baseline:
            ax.axhline(baseline[ds]["headline"], ls="--", c="grey", lw=1,
                       label="zero-shot")
        floor = g.majority_floor.dropna()
        if len(floor):
            ax.axhline(floor.iloc[0], ls=":", c="black", lw=1, label="majority floor")
        ax.set_xscale("log")
        ax.set_xlabel("trainable parameters (log)")
        ax.set_ylabel("headline accuracy")
        ax.set_title(ds)
        ax.legend(fontsize=8)
    for ax in axes[len(datasets):]:
        ax.set_axis_off()
    fig.tight_layout()
    out = out or os.path.join(FIG_DIR, "fig3_params_vs_accuracy.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


def run_all(results_dir=RESULTS_DIR):
    records = load_runs(results_dir)
    if not records:
        raise SystemExit("no result JSONs in {}".format(results_dir))
    df = decomposition_frame(records)
    summary = summary_table(records)
    corr = pd.concat([correlations(df, axis=a) for a in ("out", "in")],
                     ignore_index=True)
    corr = ft_likeness(corr)

    baseline_path = os.path.join(results_dir, "baseline_zeroshot.json")
    baseline = (json.load(open(baseline_path, encoding="utf-8"))
                if os.path.exists(baseline_path) else None)

    figs = []
    for ds in summary.dataset.unique():
        try:
            figs.append(figure1_pattern(df, dataset=ds))
        except Exception as exc:                      # a tier may not have run yet
            print("[fig1/{}] skipped: {}".format(ds, exc))
    for name, fn in (("fig2", lambda: figure2_domain_shift(df)),
                     ("fig3", lambda: figure3_params_vs_accuracy(summary, baseline))):
        try:
            figs.append(fn())
        except Exception as exc:
            print("[{}] skipped: {}".format(name, exc))

    h5 = {g: h5_test(corr[corr.axis == "out"], group=g) for g in ("vision", "text")}
    summary.to_csv(os.path.join(results_dir, "summary.csv"), index=False)
    corr.to_csv(os.path.join(results_dir, "correlations.csv"), index=False)
    with open(os.path.join(results_dir, "h5.json"), "w", encoding="utf-8") as fh:
        json.dump(h5, fh, indent=2)
    return {"summary": summary, "correlations": corr, "h5": h5, "figures": figs}


if __name__ == "__main__":
    res = run_all()
    pd.set_option("display.width", 200)
    print(res["summary"].to_string(index=False))
    print()
    cols = ["run_id", "group", "r", "slope", "ft_likeness", "headline"]
    print(res["correlations"].query("axis == 'out'")[cols].to_string(index=False))
    print()
    print(json.dumps(res["h5"], indent=2))
    print("figures:", res["figures"])

"""
Generate visualizations and explanations from a trained EBM.

Outputs:
  plots/shape_<feature>.png  — global shape function per feature
  plots/local_sample.png     — local explanation for one sample row
  plots/teacher_vs_student.png — fidelity scatter (distillation mode only)
  report.html                — interactive interpret.ml dashboard

Run after train_ebm.py.
"""
from __future__ import annotations

from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from interpret import preserve

HERE = Path(__file__).parent
PLAYGROUND = HERE.parent
MODEL_PATH = HERE / "ebm_model.pkl"
PLOTS = HERE / "plots"
REPORT = HERE / "report.html"
IDENTITY_CSV = PLAYGROUND / "train_identity.csv"
LGBM_PATH = PLAYGROUND / "lgbm_tuning.pkl"


def plot_shape(ebm, feature_idx: int, out: Path) -> None:
    """Render a single feature's shape function to PNG."""
    explanation = ebm.explain_global()
    data = explanation.data(feature_idx)
    name = explanation.feature_names[feature_idx]
    ftype = explanation.feature_types[feature_idx]

    fig, ax = plt.subplots(figsize=(8, 4))

    if ftype == "continuous":
        x = data["names"]
        scores = data["scores"]
        upper = data.get("upper_bounds")
        lower = data.get("lower_bounds")
        # Bin edges → step plot on bin centers.
        if len(x) == len(scores) + 1:
            centers = [(x[i] + x[i + 1]) / 2 for i in range(len(scores))]
        else:
            centers = list(x)
        ax.plot(centers, scores, color="#2563eb", linewidth=2, label="contribution")
        if upper is not None and lower is not None:
            ax.fill_between(centers, lower, upper, color="#2563eb", alpha=0.15)
        ax.axhline(0, color="gray", linewidth=0.7, linestyle="--")
        ax.set_xlabel(name)
    else:
        x = list(map(str, data["names"]))
        scores = data["scores"]
        # Truncate long categorical lists for readability.
        if len(x) > 20:
            order = np.argsort(np.abs(scores))[::-1][:20]
            x = [x[i] for i in order]
            scores = [scores[i] for i in order]
        ax.bar(range(len(x)), scores, color="#2563eb")
        ax.set_xticks(range(len(x)))
        ax.set_xticklabels(x, rotation=45, ha="right", fontsize=8)
        ax.axhline(0, color="gray", linewidth=0.7, linestyle="--")

    ax.set_ylabel("Contribution to log-odds(fraud)")
    ax.set_title(f"Shape function: {name}")
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)


def plot_local(ebm, X_sample: pd.DataFrame, out: Path) -> None:
    """Per-prediction breakdown for one row."""
    local = ebm.explain_local(X_sample, [1])  # predict fraud=1
    data = local.data(0)
    names = data["names"]
    scores = data["scores"]
    values = data["values"]

    pairs = sorted(zip(names, scores, values), key=lambda r: abs(r[1]), reverse=True)[:15]
    names, scores, values = zip(*pairs)
    labels = [f"{n} = {v}" for n, v in zip(names, values)]
    colors = ["#dc2626" if s > 0 else "#16a34a" for s in scores]

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(range(len(labels))[::-1], scores, color=colors)
    ax.set_yticks(range(len(labels))[::-1])
    ax.set_yticklabels(labels, fontsize=9)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Contribution to log-odds(fraud)")
    ax.set_title("Local explanation (top 15 features)")
    ax.grid(axis="x", linestyle="--", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)


def plot_fidelity(student_proba, teacher_proba, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(teacher_proba, student_proba, s=4, alpha=0.25, color="#2563eb")
    ax.plot([0, 1], [0, 1], color="gray", linestyle="--", linewidth=0.8)
    corr = float(np.corrcoef(teacher_proba, student_proba)[0, 1])
    ax.set_xlabel("Teacher (LightGBM) P(fraud)")
    ax.set_ylabel("Student (EBM) P(fraud)")
    ax.set_title(f"Teacher vs Student fidelity\nPearson r = {corr:.3f}")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(linestyle="--", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)


def main() -> None:
    PLOTS.mkdir(exist_ok=True)
    bundle = joblib.load(MODEL_PATH)
    ebm = bundle["model"]
    feature_names = bundle["feature_names"]
    print(f"[load] EBM with {len(feature_names)} features")

    # --- Global: shape function per single feature (skip pairwise terms).
    explanation = ebm.explain_global()
    n_singles = len(feature_names)
    print(
        f"[plot] {len(explanation.feature_names)} terms total "
        f"({n_singles} singletons + {len(explanation.feature_names) - n_singles} "
        "pairwise interactions — only singletons rendered)"
    )
    for i, fname in enumerate(explanation.feature_names[:n_singles]):
        safe = fname.replace(" ", "_").replace("&", "x").replace("/", "_")
        out = PLOTS / f"shape_{i:02d}_{safe}.png"
        plot_shape(ebm, i, out)
    print(f"[plot] wrote {n_singles} shape plots to {PLOTS}/")

    # --- Local: explanation for one sample row.
    print("[plot] generating local explanation ...")
    identity = pd.read_csv(IDENTITY_CSV, nrows=200)
    sample = identity.reindex(columns=feature_names).iloc[[0]]
    plot_local(ebm, sample, PLOTS / "local_sample.png")
    print(f"[plot] wrote {PLOTS}/local_sample.png")

    # --- Fidelity (distillation only): teacher vs student.
    print("[plot] computing fidelity vs teacher ...")
    teacher = joblib.load(LGBM_PATH)
    expected = list(teacher.feature_name_)
    full = pd.DataFrame(0.0, index=identity.index, columns=expected)
    for col in expected:
        if col in identity.columns:
            full[col] = identity[col]
    obj_cols = full.select_dtypes(include=["object"]).columns
    for c in obj_cols:
        full[c] = pd.factorize(full[c])[0]
    full = full.fillna(0.0)
    teacher_proba = teacher.predict_proba(full)[:, 1]

    student_X = identity.reindex(columns=feature_names)
    student_proba = ebm.predict_proba(student_X)[:, 1]
    plot_fidelity(student_proba, teacher_proba, PLOTS / "teacher_vs_student.png")
    print(f"[plot] wrote {PLOTS}/teacher_vs_student.png")

    # --- Interactive HTML dashboard via interpret.preserve.
    try:
        preserve(explanation, file_name=str(REPORT))
        print(f"[plot] wrote {REPORT} (open in browser for interactive dashboard)")
    except Exception as e:
        print(f"[warn] could not write HTML report: {e}")

    print("\n[done] all artifacts saved.")


if __name__ == "__main__":
    main()

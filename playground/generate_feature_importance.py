"""
Extract feature importance from lgbm_tuning.pkl and save documentation artifacts.

Outputs:
  - feature_importance.csv : ranked 424 features (split, gain, gain_pct, cumulative_pct)
  - feature_importance.png : horizontal bar chart of top-30 by gain
"""
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd

HERE = Path(__file__).parent
MODEL_PATH = HERE / "lgbm_tuning.pkl"
CSV_OUT = HERE / "feature_importance.csv"
PNG_OUT = HERE / "feature_importance.png"
TOP_N = 30


def main() -> None:
    model = joblib.load(MODEL_PATH)
    booster = model.booster_

    df = pd.DataFrame({
        "feature": booster.feature_name(),
        "split": booster.feature_importance(importance_type="split"),
        "gain": booster.feature_importance(importance_type="gain"),
    })
    df["gain_pct"] = df["gain"] / df["gain"].sum() * 100
    df = df.sort_values("gain", ascending=False).reset_index(drop=True)
    df.insert(0, "rank", df.index + 1)
    df["cumulative_pct"] = df["gain_pct"].cumsum().round(2)
    df["gain_pct"] = df["gain_pct"].round(2)
    df["gain"] = df["gain"].round(2)

    df.to_csv(CSV_OUT, index=False)
    print(f"Wrote {CSV_OUT} ({len(df)} rows)")

    top = df.head(TOP_N).iloc[::-1]
    fig, ax = plt.subplots(figsize=(9, 10))
    ax.barh(top["feature"], top["gain_pct"], color="#2563eb")
    ax.set_xlabel("Gain (%)")
    ax.set_title(f"Top {TOP_N} feature importance — LightGBM fraud model")
    ax.grid(axis="x", linestyle="--", alpha=0.4)
    for i, (feat, pct) in enumerate(zip(top["feature"], top["gain_pct"])):
        ax.text(pct + 0.05, i, f"{pct:.2f}%", va="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(PNG_OUT, dpi=140)
    print(f"Wrote {PNG_OUT}")

    cum = df.head(TOP_N)["gain_pct"].sum()
    zero = (df["gain"] == 0).sum()
    print(f"Top-{TOP_N} cumulative gain: {cum:.1f}%")
    print(f"Features with zero gain (unused): {zero} / {len(df)}")


if __name__ == "__main__":
    main()

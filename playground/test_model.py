"""
Smoke test for lgbm_tuning.pkl.

Usage:
  python test_model.py                  # synthetic sample (1 row)
  python test_model.py path/to/data.csv # rows from CSV (header = feature names)

Output: fraud probability (class 1) per row.
"""
import sys
import joblib
import pandas as pd

MODEL_PATH = "lgbm_tuning.pkl"


def load_model():
    model = joblib.load(MODEL_PATH)
    feats = list(model.feature_name_)
    return model, feats


def synthetic_row(features):
    """Build 1 dummy row with 0.0 for all features (smoke test only)."""
    return pd.DataFrame([{f: 0.0 for f in features}])


def main():
    model, features = load_model()
    print(f"Model loaded: {type(model).__name__}")
    print(f"Expected features: {len(features)}")
    print(f"Classes: {list(model.classes_)} (1 = fraud)\n")

    if len(sys.argv) > 1:
        df = pd.read_csv(sys.argv[1])
        missing = [c for c in features if c not in df.columns]
        if missing:
            print(f"WARN: {len(missing)} features missing, filled with 0.0 (e.g. {missing[:5]})")
            missing_df = pd.DataFrame(0.0, index=df.index, columns=missing)
            df = pd.concat([df, missing_df], axis=1)
        X = df[features].copy()
        # Encode object columns to integers (smoke-test only — production must
        # use the LabelEncoder fitted at training time).
        obj_cols = X.select_dtypes(include=["object"]).columns
        if len(obj_cols):
            print(f"Encoding {len(obj_cols)} object columns -> int (factorize)")
            for c in obj_cols:
                X[c] = pd.factorize(X[c])[0]
        X = X.fillna(0.0)
    else:
        print("Using synthetic data (1 dummy row, all features = 0.0)")
        X = synthetic_row(features)

    proba = model.predict_proba(X)[:, 1]
    pred = model.predict(X)

    out = pd.DataFrame({
        "row": range(len(X)),
        "fraud_proba": proba.round(4),
        "predicted_label": pred,
    })
    print(out.to_string(index=False))
    print(f"\nSummary: {int((pred == 1).sum())} fraud / {len(pred)} total")


if __name__ == "__main__":
    main()

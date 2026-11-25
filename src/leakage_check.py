
import os
import pandas as pd

# ==========================================
# 1. SETTINGS
# ==========================================

DATA_PATH = "data/processed/cleaned_data.csv"
OUTPUT_DIR = "data/processed"
TARGET = "low_yield_risk"
THRESHOLD = 0.90

# ==========================================
# 2. LOAD DATA
# ==========================================

print("\nLoading cleaned AgriPak dataset...")

df = pd.read_csv(DATA_PATH)

if TARGET not in df.columns:
    raise ValueError("Target column not found!")

X = df.drop(columns=[TARGET])
y = df[TARGET]

# ==========================================
# 3. ENCODE CATEGORICAL FEATURES
# ==========================================

categorical_features = X.select_dtypes(
    include=["object", "str"]
).columns.tolist()

encoded_X = pd.get_dummies(
    X,
    columns=categorical_features,
    drop_first=False,
    dtype=int
)

# ==========================================
# 4. CALCULATE CORRELATIONS
# ==========================================

correlations = encoded_X.corrwith(y).dropna()

results = pd.DataFrame({
    "feature": correlations.index,
    "correlation": correlations.values
})

results["absolute_correlation"] = (
    results["correlation"].abs()
)

results = results.sort_values(
    "absolute_correlation",
    ascending=False
)

# ==========================================
# 5. FLAG HIGH CORRELATIONS
# ==========================================

results["flagged_above_0.9"] = (
    results["absolute_correlation"] > THRESHOLD
)

flagged = results[
    results["flagged_above_0.9"]
]

# ==========================================
# 6. DISPLAY RESULTS
# ==========================================

print("\n========== FEATURE CORRELATIONS ==========")

print(results.to_string(index=False))

print("\n========== LEAKAGE FLAGS ==========")

if flagged.empty:
    print("No features exceeded the 0.90 threshold.")
else:
    print(flagged.to_string(index=False))

# ==========================================
# 7. SAVE REPORT
# ==========================================

os.makedirs(OUTPUT_DIR, exist_ok=True)

results.to_csv(
    f"{OUTPUT_DIR}/feature_correlations.csv",
    index=False
)

flagged.to_csv(
    f"{OUTPUT_DIR}/leakage_flags.csv",
    index=False
)

print("\nTotal features checked:", len(results))
print("Features flagged:", len(flagged))

print("\nCorrelation report saved successfully!")

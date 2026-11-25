
import os
import json
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)


DATA_PATH = "data/crop_data.csv"
OUTPUT_DIR = "data/processed"
RANDOM_STATE = 42
TARGET = "low_yield_risk"

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("\nLoading AgriPak dataset...")

df = pd.read_csv(DATA_PATH)

print("Original dataset shape:", df.shape)

df.columns = df.columns.str.strip()

# Remove leading/trailing spaces in text columns
for col in df.select_dtypes(include=["object", "str"]).columns:
    df[col] = df[col].str.strip()

# Convert numeric columns to proper numeric types
numeric_columns = [
    "rainfall_mm",
    "temperature_c",
    "fertilizer_used",
    "irrigation_used",
    "days_to_harvest",
    TARGET
]

for col in numeric_columns:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# Target must exist and cannot be missing
if TARGET not in df.columns:
    raise ValueError(f"Target column {TARGET} not found.")

if df[TARGET].isna().any():
    raise ValueError("Target contains missing or invalid values.")

if not set(df[TARGET].unique()).issubset({0, 1}):
    raise ValueError("Target must contain only 0 and 1.")

# Save cleaned dataset
df.to_csv(f"{OUTPUT_DIR}/cleaned_data.csv", index=False)

print("Cleaned dataset shape:", df.shape)
print("Missing values after type conversion:")
print(df.isnull().sum())

X = df.drop(columns=[TARGET])
y = df[TARGET].astype(int)

categorical_features = X.select_dtypes(
    include=["object", "str"]
).columns.tolist()

numerical_features = X.select_dtypes(
    include=["number"]
).columns.tolist()

print("\nCategorical features:", categorical_features)
print("Numerical features:", numerical_features)


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=RANDOM_STATE,
    stratify=y
)

print("\nTraining records:", len(X_train))
print("Testing records:", len(X_test))

# Save reproducible split files
X_train.to_csv(f"{OUTPUT_DIR}/X_train.csv", index=False)
X_test.to_csv(f"{OUTPUT_DIR}/X_test.csv", index=False)

y_train.to_csv(f"{OUTPUT_DIR}/y_train.csv", index=False)
y_test.to_csv(f"{OUTPUT_DIR}/y_test.csv", index=False)


numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("numeric", numeric_pipeline, numerical_features),
    ("categorical", categorical_pipeline, categorical_features)
])


model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(
        max_iter=1000,
        random_state=RANDOM_STATE
    ))
])

print("\nTraining Logistic Regression...")

model.fit(X_train, y_train)


y_pred = model.predict(X_test)

results = {
    "accuracy": accuracy_score(y_test, y_pred),
    "precision": precision_score(y_test, y_pred, zero_division=0),
    "recall": recall_score(y_test, y_pred, zero_division=0),
    "f1_score": f1_score(y_test, y_pred, zero_division=0)
}

print("\n========== BASELINE RESULTS ==========")

for metric, value in results.items():
    print(f"{metric}: {value:.4f}")

print("\n========== CONFUSION MATRIX ==========")
print(confusion_matrix(y_test, y_pred))

print("\n========== CLASSIFICATION REPORT ==========")
print(classification_report(y_test, y_pred, zero_division=0))

# Save results
with open(f"{OUTPUT_DIR}/baseline_results.json", "w") as file:
    json.dump(results, file, indent=4)

# Save model
import joblib
joblib.dump(model, f"{OUTPUT_DIR}/logistic_regression_model.joblib")

print("\nAll files saved successfully!")
print("Week 1 baseline training completed.")

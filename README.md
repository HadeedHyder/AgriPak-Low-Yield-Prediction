# AgriPak Farms – Low-Yield Risk Prediction

## 1. Project Overview

This project was developed as part of the Machine Learning Internship task for AgriPak Farms.

The objective is to build a machine learning model that predicts whether agricultural conditions are associated with low-yield risk.

The project uses a Logistic Regression model as a baseline and performs a feature-correlation check to identify possible data leakage.

## 2. Dataset Information

The dataset contains:

- 50,000 records
- 9 input features
- 1 target column: `low_yield_risk`

The target column represents:

- `0` – No low-yield risk
- `1` – Low-yield risk

Input features include region, soil type, crop, rainfall, temperature, fertilizer usage, irrigation usage, weather condition, and days to harvest.

**Note:** The original dataset is kept locally and is not included in this public repository.

## 3. Data Cleaning and Preprocessing

The following cleaning steps were performed:

1. Removed unnecessary spaces from column names.
2. Removed unnecessary spaces from text values.
3. Converted numerical columns into appropriate numeric data types.
4. Checked for missing values.
5. Checked for duplicate records.
6. Verified that the target column contains only binary values (0 and 1).

The original dataset contained no missing values or duplicate records, so no records needed to be removed.

For model preprocessing:

- Numerical features were filled using median imputation.
- Categorical features were filled using most-frequent-value imputation.
- Categorical features were converted using One-Hot Encoding.
- Numerical features were standardized using StandardScaler.

## 4. Train-Test Split

The dataset was divided using:

- Training data: 80% (40,000 records)
- Testing data: 20% (10,000 records)
- Random state: 42
- Stratification: Enabled

Stratification maintains the target-class proportions in both datasets.

## 5. Baseline Model: Logistic Regression

Logistic Regression was selected as the baseline classification model.

### Model Performance

| Metric | Result |
|---|---:|
| Accuracy | 91.87% |
| Precision | 88.84% |
| Recall | 87.80% |
| F1-Score | 88.32% |

### Confusion Matrix

```text
[[6114  386]
 [ 427 3073]]
```

The model achieved 91.87% accuracy on the test dataset.

## 6. Data Leakage Sanity Check

A correlation check was performed between the encoded input features and the target variable.

The threshold used to flag potentially suspicious features was:

`Absolute correlation > 0.90`

A total of 24 encoded features were checked.

**Result: No feature exceeded the 0.90 correlation threshold.**

The strongest correlations were:

| Feature | Correlation |
|---|---:|
| rainfall_mm | -0.636409 |
| fertilizer_used | -0.334198 |
| irrigation_used | -0.266189 |
| temperature_c | -0.068506 |
| days_to_harvest | 0.002660 |

No feature was flagged by this correlation-based check.

However, this does not guarantee that the dataset is completely free from leakage. Features should also be checked to confirm that they would genuinely be available at the time a prediction is made, particularly `days_to_harvest`.

## 7. Project Structure

```text
AgriPak-Low-Yield-Prediction/
│
├── data/
│   ├── crop_data.csv
│   └── processed/
│
├── src/
│   ├── train.py
│   └── leakage_check.py
│
├── notebooks/
│
├── README.md
├── LICENSE
├── requirements.txt
└── .gitignore
```

The dataset and generated outputs are stored locally and excluded from the public repository.

## 8. How to Run the Project

### Step 1: Clone the repository

```bash
git clone YOUR_GITHUB_REPOSITORY_LINK
cd AgriPak-Low-Yield-Prediction
```

### Step 2: Create and activate a virtual environment

```bash
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Add the dataset

Place the provided dataset inside the `data/` folder with the filename:

`crop_data.csv`

### Step 5: Run the training script

```bash
python src/train.py
```

### Step 6: Run the leakage check

```bash
python src/leakage_check.py
```

## 9. Conclusion

The Logistic Regression baseline achieved 91.87% accuracy, with an F1-score of 88.32%.

The correlation-based leakage check did not identify any feature with an absolute correlation greater than 0.90. Further checks of feature availability at prediction time are still important before considering the model reliable for real-world use.
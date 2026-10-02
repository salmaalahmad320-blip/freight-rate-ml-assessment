# Freight Rate Prediction Challenge

This repository contains my solution for the Spotter Machine Learning Engineer assessment.

The objective is to build a regression model that predicts freight load rates using historical shipment information, validate the model using a realistic chronological split, generate predictions for the provided validation dataset, and produce daily December 2025 predictions for the fixed Lexington → Fort Wayne lane.

---

## Project Structure

```text
machinelearning_usa/
│
├── data/
│   ├── train-test (1).csv
│   ├── validation (1).csv
│   ├── validation-predictions-template (2).csv
│   ├── december-chart-inputs (2).csv
│   └── december_chart_inputs.csv
│
├── notebooks/
│   └── exploration.ipynb
│
├── src/
│   └── train_and_predict.py
│
├── scorer_results/
│   └── candidate_december.png
│
├── validation_predictions.csv
├── requirements.txt
├── score.py
└── README.md
```

The prediction script supports both the official assessment filenames and the alternate filenames created by local downloads.

---

## Dataset

The labeled development dataset contains 48,000 freight loads.

The prediction target is:

```text
posted_rate
```

The main model inputs include:

- pickup city
- delivery city
- pickup latitude and longitude
- delivery latitude and longitude
- distance
- equipment type
- shipment weight
- date
- market index
- quote signal

The final validation dataset contains 12,000 loads that require predictions.

---

## Data Quality Findings

### Negative Shipment Weights

The development and validation datasets contained negative shipment weights.

Observed counts:

```text
Training negative weights:   292
Validation negative weights: 145
```

Shipment weight cannot physically be negative.

To investigate whether these records represented invalid observations or a sign-entry issue, I compared the absolute values of the negative weights with the distribution of valid positive weights.

The distributions were very similar:

```text
Absolute negative weight mean:   31,724
Positive weight mean:            31,415

Absolute negative weight median: 31,821
Positive weight median:          31,494
```

This suggested that the negative sign was most likely a data-entry or encoding issue rather than an invalid shipment record.

The weights were therefore corrected using:

```python
df["weight"] = df["weight"].abs()
```

### Missing Values

The development dataset contained missing values in:

```text
weight:        300 rows
market_index:  374 rows
```

The final validation dataset also contained missing values in these variables.

Missing numeric values were filled using median values learned from the training data.

Using training-derived medians avoids introducing information from future or held-out observations during preprocessing.

---

## Feature Engineering

The original `date` column was converted to a datetime feature.

The following calendar features were extracted:

```text
year
month
day
dayofweek
dayofyear
```

These features allow the model to capture temporal patterns in freight rates.

The categorical features used directly by CatBoost were:

```text
pickup
delivery
equipment
```

The complete final model feature set contains:

```text
pickup
delivery
pickup_lat
pickup_lon
delivery_lat
delivery_lon
distance
equipment
weight
market_index
quote_signal
year
month
day
dayofweek
dayofyear
```

---

## Validation Strategy

A chronological holdout split was used instead of a random split.

The labeled development dataset covers:

```text
2025-01-01 through 2025-10-31
```

The final validation dataset covers:

```text
2025-11-01 through 2025-12-31
```

Because the final prediction period occurs after the labeled development period, a chronological split better represents the actual evaluation setting and reduces the risk of temporal leakage.

The internal validation split was:

```text
Training period:
2025-01-01 through 2025-08-31

Validation period:
2025-09-01 through 2025-10-31
```

This produced:

```text
Training rows:   38,477
Validation rows:  9,523
```

---

## Model Comparison

Three regression approaches were evaluated using the same chronological holdout period.

| Model | MAE | RMSE | R² |
|---|---:|---:|---:|
| CatBoost | $111.41 | $633.39 | 0.8277 |
| XGBoost | $157.51 | $652.87 | 0.8170 |
| Random Forest | $243.80 | $713.20 | 0.7816 |

CatBoost produced the lowest MAE and RMSE and the highest R² on the internal chronological validation set.

For this reason, CatBoost was selected as the final model.

---

## Final Model

The final CatBoost configuration is:

```python
CatBoostRegressor(
    iterations=123,
    learning_rate=0.05,
    depth=8,
    loss_function="RMSE",
    random_seed=42,
)
```

During model development, early stopping was used on the chronological validation period.

The best validation iteration was:

```text
122
```

The final model therefore uses 123 boosting iterations and is retrained on all 48,000 labeled development rows before generating the final predictions.

---

## Internal Validation Performance

The selected CatBoost model achieved:

```text
MAE:  $111.41
RMSE: $633.39
R²:   0.8277
```

These metrics are based on the internal chronological holdout set.

The labels for the final 12,000-row validation dataset are not provided, so final evaluation metrics are calculated by Spotter after submission.

---

## December Prediction Approach

The supplied December input represents a fixed load with:

```text
Pickup:    Lexington
Delivery:  Fort Wayne
Distance:  360 miles
Equipment: Dry Van
Weight:    32,000 lb
```

There is one row for each day from December 1 through December 31, 2025.

The supplied December file does not contain every variable required by the final model, specifically:

```text
pickup_lat
pickup_lon
delivery_lat
delivery_lon
market_index
quote_signal
```

The development dataset contains 32 historical observations for the Lexington → Fort Wayne lane.

Median values from those historical lane observations were used for the missing coordinates, `market_index`, and `quote_signal`.

These values are held constant across all 31 December rows, while the date-derived model features change each day.

The final December prediction file used by the scorer is:

```text
data/december_chart_inputs.csv
```

The scorer creates:

```text
scorer_results/candidate_december.png
```

---

## Requirements

The project dependencies are listed in:

```text
requirements.txt
```

Install them with:

```bash
python -m pip install -r requirements.txt
```

---

## Run the Prediction Pipeline

From the project root, run:

```bash
python src/train_and_predict.py
```

The script performs the complete final prediction workflow:

1. Loads the development and validation datasets.
2. Corrects negative shipment weights.
3. Converts dates and creates calendar features.
4. Handles missing numeric values.
5. Trains the final CatBoost model on all labeled development data.
6. Generates predictions for all 12,000 validation loads.
7. Saves `validation_predictions.csv`.
8. Generates predictions for all 31 December rows.
9. Saves the scorer-ready December file as `data/december_chart_inputs.csv`.

A successful run ends with:

```text
Pipeline completed successfully.
```

---

## Run the Provided Scorer

After running the prediction pipeline, execute:

```bash
python score.py --predictions validation_predictions.csv --december-predictions data/december_chart_inputs.csv
```

A successful scorer run produces:

```text
Validated 12,000 final predictions.
Validated 31 fixed December predictions.
Created chart: scorer_results/candidate_december.png
Final validation metrics are calculated by Spotter after submission.
```

---

## Final Outputs

The main generated deliverables are:

```text
validation_predictions.csv
data/december_chart_inputs.csv
scorer_results/candidate_december.png
```

`validation_predictions.csv` contains predictions for all 12,000 final validation loads.

`data/december_chart_inputs.csv` contains the 31 fixed-lane December predictions.

`scorer_results/candidate_december.png` contains the chart generated by the provided scorer.

---

## Reproducibility

The final CatBoost model uses:

```text
random_seed = 42
```

The clean end-to-end prediction pipeline is implemented in:

```text
src/train_and_predict.py
```

Exploratory analysis, data-quality investigation, validation experiments, and model comparisons are retained in:

```text
notebooks/exploration.ipynb
```

This separates exploratory model development from the final reproducible prediction pipeline.
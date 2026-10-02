from __future__ import annotations

from pathlib import Path

import pandas as pd
from catboost import CatBoostRegressor


# =========================================================
# 1. Project paths
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"


def resolve_file(*names: str) -> Path:
    """
    Return the first matching file from the data directory.

    Official assessment filenames are checked first.
    Local downloaded filename variants are supported as fallbacks.
    """
    for name in names:
        path = DATA_DIR / name

        if path.exists():
            return path

    raise FileNotFoundError(
        f"Could not find any of these files: {names}"
    )


# Support both the official filenames and the local downloaded names
TRAIN_PATH = resolve_file(
    "train_test.csv",
    "train-test.csv",
    "train-test (1).csv",
)

VALIDATION_PATH = resolve_file(
    "validation.csv",
    "validation (1).csv",
)

TEMPLATE_PATH = resolve_file(
    "validation_predictions_template.csv",
    "validation-predictions-template.csv",
    "validation-predictions-template (2).csv",
)

DECEMBER_PATH = resolve_file(
    "december_chart_inputs.csv",
    "december-chart-inputs.csv",
    "december-chart-inputs (2).csv",
)

SUBMISSION_PATH = PROJECT_ROOT / "validation_predictions.csv"


# =========================================================
# 2. Model configuration
# =========================================================

TARGET = "posted_rate"

CATEGORICAL_FEATURES = [
    "pickup",
    "delivery",
    "equipment",
]

MODEL_FEATURES = [
    "pickup",
    "delivery",
    "pickup_lat",
    "pickup_lon",
    "delivery_lat",
    "delivery_lon",
    "distance",
    "equipment",
    "weight",
    "market_index",
    "quote_signal",
    "year",
    "month",
    "day",
    "dayofweek",
    "dayofyear",
]


# =========================================================
# 3. Data preparation functions
# =========================================================

def clean_weight(df: pd.DataFrame) -> pd.DataFrame:
    """
    Correct invalid negative shipment weights.

    During EDA, negative weights were found to have an
    absolute-value distribution very similar to valid positive
    weights. This suggests a sign-entry / encoding issue.
    """
    result = df.copy()

    result["weight"] = result["weight"].abs()

    return result


def add_date_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert date to datetime and create calendar features.
    """
    result = df.copy()

    result["date"] = pd.to_datetime(
        result["date"],
        errors="raise",
    )

    result["year"] = result["date"].dt.year
    result["month"] = result["date"].dt.month
    result["day"] = result["date"].dt.day
    result["dayofweek"] = result["date"].dt.dayofweek
    result["dayofyear"] = result["date"].dt.dayofyear

    return result


def fill_numeric_missing(
    train_features: pd.DataFrame,
    other_features: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Fill missing numeric values using medians learned
    from the training data only.
    """
    train_features = train_features.copy()
    other_features = other_features.copy()

    columns_to_impute = [
        "weight",
        "market_index",
    ]

    for column in columns_to_impute:
        median_value = train_features[column].median()

        train_features[column] = (
            train_features[column]
            .fillna(median_value)
        )

        other_features[column] = (
            other_features[column]
            .fillna(median_value)
        )

    return train_features, other_features


# =========================================================
# 4. Load training and final validation data
# =========================================================

def load_model_data():
    """
    Load, clean, and prepare development and validation data.
    """

    train = pd.read_csv(TRAIN_PATH)
    validation = pd.read_csv(VALIDATION_PATH)

    # Correct negative weight values
    train = clean_weight(train)
    validation = clean_weight(validation)

    # Add temporal features
    train = add_date_features(train)
    validation = add_date_features(validation)

    # Build model matrices
    X_train = train[MODEL_FEATURES].copy()
    y_train = train[TARGET].copy()

    X_validation = validation[MODEL_FEATURES].copy()

    # Impute missing numeric values using training medians
    X_train, X_validation = fill_numeric_missing(
        X_train,
        X_validation,
    )

    return (
        train,
        validation,
        X_train,
        y_train,
        X_validation,
    )


# =========================================================
# 5. Train final model
# =========================================================

def train_final_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> CatBoostRegressor:
    """
    Train the final CatBoost regression model.

    Model selection was performed using a chronological split:

        Training:   2025-01-01 through 2025-08-31
        Validation: 2025-09-01 through 2025-10-31

    CatBoost outperformed Random Forest and XGBoost on the
    internal holdout.

    Best CatBoost validation iteration:
        iteration 122

    Therefore the final model uses 123 boosting iterations
    when trained on the full labeled development dataset.
    """

    model = CatBoostRegressor(
        iterations=123,
        learning_rate=0.05,
        depth=8,
        loss_function="RMSE",
        random_seed=42,
        verbose=50,
    )

    model.fit(
        X_train,
        y_train,
        cat_features=CATEGORICAL_FEATURES,
    )

    return model


# =========================================================
# 6. Generate final validation predictions
# =========================================================

def create_validation_predictions(
    model: CatBoostRegressor,
    validation: pd.DataFrame,
    X_validation: pd.DataFrame,
) -> None:
    """
    Predict all 12,000 final validation loads and save
    validation_predictions.csv using the provided template.
    """

    predictions = model.predict(X_validation)

    if len(predictions) != len(validation):
        raise ValueError(
            "Prediction count does not match validation row count."
        )

    # Load the provided template so the official load_id order
    # is preserved exactly.
    template = pd.read_csv(TEMPLATE_PATH)

    if list(template.columns) != [
        "load_id",
        "predicted_rate",
    ]:
        raise ValueError(
            "Unexpected validation prediction template format."
        )

    # Map predictions using load_id rather than relying
    # only on row position.
    prediction_map = pd.Series(
        data=predictions,
        index=validation["load_id"].astype(str),
    )

    template["predicted_rate"] = (
        template["load_id"]
        .astype(str)
        .map(prediction_map)
    )

    # Final safety checks
    if template["predicted_rate"].isna().any():
        raise ValueError(
            "Some validation IDs did not receive predictions."
        )

    if (template["predicted_rate"] <= 0).any():
        raise ValueError(
            "Validation predictions contain non-positive rates."
        )

    if template["load_id"].duplicated().any():
        raise ValueError(
            "Duplicate load IDs found in prediction template."
        )

    template.to_csv(
        SUBMISSION_PATH,
        index=False,
    )

    print(
        f"Saved {len(template):,} validation predictions "
        f"to {SUBMISSION_PATH}"
    )


# =========================================================
# 7. Prepare December fixed-lane features
# =========================================================

def prepare_december_features(
    train: pd.DataFrame,
    december: pd.DataFrame,
) -> pd.DataFrame:
    """
    Prepare the provided December fixed-lane data.

    The December input file contains:

        pickup
        delivery
        distance
        equipment
        weight
        date
        predicted_rate

    The final model also requires coordinates, market_index,
    and quote_signal.

    Historical observations for the same
    Lexington -> Fort Wayne lane are therefore used to recover
    these features.

    Median historical values are used so that date remains
    the only changing feature across the 31 December rows.
    """

    december = clean_weight(december)
    december = add_date_features(december)

    pickup = december["pickup"].iloc[0]
    delivery = december["delivery"].iloc[0]

    lane_history = train[
        (train["pickup"] == pickup)
        & (train["delivery"] == delivery)
    ].copy()

    if lane_history.empty:
        raise ValueError(
            f"No historical training observations were found "
            f"for lane {pickup} -> {delivery}."
        )

    lane_features = [
        "pickup_lat",
        "pickup_lon",
        "delivery_lat",
        "delivery_lon",
        "market_index",
        "quote_signal",
    ]

    for column in lane_features:
        median_value = lane_history[column].median()

        if pd.isna(median_value):
            raise ValueError(
                f"Could not calculate historical median "
                f"for December feature: {column}"
            )

        december[column] = median_value

    X_december = december[MODEL_FEATURES].copy()

    # Fallback imputation using full development data
    for column in [
        "weight",
        "market_index",
    ]:
        X_december[column] = (
            X_december[column]
            .fillna(train[column].median())
        )

    return X_december


# =========================================================
# 8. Generate December predictions
# =========================================================

def create_december_predictions(
    model: CatBoostRegressor,
    train: pd.DataFrame,
) -> None:
    """
    Predict the fixed Lexington -> Fort Wayne load for every
    day in December 2025.
    """

    december = pd.read_csv(DECEMBER_PATH)

    if len(december) != 31:
        raise ValueError(
            f"Expected 31 December rows, found {len(december)}."
        )

    X_december = prepare_december_features(
        train,
        december,
    )

    predictions = model.predict(X_december)

    december["predicted_rate"] = predictions

    # The scorer requires exactly these seven columns
    # in exactly this order.
    required_columns = [
        "pickup",
        "delivery",
        "distance",
        "equipment",
        "weight",
        "date",
        "predicted_rate",
    ]

    december = december[required_columns]

    if december["predicted_rate"].isna().any():
        raise ValueError(
            "December predictions contain missing values."
        )

    if (december["predicted_rate"] <= 0).any():
        raise ValueError(
            "December predictions contain non-positive values."
        )

    december.to_csv(
        DECEMBER_PATH,
        index=False,
    )

    print(
        f"Saved {len(december)} December predictions "
        f"to {DECEMBER_PATH}"
    )


# =========================================================
# 9. Main pipeline
# =========================================================

def main() -> None:
    print("=" * 60)
    print("Freight Rate Prediction Pipeline")
    print("=" * 60)

    print("\nLoading and preparing data...")

    (
        train,
        validation,
        X_train,
        y_train,
        X_validation,
    ) = load_model_data()

    print(f"Training rows: {len(train):,}")
    print(f"Final validation rows: {len(validation):,}")

    print("\nTraining final CatBoost model...")

    model = train_final_model(
        X_train,
        y_train,
    )

    print("\nCreating final validation predictions...")

    create_validation_predictions(
        model,
        validation,
        X_validation,
    )

    print("\nCreating December predictions...")

    create_december_predictions(
        model,
        train,
    )

    print("\nPipeline completed successfully.")
    print("=" * 60)


if __name__ == "__main__":
    main()


    
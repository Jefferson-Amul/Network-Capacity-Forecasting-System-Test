import numpy as np

from src.ncf.data_loader import load_data
from src.ncf.data_processing import run_data_quality_pipeline
from src.ncf.feature_engineering import (
    engineer_features,
    create_forecasting_target,
)
from src.ncf.data_split import (
    chronological_train_val_test_split,
    split_features_target,
)
from src.ncf.data_scaling import scale_features
from src.ncf.models import (
    save_model,
)
from src.ncf.hyperparameter_search import (
    run_hyperparameter_search,
)
from src.ncf.evaluation import (
    evaluate_regression_model,
)
from src.ncf.explainability import (
    calculate_feature_importance,
)
from src.ncf.mlflow_tracking import (
    start_experiment,
    log_model_experiment,
)


DATA_PATH = "data/institutions/agg_10_minutes/0.csv"

TARGET_COLUMN = "future_n_bytes"

HORIZON = 1


def main():

    print("\n" + "=" * 60)
    print("NETWORK CAPACITY FORECASTING SYSTEM")
    print("MODEL TRAINING")
    print("=" * 60)

    # Start MLflow experiment
    start_experiment(
        "Network Capacity Forecasting"
    )

    # 1. Load data
    print("\n[1/10] Loading data...")

    df = load_data(DATA_PATH)

    print("Loaded shape:", df.shape)

    # 2. Data processing
    print("\n[2/10] Running data quality pipeline...")

    df = run_data_quality_pipeline(df)

    print("Processed shape:", df.shape)

    # 3. Feature engineering
    print("\n[3/10] Creating forecasting features...")

    df = engineer_features(df)

    df = create_forecasting_target(
        df,
        target_column="n_bytes",
        horizon=HORIZON,
    )

    print(
        "Feature-engineered shape:",
        df.shape
    )

    # 4. Train / validation / test split
    print(
        "\n[4/10] Splitting data chronologically..."
    )

    (
        train_df,
        validation_df,
        test_df,
    ) = chronological_train_val_test_split(df)

    (
        X_train,
        y_train,
        X_validation,
        y_validation,
        X_test,
        y_test,
    ) = split_features_target(
        train_df,
        validation_df,
        test_df,
        target_column=TARGET_COLUMN,
    )

    # Log-transform training targets
    y_train = np.log1p(y_train)
    y_validation = np.log1p(y_validation)

    print(
        "Training data:",
        X_train.shape
    )

    print(
        "Validation data:",
        X_validation.shape
    )

    print(
        "Test data:",
        X_test.shape
    )

    # 5. Feature scaling
    print("\n[5/10] Scaling features...")

    (
        X_train_scaled,
        X_validation_scaled,
        X_test_scaled,
        scaler,
    ) = scale_features(
        X_train,
        X_validation,
        X_test,
    )

    print(
        "Scaled training data:",
        X_train_scaled.shape
    )

    print(
        "Scaled validation data:",
        X_validation_scaled.shape
    )

    print(
        "Scaled test data:",
        X_test_scaled.shape
    )

    # 6. Hyperparameter search
    print(
        "\n[6/10] Running hyperparameter search..."
    )

    best_result = run_hyperparameter_search(
        X_train_scaled,
        y_train,
        X_validation_scaled,
        y_validation,
    )

    model = best_result["model"]

    # 7. Model evaluation
    print(
        "\n[7/10] Evaluating best model on test data..."
    )

    evaluation_results = (
        evaluate_regression_model(
            model,
            X_test_scaled,
            y_test,
        )
    )

    # 8. Feature importance
    print(
        "\n[8/10] Calculating feature importance..."
    )

    feature_names = X_train.drop(
        columns=["id_time"]
    ).columns

    feature_importance = (
        calculate_feature_importance(
            model,
            X_test_scaled,
            y_test,
            feature_names,
        )
    )

    # 9. MLflow experiment tracking
    print(
        "\n[9/10] Logging experiment to MLflow..."
    )

    log_model_experiment(
        model=model,
        config=best_result,
        evaluation_results=evaluation_results,
    )

    # 10. Save model
    print(
        "\n[10/10] Saving best trained model..."
    )

    model_path = save_model(
        model,
        scaler=scaler,
    )

    print(
        "\nBest model saved successfully:"
    )

    print(model_path)

    print("\n" + "=" * 60)
    print("MODEL TRAINING COMPLETED")
    print("=" * 60)


if __name__ == "__main__":
    main()

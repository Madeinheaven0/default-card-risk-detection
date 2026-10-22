from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from feast import FeatureStore
import mlflow
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from sklearn.model_selection import KFold, learning_curve, train_test_split, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier

from src.config import MODEL_IMAGE_DIR, REPO_PATH, USED_DATASET_PATH, MODEL_CONFUSION_MATRIX_DIR, MODEL_LEARNING_CURVE_DIR, db_path
from src.data.preprocessing import (
    create_and_save_pipeline,
    create_transformer,
    split_and_balance_dataset,
)

from evaluate import evaluate_model
from src.data.preprocessing import split_and_balance_dataset
from src.logger import logger

store = FeatureStore(repo_path=str(REPO_PATH))

mlflow.set_tracking_uri(f"sqlite:///{db_path}")
mlflow.set_experiment("credit_default_experiment")


def plot_and_save_confusion_matrix(y_true, y_pred, model_name: str, output_dir: Path) -> Path:
    """Generate and save the confusion matrix figure."""
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)

    fig, ax = plt.subplots(figsize=(6, 6))
    disp.plot(cmap="Blues", ax=ax)
    plt.title(f"Confusion Matrix - {model_name}")

    save_path = output_dir / f"{model_name}_confusion_matrix.png"
    plt.savefig(save_path, bbox_inches="tight")
    plt.close(fig)

    logger.info(f"Confusion matrix saved : {save_path}")
    return save_path


def plot_and_save_learning_curve(model, X, y, model_name: str, output_dir: Path) -> Path:
    """Generate and save the Learning Curve."""
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=0)

    train_sizes, train_scores, test_scores = learning_curve(
        estimator=model,
        X=X,
        y=y,
        cv=cv,
        scoring="recall",
        n_jobs=-1,
        train_sizes=np.linspace(0.1, 1.0, 5),
    )

    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(train_sizes, train_mean, "o-", color="r", label="Training score")
    ax.plot(train_sizes, test_mean, "o-", color="g", label="Cross-validation score")

    ax.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1, color="r")
    ax.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, alpha=0.1, color="g")

    ax.set_title(f"Learning Curve - {model_name}")
    ax.set_xlabel("Training Examples")
    ax.set_ylabel("Recall Score")
    ax.legend(loc="best")
    ax.grid(True)

    save_path = output_dir / f"{model_name}_learning_curve.png"
    plt.savefig(save_path, bbox_inches="tight")
    plt.close(fig)

    logger.info(f"Learning curve saved : {save_path}")
    return save_path


def evaluate_and_log_model(
        model_name: str,
        model,
        X_train,
        X_test,
        y_train,
        y_test,
        output_dir_matrix: Path,
        output_dir_learning_curve: Path,
        threshold: float
    ):
    """Train, evaluate, draw plots et save all in MLflow."""
    logger.info(f"--- Start Training : {model_name} ---")

    output_dir_matrix.mkdir(parents=True, exist_ok=True)
    output_dir_learning_curve.mkdir(parents=True, exist_ok=True)

    with mlflow.start_run(run_name=model_name):
        # 1. Training
        model.fit(X_train, y_train)

        THRESHOLD = threshold

        if hasattr(model, "predict_proba"):
            y_score = model.predict_proba(X_test)[:, 1]
            y_pred = (y_score >= THRESHOLD).astype(int)
        else:
            y_score = model.predict(X_test)
            y_pred = y_score

        # 3. Metrics
        recall, f1, roc_auc = evaluate_model(y_test, y_pred, y_score)

        # 4. Saving the parameters of the model
        mlflow.log_params(model.get_params())

        # 5. Plots
        cm_path = plot_and_save_confusion_matrix(y_test, y_pred, model_name, output_dir_matrix)
        lc_path = plot_and_save_learning_curve(model, X_train, y_train, model_name, output_dir_learning_curve)

        # Logging artefacts in MLflow
        mlflow.log_artifact(str(cm_path))
        mlflow.log_artifact(str(lc_path))

        logger.info(f"--- Ends Training : {model_name} ---\n")


def train_models():
    logger.info("Feature Store succesfully initialized.")
    entity_df = pd.read_parquet(USED_DATASET_PATH)

    MAX_SAMPLES = 100_000

    _, entity_df = train_test_split(
        entity_df,
        test_size=MAX_SAMPLES,
        stratify=entity_df['flag'],
        random_state=0
    )

    features_to_fetch = [
        "customer_params:rn",
        "customer_params:pre_since_opened",
        "customer_params:pre_since_confirmed",
        "customer_params:pre_loans_credit_limit",
        "customer_params:pre_loans_credit_cost_rate",
        "customer_params:paym_mean_status",
        "customer_params:paym_count_incidents",
        "customer_params:paym_incidents_over_mean",
        "customer_params:total_zero_loans_count",
        "customer_params:paym_trend_3m_vs_12m",
        "customer_params:paym_hist_mean_1_23",
        "customer_params:total_delinquency_score",
        "customer_params:count_paym_status_0_24m",
        "customer_params:count_paym_status_2_24m",
        "customer_params:count_paym_status_3_24m",
        "customer_params:enc_loans_account_holder_type",
        "customer_params:enc_loans_credit_status",
        "customer_params:enc_loans_credit_type",
        "customer_params:enc_loans_account_cur",
        "customer_params:pclose_flag",
        "customer_params:absolute_zero_defect",
        "customer_params:paym_latest_status",
        "customer_params:paym_max_status",
        "customer_params:is_zero_loans5",
        "customer_params:is_zero_loans530",
        "customer_params:is_zero_loans3060",
        "customer_params:is_zero_loans6090",
        "customer_params:is_zero_loans90",
        "customer_params:is_zero_util",
        "customer_params:is_zero_over2limit",
        "customer_params:is_zero_maxover2limit",
        "customer_params:pre_loans_next_pay_summ",
        "customer_params:pre_loans_outstanding",
    ]

    logger.info("Extraction of the features from feast")

    dataset_job = store.get_historical_features(
        entity_df=entity_df,
        features=features_to_fetch
    )

    dataset = dataset_job.to_df()

    if "flag_x" in dataset.columns:
        dataset["flag"] = dataset["flag_x"]
        dataset = dataset.drop(columns=[col for col in dataset.columns if "flag_" in col])

    logger.debug("Dataset insight")
    logger.debug(f"Shape: {dataset.shape}")
    logger.debug(dataset.head())

    X_train, y_train, X_test, y_test = split_and_balance_dataset(
        data=dataset,
        target='flag',
        features_names=features_to_fetch,
        test_size=0.2,
    )

    models = {
        "RandomForestClassifier": RandomForestClassifier(random_state=0, class_weight='balanced'),
        "AdaBoostClassifier": AdaBoostClassifier(random_state=0),
        "XGBClassifier": XGBClassifier(random_state=0),
        "LogisticRegression": LogisticRegression(max_iter=5000, random_state=0, class_weight='balanced'),
        "KNeighborsClassifier": KNeighborsClassifier(),
    }

    image_dir = Path(MODEL_IMAGE_DIR)
    matrix_dir = Path(MODEL_CONFUSION_MATRIX_DIR)
    learning_curve_dir = Path(MODEL_LEARNING_CURVE_DIR)

    for name, model in models.items():
        evaluate_and_log_model(
            model_name=name,
            model=model,
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test,
            output_dir_matrix=matrix_dir,
            output_dir_learning_curve=learning_curve_dir,
            threshold=0.2
        )


if __name__ == "__main__":
    train_models()
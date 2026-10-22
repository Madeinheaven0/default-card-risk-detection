from typing import Dict, Tuple, Union
import numpy as np
from sklearn.metrics import f1_score, recall_score, roc_auc_score
import mlflow

from src.logger import logger


mlflow.set_experiment("Evaluation")


def evaluate_model_recall(y_test, y_pred) -> float:
    logger.info("Evaluating model recall")
    recall = float(recall_score(y_test, y_pred))
    mlflow.log_metric("Recall", recall)
    logger.info(f"Recall: {recall:.4f}")
    return recall


def evaluate_model_f1(y_test, y_pred) -> float:
    logger.info("Evaluating model f1")
    f1 = float(f1_score(y_test, y_pred))
    mlflow.log_metric("F1", f1)
    logger.info(f"F1: {f1:.4f}")
    return f1


def evaluate_model_roc_score(y_test, y_score) -> float:
    logger.info("Evaluating model roc_auc")
    roc_auc = float(roc_auc_score(y_test, y_score))
    mlflow.log_metric("Roc_auc", roc_auc)
    logger.info(f"ROC AUC: {roc_auc:.4f}")
    return roc_auc


def evaluate_model(y_test, y_pred, y_score=None) -> Tuple[float, float, float]:
    """
    Évalue le modèle avec Recall, F1 et ROC AUC.
    Si y_score (probabilités/décisions) n'est pas fourni, y_pred est utilisé pour ROC AUC.
    """
    y_score_for_auc = y_score if y_score is not None else y_pred

    recall = evaluate_model_recall(y_test, y_pred)
    f1 = evaluate_model_f1(y_test, y_pred)
    roc_auc = evaluate_model_roc_score(y_test, y_score_for_auc)

    return recall, f1, roc_auc
"""
Model Evaluation component: turns the training report into charts saved in
results/<module>/ (model comparison bar chart, confusion matrix, ROC curve).
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")  # no GUI needed
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay

from src.exception import CustomException
from src.logger import logging
from src.utils import load_object

MODEL_PATHS = {m: os.path.join("models", m, "best_model.pkl") for m in ["diabetes", "stroke", "mental_health"]}


def make_report_charts(module_name, best_name, report, test_data):
    try:
        out = os.path.join("results", module_name)
        os.makedirs(out, exist_ok=True)
        X_test, y_test = test_data
        if hasattr(X_test, "toarray") and module_name != "mental_health":
            X_test = X_test.toarray()

        # 1) metric comparison across all 6 models
        df = pd.DataFrame(report).T[["accuracy", "precision", "recall", "f1_score"]].astype(float)
        ax = df.plot(kind="bar", figsize=(10, 5), width=0.8)
        ax.set_title(f"{module_name}: model comparison (test set)")
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("score")
        plt.xticks(rotation=30, ha="right")
        plt.tight_layout()
        plt.savefig(os.path.join(out, "model_comparison.png"), dpi=130)
        plt.close()

        # 2) confusion matrix + ROC for the best saved model
        model = load_object(MODEL_PATHS[module_name])
        fig, ax = plt.subplots(figsize=(5, 4.5))
        ConfusionMatrixDisplay.from_estimator(model, X_test, y_test, ax=ax, cmap="Blues", colorbar=False)
        ax.set_title(f"{module_name}: {best_name}")
        plt.tight_layout()
        plt.savefig(os.path.join(out, "confusion_matrix.png"), dpi=130)
        plt.close()

        if len(np.unique(y_test)) == 2 and hasattr(model, "predict_proba"):
            fig, ax = plt.subplots(figsize=(5, 4.5))
            RocCurveDisplay.from_estimator(model, X_test, y_test, ax=ax)
            ax.plot([0, 1], [0, 1], "k--", alpha=0.4)
            ax.set_title(f"{module_name}: ROC curve ({best_name})")
            plt.tight_layout()
            plt.savefig(os.path.join(out, "roc_curve.png"), dpi=130)
            plt.close()
        logging.info(f"[{module_name}] Evaluation charts saved to {out}")
    except Exception as e:
        raise CustomException(e, sys)

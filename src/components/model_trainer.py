"""
Model Trainer component.
Trains the 6 candidate models, evaluates them, picks the best one per your
chosen primary metric, and saves it. One file handles all 3 modules.
"""

import os
import sys

from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB, GaussianNB

from src.exception import CustomException
from src.logger import logging
from src.utils import evaluate_classification_models, get_best_model, save_object

# ------------------------------------------------------------------
# Which 6 models to try per module, and which averaging + primary
# metric to use. See "Model Suitability Notes" from our architecture
# doc — KNN is included for mental_health mainly as a weak baseline
# for comparison; Naive Bayes for tabular is GaussianNB, included for
# the same reason.
# ------------------------------------------------------------------
MODEL_TRAINER_CONFIG = {
    "mental_health": {
        "model_path": os.path.join("models", "mental_health", "best_model.pkl"),
        "average": "macro",          # multi-class
        "primary_metric": "f1_score",
        "use_multinomial_nb": True,  # TF-IDF vectors -> MultinomialNB, not GaussianNB
    },
    "stroke": {
        "model_path": os.path.join("models", "stroke", "best_model.pkl"),
        "average": "binary",
        "primary_metric": "f1_score",   # prioritize F1 over accuracy: stroke data is imbalanced
        "use_multinomial_nb": False,
    },
    "diabetes": {
        "model_path": os.path.join("models", "diabetes", "best_model.pkl"),
        "average": "binary",
        "primary_metric": "f1_score",
        "use_multinomial_nb": False,
    },
}


def get_model_dict(module_name: str, config: dict):
    """
    Returns the 6 model instances for this module.
    Naive Bayes variant differs: MultinomialNB for sparse TF-IDF (text),
    GaussianNB for dense tabular clinical data.
    """
    # class_weight="balanced" fixes the imbalance in stroke (~5% positives) and
    # is harmless for the balanced datasets.
    cw = "balanced"
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, class_weight=cw),
        "Decision Tree": DecisionTreeClassifier(random_state=42, class_weight=cw, max_depth=8, min_samples_leaf=5),
        "Random Forest": RandomForestClassifier(random_state=42, class_weight=cw, n_estimators=300,
                                                min_samples_leaf=3, n_jobs=-1),
        "SVM": SVC(probability=True, class_weight=cw, random_state=42,
                   kernel="linear" if module_name == "mental_health" else "rbf"),
        "KNN": KNeighborsClassifier(),
        "Naive Bayes": MultinomialNB() if config["use_multinomial_nb"] else GaussianNB(),
    }
    return models


class ModelTrainer:
    def __init__(self, module_name: str):
        if module_name not in MODEL_TRAINER_CONFIG:
            raise ValueError(f"Unknown module_name '{module_name}'")
        self.module_name = module_name
        self.config = MODEL_TRAINER_CONFIG[module_name]

    def initiate_model_training(self, X_train, y_train, X_test, y_test):
        try:
            logging.info(f"[{self.module_name}] Training 6 candidate models")

            models = get_model_dict(self.module_name, self.config)

            # GaussianNB and SVC/KNN need dense arrays, not sparse matrices
            if hasattr(X_train, "toarray") and self.module_name != "mental_health":
                X_train = X_train.toarray()
                X_test = X_test.toarray()

            # --- model SELECTION uses 5-fold CV on the TRAIN set only (no test leakage) ---
            scoring = "f1" if self.config["average"] == "binary" else "f1_macro"
            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            cv_scores = {}
            for name, model in models.items():
                scores = cross_val_score(clone(model), X_train, y_train, cv=skf, scoring=scoring)
                cv_scores[name] = float(scores.mean())
                logging.info(f"[{self.module_name}] CV {scoring} {name}: {cv_scores[name]:.4f}")

            # --- final REPORTING is on the untouched test set ---
            report = evaluate_classification_models(
                models, X_train, y_train, X_test, y_test, average=self.config["average"]
            )
            for name in report:
                report[name]["cv_f1"] = cv_scores[name]

            best_name, best_model, best_score = get_best_model(report, "cv_f1")
            logging.info(f"[{self.module_name}] Best model (by CV): {best_name} (cv_f1={best_score:.4f})")
            # also keep test arrays for the evaluation charts
            self.test_data = (X_test, y_test)

            save_object(self.config["model_path"], best_model)

            # strip fitted model objects out before returning report (keep it JSON/CSV-friendly)
            clean_report = {
                name: {k: v for k, v in metrics.items() if k != "fitted_model"}
                for name, metrics in report.items()
            }

            return best_name, best_score, clean_report

        except Exception as e:
            raise CustomException(e, sys)




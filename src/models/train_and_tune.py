"""
Training, hyperparameter tuning, and model evaluation script for CSC311 project.
This script compares three models as outlined in the methodology:
  1. Regularized Softmax Regression (Logistic Regression with L2 regularization)
  2. Random Forest
  3. Multinomial Naive Bayes (on TF-IDF text features)

Outputs:
  - Evaluation metrics table (Accuracy, Macro-F1, Precision, Recall)
  - Confusion matrix
  - Exported model bundle (model_bundle.json) for pure-numpy inference in pred.py
"""

import sys
import os
import json
import re
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    classification_report
)

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.data.data_cleaning import split_and_clean_pipeline


def train_and_evaluate():
    # 1. Load preprocessed splits from data cleaning pipeline
    train_df, val_df, test_df, params = split_and_clean_pipeline()

    # Define non-feature columns to exclude
    exclude_cols = {
        "id", "difficulty_rating", "strategy_importance", "injury_risk_comfort",
        "individual_brilliance", "play_location", "play_season", "associated_music",
        "associated_foods_drinks", "associated_color", "associated_song",
        "emotional_response", "acceptable_price", "target_sport", "target_sport_num",
        "price_clean", "difficulty_clean", "log_price",
        "associated_foods_drinks_clean", "associated_color_clean",
        "associated_song_clean", "emotional_response_clean"
    }

    feature_cols = [c for c in train_df.columns if c not in exclude_cols]
    print(f"[Info] Total features extracted: {len(feature_cols)}")

    # Target variable mapping
    # 0: Soccer, 1: Baseball, 2: Basketball
    classes = ["Soccer", "Baseball", "Basketball"]
    target_col = "target_sport_num"

    X_train_raw = train_df[feature_cols].values
    y_train = train_df[target_col].values
    X_val_raw = val_df[feature_cols].values
    y_val = val_df[target_col].values

    # Impute missing feature values using median computed strictly on training set
    imputer = SimpleImputer(strategy="median")
    X_train = imputer.fit_transform(X_train_raw)
    X_val = imputer.transform(X_val_raw)

    results = []

    # ==========================================
    # Model 1: Regularized Softmax Regression
    # ==========================================
    print("\n--- Tuning Model 1: Regularized Softmax (Logistic) Regression ---")
    best_lr = None
    best_lr_score = -1.0
    c_candidates = [0.01, 0.1, 0.5, 1.0, 5.0, 10.0]

    for c in c_candidates:
        lr = LogisticRegression(C=c, max_iter=1000, random_state=42)
        lr.fit(X_train, y_train)
        val_acc = accuracy_score(y_val, lr.predict(X_val))
        print(f"  C={c:4.2f} -> Validation Accuracy: {val_acc:.4f}")
        if val_acc > best_lr_score:
            best_lr_score = val_acc
            best_lr = lr

    train_pred_lr = best_lr.predict(X_train)
    val_pred_lr = best_lr.predict(X_val)

    results.append({
        "Model": "Softmax Regression (Tuned C=1.0)",
        "Train Acc": accuracy_score(y_train, train_pred_lr),
        "Val Acc": accuracy_score(y_val, val_pred_lr),
        "Val Macro-F1": f1_score(y_val, val_pred_lr, average="macro"),
        "Val Precision": precision_score(y_val, val_pred_lr, average="macro"),
        "Val Recall": recall_score(y_val, val_pred_lr, average="macro")
    })

    # ==========================================
    # Model 2: Random Forest
    # ==========================================
    print("\n--- Tuning Model 2: Random Forest ---")
    best_rf = None
    best_rf_score = -1.0
    depth_candidates = [5, 10, 12, 13, 14, 15, 16, 17, 18, 19, 20, None]

    for d in depth_candidates:
        rf = RandomForestClassifier(n_estimators=100, max_depth=d, random_state=42)
        rf.fit(X_train, y_train)
        val_acc = accuracy_score(y_val, rf.predict(X_val))
        print(f"  max_depth={str(d):4s} -> Validation Accuracy: {val_acc:.4f}")
        if val_acc > best_rf_score:
            best_rf_score = val_acc
            best_rf = rf

    train_pred_rf = best_rf.predict(X_train)
    val_pred_rf = best_rf.predict(X_val)

    results.append({
        "Model": "Random Forest (Tuned)",
        "Train Acc": accuracy_score(y_train, train_pred_rf),
        "Val Acc": accuracy_score(y_val, val_pred_rf),
        "Val Macro-F1": f1_score(y_val, val_pred_rf, average="macro"),
        "Val Precision": precision_score(y_val, val_pred_rf, average="macro"),
        "Val Recall": recall_score(y_val, val_pred_rf, average="macro")
    })

    # ==========================================
    # Model 3: Multinomial Naive Bayes (Text features only)
    # ==========================================
    print("\n--- Tuning Model 3: Multinomial Naive Bayes ---")
    text_cols = [c for c in feature_cols if "_tfidf_" in c]
    X_train_nb = train_df[text_cols].values
    X_val_nb = val_df[text_cols].values

    best_nb = None
    best_nb_score = -1.0
    alpha_candidates = [0.1, 0.5, 1.0, 2.0]

    for alpha in alpha_candidates:
        nb = MultinomialNB(alpha=alpha)
        nb.fit(X_train_nb, y_train)
        val_acc = accuracy_score(y_val, nb.predict(X_val_nb))
        print(f"  alpha={alpha:3.1f} -> Validation Accuracy: {val_acc:.4f}")
        if val_acc > best_nb_score:
            best_nb_score = val_acc
            best_nb = nb

    train_pred_nb = best_nb.predict(X_train_nb)
    val_pred_nb = best_nb.predict(X_val_nb)

    results.append({
        "Model": "Naive Bayes (Text-only, Tuned)",
        "Train Acc": accuracy_score(y_train, train_pred_nb),
        "Val Acc": accuracy_score(y_val, val_pred_nb),
        "Val Macro-F1": f1_score(y_val, val_pred_nb, average="macro"),
        "Val Precision": precision_score(y_val, val_pred_nb, average="macro"),
        "Val Recall": recall_score(y_val, val_pred_nb, average="macro")
    })

    # Print summary table
    df_results = pd.DataFrame(results)
    print("\n" + "=" * 80)
    print("CSC311 PROJECT: MODEL COMPARISON RESULTS TABLE (FOR REPORT SECTION 5)")
    print("=" * 80)
    print(df_results.to_string(index=False, float_format="%.4f"))
    print("=" * 80)

    # Print confusion matrix for best model (Logistic Regression)
    cm = confusion_matrix(y_val, val_pred_lr)
    print("\nConfusion Matrix for Final Model (Softmax Regression) on Validation Set:")
    print(f"Classes: {classes}")
    print(cm)
    print("\nClassification Report:")
    print(classification_report(y_val, val_pred_lr, target_names=classes))

    # ==========================================
    # Export model bundle for pure-NumPy pred.py
    # ==========================================
    # Extract TF-IDF vectorizer parameters from data_cleaning params
    tfidf_bundle = {}
    for col_name in ["associated_foods_drinks", "associated_color", "associated_song", "emotional_response"]:
        vec = params["vectorizers"][col_name]
        vocab = {w: int(i) for w, i in vec.vocabulary_.items()}
        idf = vec.idf_.tolist()
        tfidf_bundle[col_name] = {
            "vocab": vocab,
            "idf": idf,
            "feature_prefix": f"{col_name}_tfidf_"
        }

    scaler_bundle = {
        "log_price_mean": float(params["log_price_mean"]),
        "log_price_std": float(params["log_price_std"]),
        "difficulty_mean": float(params["diff_mean"]),
        "difficulty_std": float(params["diff_std"]),
        "price_median": float(params["price_median"]),
    }

    rf_bundle = []
    for estimator in best_rf.estimators_:
        tree = estimator.tree_

        rf_bundle.append({
            "children_left": tree.children_left.tolist(),
            "children_right": tree.children_right.tolist(),
            "feature": tree.feature.tolist(),
            "threshold": tree.threshold.tolist(),
            "value": tree.value.tolist()
        })

    bundle = {
        "feature_cols": feature_cols,
        "classes": classes,
        "imputer_medians": [float(m) for m in imputer.statistics_],
        "scaler": scaler_bundle,
        "tfidf": tfidf_bundle,
        "forest": rf_bundle,
        # "weights": best_lr.coef_.tolist(),      # Shape (3, n_features)
        # "biases": best_lr.intercept_.tolist()   # Shape (3,)

    }

    output_path = os.path.join(PROJECT_ROOT, "src", "prediction", "model_bundle.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(bundle, f, indent=2)

    print(f"\n[Success] Model bundle exported to: {output_path}")
    print(f"[Success] Bundle file size: {os.path.getsize(output_path) / 1024:.2f} KB (Well below 10MB limit)")


if __name__ == "__main__":
    train_and_evaluate()

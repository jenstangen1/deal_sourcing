import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    auc,
    precision_score,
    recall_score,
)
from sklearn.ensemble import HistGradientBoostingClassifier
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns


def train_and_evaluate_boosted(outreach_capacity: int = 500):
    """Train and evaluate a boosted-tree model. If XGBoost is available it is used; otherwise fall back to
    scikit-learn's HistGradientBoostingClassifier.
    Saves evaluation plots and a CSV containing the top-N leads as per outreach_capacity.
    """
    # --- 1. Load Data ---
    data_path = os.path.join('data', 'ml_ready_dataset.csv')
    try:
        df = pd.read_csv(data_path)
        print(f"Loaded {data_path}. Shape: {df.shape}")
    except FileNotFoundError:
        print(f"Error: {data_path} not found. Run create_dataset.py first.")
        return

    # --- 2. Feature Engineering & Selection ---
    TARGET = 'Deal_Next_Year'
    features = df.drop(columns=[TARGET, 'Orgnr', 'Juridisk selskapsnavn'])
    target = df[TARGET]

    nace_col = 'NACE-bransjekode'
    if nace_col in features.columns:
        features = pd.get_dummies(features, columns=[nace_col], prefix='NACE')

    features = features.replace([np.inf, -np.inf], 0).fillna(0)

    # --- 3. Time-based split ---
    split_year = 2021
    train_idx = df[df['Year'] < split_year].index
    test_idx = df[df['Year'] >= split_year].index

    X_train = features.loc[train_idx].drop(columns=['Year'])
    y_train = target.loc[train_idx]
    X_test = features.loc[test_idx].drop(columns=['Year'])
    y_test = target.loc[test_idx]

    print(f"Training samples: {X_train.shape[0]} | Testing samples: {X_test.shape[0]}")
    print(f"Positives in train: {y_train.sum()} | Positives in test: {y_test.sum()}")

    # --- 4. Model training ---
    # Try to use XGBoost if available
    model_name = ''
    try:
        from xgboost import XGBClassifier  # type: ignore

        # scale_pos_weight = ratio of negatives to positives helps with imbalance
        scale_pos_weight = (y_train == 0).sum() / max(1, (y_train == 1).sum())
        model = XGBClassifier(
            n_estimators=400,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            objective='binary:logistic',
            eval_metric='aucpr',
            scale_pos_weight=scale_pos_weight,
            random_state=42,
            n_jobs=-1,
        )
        model_name = 'XGBoost'
    except ImportError:
        print("xgboost not installed; falling back to HistGradientBoostingClassifier.")
        model = HistGradientBoostingClassifier(
            max_depth=6,
            learning_rate=0.05,
            max_iter=400,
            class_weight='balanced',
            random_state=42,
        )
        model_name = 'HistGradientBoosting'

    print(f"Training {model_name} model…")
    model.fit(X_train, y_train)

    # --- 5. Evaluation ---
    print("\n--- Evaluation ---")
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_pred_default = model.predict(X_test)

    roc_auc = roc_auc_score(y_test, y_pred_proba)
    precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
    pr_auc = auc(recall, precision)
    print(f"ROC-AUC: {roc_auc:.4f} | PR-AUC: {pr_auc:.4f}")

    print("Classification report (threshold=0.5):")
    print(classification_report(y_test, y_pred_default))

    cm = confusion_matrix(y_test, y_pred_default)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['No Deal', 'Deal'], yticklabels=['No Deal', 'Deal'])
    plt.title(f'{model_name} Confusion Matrix (0.5 threshold)')
    plt.tight_layout()
    plt.savefig(f'{model_name.lower()}_confusion_matrix.png')
    plt.close()

    # Outreach capacity evaluation
    if outreach_capacity < len(y_test):
        sorted_idx = np.argsort(y_pred_proba)[::-1]
        top_idx = sorted_idx[:outreach_capacity]
        threshold_cap = y_pred_proba[top_idx[-1]]
        y_pred_cap = (y_pred_proba >= threshold_cap).astype(int)
        prec_cap = precision_score(y_test, y_pred_cap)
        rec_cap = recall_score(y_test, y_pred_cap)
        print(f"\nTop {outreach_capacity} firms — threshold {threshold_cap:.4f}")
        print(f"Precision: {prec_cap:.4f} | Recall: {rec_cap:.4f} (captures {int(rec_cap*y_test.sum())} deals)")

        leads_df = df.loc[test_idx].copy()
        leads_df['Deal_Probability'] = y_pred_proba
        leads_topN = leads_df.iloc[top_idx][['Orgnr', 'Juridisk selskapsnavn', 'Deal_Probability']]
        leads_topN.to_csv(f'top_{outreach_capacity}_leads_{model_name.lower()}.csv', index=False)
        print(f"Saved top-{outreach_capacity} leads to 'top_{outreach_capacity}_leads_{model_name.lower()}.csv'.")

    # Feature importance (if available)
    if hasattr(model, 'feature_importances_'):
        importances = pd.DataFrame({'feature': X_train.columns, 'importance': model.feature_importances_})
        importances = importances.sort_values('importance', ascending=False).head(20)
        print("\nTop 20 feature importances:")
        print(importances)

        plt.figure(figsize=(10, 8))
        sns.barplot(x='importance', y='feature', data=importances)
        plt.title(f'{model_name} – Top 20 Feature Importances')
        plt.tight_layout()
        plt.savefig(f'{model_name.lower()}_feature_importances.png')
        plt.close()
    else:
        print("Feature importances not available for this model.")


if __name__ == '__main__':
    # Adjust outreach_capacity here if desired
    train_and_evaluate_boosted(outreach_capacity=500) 
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    auc,
    precision_score,
    recall_score,
)
# Use a non-interactive backend for matplotlib to allow running on headless environments
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

def train_and_evaluate():
    """
    Trains and evaluates a machine learning model to predict company transactions.
    """
    # --- 1. Load Data ---
    data_path = os.path.join('data', 'ml_ready_dataset.csv')
    try:
        df = pd.read_csv(data_path)
        print(f"Successfully loaded {data_path}. Shape: {df.shape}")
    except FileNotFoundError:
        print(f"Error: {data_path} not found. Please run create_dataset.py first.")
        return

    # --- 2. Feature Engineering & Selection ---
    # Define target and features
    TARGET = 'Deal_Next_Year'
    
    # Exclude identifiers and the raw deal date columns
    features = df.drop(columns=[TARGET, 'Orgnr', 'Juridisk selskapsnavn'])
    target = df[TARGET]

    # One-Hot Encode the NACE code column
    nace_col = 'NACE-bransjekode'
    if nace_col in features.columns:
        print(f"One-hot encoding '{nace_col}'...")
        features = pd.get_dummies(features, columns=[nace_col], prefix='NACE')

    # Replace any remaining inf / -inf / NaN values across all features
    features = features.replace([np.inf, -np.inf], 0).fillna(0)

    # --- 3. Time-Based Splitting ---
    # Split data to train on past and test on future to prevent data leakage
    split_year = 2021
    train_indices = df[df['Year'] < split_year].index
    test_indices = df[df['Year'] >= split_year].index

    X_train = features.loc[train_indices].drop(columns=['Year'])
    y_train = target.loc[train_indices]
    X_test = features.loc[test_indices].drop(columns=['Year'])
    y_test = target.loc[test_indices]
    
    print(f"\nTraining on data before {split_year}: {X_train.shape[0]} samples")
    print(f"Testing on data from {split_year} onwards: {X_test.shape[0]} samples")
    print(f"Positive cases in training set: {y_train.sum()}")
    print(f"Positive cases in testing set: {y_test.sum()}")

    if y_test.sum() == 0:
        print("\nWarning: No positive cases in the test set. Evaluation will be limited.")
        
    # --- 4. Model Training ---
    print("\nTraining RandomForestClassifier with balanced class weights...")
    model = RandomForestClassifier(
        n_estimators=150,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1,
        max_depth=10,
        min_samples_leaf=5
    )
    model.fit(X_train, y_train)

    # --- 5. Evaluation ---
    print("\n--- Model Evaluation ---")
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)

    # ROC AUC
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    print(f"ROC AUC Score: {roc_auc:.4f}")

    # Precision-Recall AUC
    precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
    pr_auc = auc(recall, precision)
    print(f"Precision-Recall AUC Score: {pr_auc:.4f}")

    # Classification Report at default 0.5 threshold
    print("\nClassification Report (threshold=0.5):")
    print(classification_report(y_test, y_pred))

    # Confusion Matrix (threshold=0.5)
    print("Confusion Matrix (threshold=0.5):")
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['No Deal', 'Deal'], yticklabels=['No Deal', 'Deal'])
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    plt.close()

    # --- 5b. Evaluation at Outreach Capacity ---
    outreach_capacity = 500  # number of companies your team can contact
    if outreach_capacity < len(y_test):
        # Determine probability threshold that selects exactly `outreach_capacity` highest-scoring firms
        sorted_idx = np.argsort(y_pred_proba)[::-1]
        top_idx = sorted_idx[:outreach_capacity]
        threshold_cap = y_pred_proba[top_idx[-1]]

        y_pred_cap = (y_pred_proba >= threshold_cap).astype(int)
        precision_cap = precision_score(y_test, y_pred_cap)
        recall_cap = recall_score(y_test, y_pred_cap)
        n_positive_preds = y_pred_cap.sum()

        print(f"\n--- Outreach-capacity evaluation (top {outreach_capacity} firms) ---")
        print(f"Threshold probability: {threshold_cap:.4f}")
        print(f"Predicted positives: {n_positive_preds}")
        print(f"Precision: {precision_cap:.4f} (≈ {precision_cap*100:.2f}%)")
        print(f"Recall: {recall_cap:.4f}  – captures {int(recall_cap * y_test.sum())} of {int(y_test.sum())} true deals")

        # Save ranked list of leads to CSV
        leads_df = df.loc[test_indices].copy()
        leads_df['Deal_Probability'] = y_pred_proba
        leads_topN = leads_df.iloc[top_idx][['Orgnr', 'Juridisk selskapsnavn', 'Deal_Probability']]
        leads_topN.to_csv('top_1000_leads.csv', index=False)
        print("Saved top 1000 leads to 'top_1000_leads.csv'.")

    # --- 6. Feature Importances ---
    print("\n--- Top 20 Feature Importances ---")
    importances = pd.DataFrame({
        'feature': X_train.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False).head(20)
    
    print(importances)
    
    plt.figure(figsize=(10, 8))
    sns.barplot(x='importance', y='feature', data=importances)
    plt.title('Top 20 Feature Importances')
    plt.tight_layout()
    plt.savefig('feature_importances.png')
    plt.close()


if __name__ == '__main__':
    train_and_evaluate() 
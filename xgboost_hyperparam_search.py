import os
import pandas as pd
import numpy as np
from itertools import product
from sklearn.metrics import average_precision_score, precision_recall_curve, auc, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def load_data():
    path = os.path.join('data', 'ml_ready_dataset.csv')
    df = pd.read_csv(path)
    TARGET = 'Deal_Next_Year'
    X = df.drop(columns=[TARGET, 'Orgnr', 'Juridisk selskapsnavn'])
    y = df[TARGET]

    # one-hot NACE
    nace = 'NACE-bransjekode'
    if nace in X.columns:
        X = pd.get_dummies(X, columns=[nace], prefix='NACE')

    X = X.replace([np.inf, -np.inf], 0).fillna(0)
    return df, X, y


def time_split(df, X, y, split_year=2021):
    train_idx = df[df['Year'] < split_year].index
    test_idx = df[df['Year'] >= split_year].index
    X_train = X.loc[train_idx].drop(columns=['Year'])
    y_train = y.loc[train_idx]
    X_test = X.loc[test_idx].drop(columns=['Year'])
    y_test = y.loc[test_idx]
    return X_train, y_train, X_test, y_test, test_idx


def main(outreach_capacity: int = 500):
    print('Loading data…')
    df, X, y = load_data()
    X_train, y_train, X_test, y_test, test_idx = time_split(df, X, y)

    try:
        from xgboost import XGBClassifier  # type: ignore
    except ImportError:
        print('xgboost is not installed.')
        return

    # baseline pos weight
    base_pos_w = (y_train == 0).sum() / max(1, (y_train == 1).sum())

    param_grid = {
        'n_estimators': [600, 800, 1000],
        'learning_rate': [0.03, 0.05, 0.07],
        'max_depth': [6, 8, 10],
        'subsample': [0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0],
        'scale_pos_weight': [base_pos_w, base_pos_w * 2, base_pos_w * 3],
    }

    # limit combinations for runtime: sample 20 random combos
    import random
    all_combos = list(product(param_grid['n_estimators'], param_grid['learning_rate'], param_grid['max_depth'], param_grid['subsample'], param_grid['colsample_bytree'], param_grid['scale_pos_weight']))
    random.seed(42)
    sample_combos = random.sample(all_combos, 20)

    best_score = -1
    best_params = None

    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    print('Beginning parameter search (20 combos, 3-fold CV)…')

    for idx, (n_est, lr, depth, subs, colsub, spw) in enumerate(sample_combos, 1):
        params = {
            'n_estimators': n_est,
            'learning_rate': lr,
            'max_depth': depth,
            'subsample': subs,
            'colsample_bytree': colsub,
            'scale_pos_weight': spw,
            'objective': 'binary:logistic',
            'eval_metric': 'aucpr',
            'n_jobs': -1,
            'random_state': 42,
        }
        model = XGBClassifier(**params)
        fold_scores = []
        for train_idx_cv, val_idx_cv in cv.split(X_train, y_train):
            model.fit(X_train.iloc[train_idx_cv], y_train.iloc[train_idx_cv])
            preds = model.predict_proba(X_train.iloc[val_idx_cv])[:, 1]
            fold_scores.append(average_precision_score(y_train.iloc[val_idx_cv], preds))
        mean_score = np.mean(fold_scores)
        print(f"[{idx}/20] AP={mean_score:.4f} | params {params}")
        if mean_score > best_score:
            best_score = mean_score
            best_params = params

    if best_params is None:
        print('No model trained.')
        return

    print(f"\nBest params based on CV average precision {best_score:.4f}:\n{best_params}")

    # Train best model on full training data
    best_model = XGBClassifier(**best_params)
    best_model.fit(X_train, y_train)

    y_pred_proba = best_model.predict_proba(X_test)[:, 1]
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    precision_curve, recall_curve, _ = precision_recall_curve(y_test, y_pred_proba)
    pr_auc = auc(recall_curve, precision_curve)
    print(f"Test ROC-AUC {roc_auc:.4f} | Test PR-AUC {pr_auc:.4f}")

    # Outreach capacity eval
    sorted_idx = np.argsort(y_pred_proba)[::-1]
    top_idx = sorted_idx[:outreach_capacity]
    threshold_cap = y_pred_proba[top_idx[-1]]
    y_pred_cap = (y_pred_proba >= threshold_cap).astype(int)
    prec_cap = precision_score(y_test, y_pred_cap)
    rec_cap = recall_score(y_test, y_pred_cap)
    print(f"Top {outreach_capacity} threshold {threshold_cap:.4f} | Precision {prec_cap:.4f} | Recall {rec_cap:.4f} (captures {int(rec_cap*y_test.sum())} deals)")

    # Save leads
    leads_df = df.loc[test_idx].copy()
    leads_df['Deal_Probability'] = y_pred_proba
    leads_topN = leads_df.iloc[top_idx][['Orgnr', 'Juridisk selskapsnavn', 'Deal_Probability']]
    leads_topN.to_csv(f'top_{outreach_capacity}_leads_xgboost_tuned.csv', index=False)
    print(f"Saved top leads to 'top_{outreach_capacity}_leads_xgboost_tuned.csv'.")


if __name__ == '__main__':
    main(outreach_capacity=500) 
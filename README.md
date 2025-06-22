# Deal Sourcing Pipeline

This repository contains utilities for preparing company data and training machine learning models to predict potential M&A or investment deals. Data files (CSV/XLSX) are tracked via Git LFS.

## Repository Structure

- `combine_excel.py` – merges multiple Excel transaction files into `data/transactions/combined_transactions.xlsx`.
- `create_dataset.py` – loads company data, board changes and transactions, performs fuzzy matching of company names, engineers features and saves `data/ml_ready_dataset.csv`.
- `match_orgnr.py` – attaches organisation numbers to transaction records using the large `enheter` dataset.
- `train_model.py` – baseline RandomForest model for deal prediction.
- `train_boosted_model.py` – trains a boosted tree model (uses XGBoost if installed, otherwise `HistGradientBoostingClassifier`).
- `xgboost_hyperparam_search.py` – parameter search for an XGBoost model.
- `data/` – placeholder directory for raw and processed datasets.
- `API documentation.json` – reference for the Proff API used to gather company data.

## Requirements

- Python 3.8+
- pandas, numpy
- scikit-learn
- rapidfuzz
- seaborn, matplotlib
- xgboost (optional for boosted models)

Large CSV/XLSX files are stored with Git LFS; ensure LFS is installed:

```bash
git lfs install
```

`.gitattributes` shows the LFS patterns:

```text
*.xlsx filter=lfs diff=lfs merge=lfs -text
*.csv  filter=lfs diff=lfs merge=lfs -text
```

## Usage

1. Place your raw Excel files under `data/companies/` and `data/transactions/`.
2. Combine transactions:
   ```bash
   python combine_excel.py
   ```
3. Build the machine-learning dataset:
   ```bash
   python create_dataset.py
   ```
4. (Optional) Attach organisation numbers:
   ```bash
   python match_orgnr.py
   ```
5. Train and evaluate models:
   ```bash
   python train_model.py              # RandomForest baseline
   python train_boosted_model.py      # XGBoost/HistGradientBoosting
   ```
6. For extensive XGBoost tuning run:
   ```bash
   python xgboost_hyperparam_search.py
   ```

The training scripts output evaluation plots and ranked lead files such as `top_1000_leads.csv`.

## Notes

Actual company and transaction data are not included in this repository. Add your own data files in the `data/` directory before running the scripts.

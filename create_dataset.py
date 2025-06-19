import pandas as pd
import os
from functools import reduce
from rapidfuzz import process, utils
import numpy as np

# Define file paths
COMPANIES_FILE = os.path.join('data', 'companies', 'companies.xlsx')
TRANSACTIONS_FILE = os.path.join('data', 'transactions', 'combined_transactions.xlsx')

def load_data():
    """
    Loads the companies and transactions data from the specified Excel files.
    """
    print("Loading data...")
    
    # Load companies data
    try:
        companies_main_df = pd.read_excel(COMPANIES_FILE, sheet_name='Main')
        print("Successfully loaded 'Main' sheet from companies.xlsx")
        print("Columns in 'Main' sheet:", companies_main_df.columns.tolist())
        print(companies_main_df.head())
    except Exception as e:
        print(f"Error loading 'Main' sheet from {COMPANIES_FILE}: {e}")
        companies_main_df = pd.DataFrame()

    # Load board changes data
    try:
        board_changes_df = pd.read_excel(COMPANIES_FILE, sheet_name='Board changes')
        print("\nSuccessfully loaded 'Board changes' sheet from companies.xlsx")
        print("Columns in 'Board changes' sheet:", board_changes_df.columns.tolist())
        print(board_changes_df.head())
    except Exception as e:
        print(f"Error loading 'Board changes' sheet from {COMPANIES_FILE}: {e}")
        board_changes_df = pd.DataFrame()

    # Load transactions data
    try:
        transactions_df = pd.read_excel(TRANSACTIONS_FILE)
        print("\nSuccessfully loaded combined_transactions.xlsx")
        print("Columns in 'transactions' sheet:", transactions_df.columns.tolist())
        print(transactions_df.head())
    except Exception as e:
        print(f"Error loading {TRANSACTIONS_FILE}: {e}")
        transactions_df = pd.DataFrame()

    return companies_main_df, board_changes_df, transactions_df

def normalize_name(name):
    """
    Simple normalization for company names.
    Converts to lowercase and removes common suffixes.
    """
    if not isinstance(name, str):
        return ''
    name = name.lower()
    # Remove common Norwegian corporate identifiers
    suffixes = [' as', ' asa', ' ab', ' oy', ' oyj']
    for suffix in suffixes:
        if name.endswith(suffix):
            name = name[:-len(suffix)]
    return name.strip()

def fuzzy_match_companies(companies_df, transactions_df, score_cutoff=90):
    """
    Performs fuzzy matching between transaction target companies and the main company list
    using the high-performance rapidfuzz library.
    """
    # Create a clean version of the company name for matching
    companies_df['Clean Name'] = companies_df['Juridisk selskapsnavn'].apply(utils.default_process)
    company_names_map = companies_df[['Clean Name', 'Juridisk selskapsnavn']].set_index('Clean Name')['Juridisk selskapsnavn']
    
    unique_clean_company_names = companies_df['Clean Name'].unique()
    
    transaction_targets = transactions_df['Target Company'].unique()
    
    total_targets = len(transaction_targets)
    print(f"Starting optimized fuzzy matching for {total_targets} unique transaction targets...")
    
    mapping = {}
    processed_count = 0
    # Process transaction targets in a batch for efficiency
    for target in transaction_targets:
        # Also clean the target name before matching
        clean_target = utils.default_process(target)
        match = process.extractOne(clean_target, unique_clean_company_names, score_cutoff=score_cutoff)
        
        if match:
            # Map back from the clean name to the original juridisk name
            mapping[target] = company_names_map[match[0]]
        
        processed_count += 1
        if processed_count % 1000 == 0:
            print(f"  Processed {processed_count}/{total_targets} transaction targets...")
            
    transactions_df['Matched Company Name'] = transactions_df['Target Company'].map(mapping)
    
    print(f"\nFuzzy matching mapped {len(mapping)} transaction targets to company names.")
    
    # Drop the temporary column after use
    companies_df.drop(columns=['Clean Name'], inplace=True)
    
    return transactions_df

def preprocess_transactions(transactions_df):
    """
    Preprocesses the transactions dataframe.
    """
    if transactions_df.empty:
        return transactions_df
    
    print("\nPreprocessing transactions data...")
    # Convert date columns to datetime objects
    transactions_df['Announced Date'] = pd.to_datetime(transactions_df['Announced Date'], errors='coerce')
    transactions_df['Completed Date'] = pd.to_datetime(transactions_df['Completed Date'], errors='coerce')
    
    # Handle missing dates - using announced date as the primary date
    transactions_df['Transaction Date'] = transactions_df['Announced Date'].fillna(transactions_df['Completed Date'])
    
    # Drop rows where transaction date could not be determined
    transactions_df.dropna(subset=['Transaction Date'], inplace=True)
    
    # Clean target company names for better matching
    transactions_df['Target Company'] = transactions_df['Target Company'].str.strip()
    
    print("Transactions data preprocessed.")
    return transactions_df

def preprocess_companies(companies_df):
    """
    Preprocesses the companies dataframe.
    """
    if companies_df.empty:
        return companies_df
    print("\nPreprocessing companies data...")
    # Clean company names
    companies_df['Juridisk selskapsnavn'] = companies_df['Juridisk selskapsnavn'].str.strip()
    
    # Drop unnamed columns that might appear from Excel
    companies_df = companies_df.loc[:, ~companies_df.columns.str.contains('^Unnamed')]
    
    print("Companies data preprocessed.")
    return companies_df

def preprocess_board_changes(board_changes_df):
    """
    Preprocesses the board changes dataframe.
    """
    if board_changes_df.empty:
        return board_changes_df
    print("\nPreprocessing board changes data...")
    # Convert date columns to datetime
    board_changes_df['Tiltrådt'] = pd.to_datetime(board_changes_df['Tiltrådt'], errors='coerce')

    # Clean company names
    board_changes_df['Juridisk selskapsnavn'] = board_changes_df['Juridisk selskapsnavn'].str.strip()
    
    print("Board changes data preprocessed.")
    return board_changes_df

def merge_data(companies_df, transactions_df):
    """
    Robustly merge the companies and transactions dataframes (after fuzzy matching) and
    print a concise match-rate report.  All suffix issues are handled immediately
    after the merge so later code can rely on clean column names.
    """
    if companies_df.empty or transactions_df.empty:
        print("\nSkipping merge – one of the frames is empty.")
        return pd.DataFrame()

    print("\nMerging data …")

    # Ensure keys are plain strings (avoids the previous *unhashable* error)
    companies_df['Juridisk selskapsnavn'] = companies_df['Juridisk selskapsnavn'].astype(str)
    transactions_df['Matched Company Name'] = transactions_df['Matched Company Name'].astype(str)

    merged_df = pd.merge(
        companies_df,
        transactions_df,
        left_on="Juridisk selskapsnavn",
        right_on="Matched Company Name",
        how="left",
    )

    # ---- tidy up column names created by pandas on overlapping cols ----
    if "Juridisk selskapsnavn_x" in merged_df.columns:
        merged_df = merged_df.rename(columns={"Juridisk selskapsnavn_x": "Juridisk selskapsnavn"})
    if "Juridisk selskapsnavn_y" in merged_df.columns:
        merged_df = merged_df.drop(columns=["Juridisk selskapsnavn_y"])
    # -------------------------------------------------------------------

    # ---------- match-rate report ----------
    matched_tx = merged_df["Transaction Date"].notna().sum()
    total_tx   = len(transactions_df)
    uniq_tgt   = transactions_df["Target Company"].nunique()
    uniq_cmp   = merged_df.loc[merged_df["Transaction Date"].notna(), "Juridisk selskapsnavn"].nunique()

    print("\n--- Match-rate report ---")
    print(f"transactions in file      : {total_tx:,}")
    print(f"unique target companies   : {uniq_tgt:,}")
    print(f"transactions matched      : {matched_tx:,}")
    print(f"companies with ≥1 match    : {uniq_cmp:,}")
    if total_tx > 0:
        print(f"match percentage          : {matched_tx/total_tx:6.2%}")
    print("---------------------------\n")
    # ---------------------------------------

    return merged_df

def create_feature_dataset(merged_df, board_changes_df):
    """
    Creates a feature-rich dataset for machine learning.
    This version ensures no companies are dropped.
    """
    if merged_df.empty:
        print("\nSkipping feature dataset creation due to empty merged dataframe.")
        return pd.DataFrame()

    print("\nCreating feature dataset...")

    # Create a target variable: 1 if a transaction occurred, 0 otherwise
    merged_df['Transaction'] = merged_df['Transaction Date'].notna().astype(int)
    
    # Identify the base columns (non-financials) and financial columns
    base_cols = ['Orgnr', 'Juridisk selskapsnavn', 'Transaction', 'Transaction Date']
    nace_col_name = 'NACE-bransjekode'
    if nace_col_name in merged_df.columns:
        base_cols.append(nace_col_name)

    financial_metric_stems = ['Sum driftsinnt.', 'Driftsres.', 'Vareforbr.', 'Lønnskostnader', 'Avskr. varige driftsmidl.', 'Andre driftskostnader', 'Sum driftskostn.']
    
    all_financial_cols = [col for col in merged_df.columns if any(col.startswith(stem) for stem in financial_metric_stems)]

    # Reshape each financial metric from wide to long and collect them
    yearly_data_frames = []
    for stem in financial_metric_stems:
        metric_cols = [col for col in all_financial_cols if col.startswith(stem)]
        if not metric_cols:
            continue
        
        melted_metric = pd.melt(
            merged_df, 
            id_vars=['Orgnr'], 
            value_vars=metric_cols, 
            var_name='Metric_Year', 
            value_name=stem.strip()
        )
        melted_metric['Year'] = pd.to_numeric(melted_metric['Metric_Year'].str.rsplit(', ', n=1, expand=True)[1])
        melted_metric.drop(columns=['Metric_Year'], inplace=True)
        # Remove potential duplicate rows for the same company and year (can otherwise lead to Cartesian explosion later)
        melted_metric = melted_metric.drop_duplicates(subset=['Orgnr', 'Year'])
        yearly_data_frames.append(melted_metric)

    # Merge all the long-format financial dataframes together
    if not yearly_data_frames:
        print("No financial data to process.")
        return pd.DataFrame()
        
    features_df = reduce(lambda left, right: pd.merge(left, right, on=['Orgnr', 'Year'], how='outer'), yearly_data_frames)

    # Ensure we still have at most one row per company-year after merging all metrics
    pre_dedupe_rows = len(features_df)
    features_df = features_df.drop_duplicates(subset=['Orgnr', 'Year'])
    if len(features_df) < pre_dedupe_rows:
        print(f"Removed {pre_dedupe_rows - len(features_df):,} duplicate rows after consolidating metrics (based on Orgnr-Year).")

    # Merge back with base company info to ensure all companies are kept
    company_info = merged_df[base_cols].drop_duplicates(subset=['Orgnr'])
    features_df = pd.merge(company_info, features_df, on='Orgnr', how='left')

    # Board changes feature engineering
    if not board_changes_df.empty:
        board_changes_df['Year'] = board_changes_df['Tiltrådt'].dt.year
        board_changes_count = board_changes_df.groupby(['Orgnr', 'Year']).size().reset_index(name='BoardChanges_LastYear')
        
        features_df['PreviousYear'] = features_df['Year'] - 1
        features_df = pd.merge(features_df, board_changes_count.rename(columns={'Year': 'PreviousYear'}), on=['Orgnr', 'PreviousYear'], how='left')
        features_df['BoardChanges_LastYear'] = features_df['BoardChanges_LastYear'].fillna(0)
        features_df.drop(columns=['PreviousYear'], inplace=True)

    # Filter out data points that are after a transaction has occurred
    features_df = features_df[ (features_df['Transaction'] == 0) | (features_df['Year'] < features_df['Transaction Date'].dt.year) | (features_df['Transaction Date'].isna())].copy()
    
    # Final safety check – drop any remaining duplicates just in case
    final_pre_dedupe = len(features_df)
    features_df = features_df.drop_duplicates(subset=['Orgnr', 'Year'])
    if len(features_df) < final_pre_dedupe:
        print(f"Dropped {final_pre_dedupe - len(features_df):,} duplicate rows during final deduplication step.")

    print("Feature dataset created.")
    return features_df

def prepare_for_training(features_df):
    """
    Prepares the dataset for ML training with specific logic for predicting a deal "next year".

    This function will:
    1.  Define the target variable `Deal_Next_Year`.
    2.  Engineer time-series features (YoY growth).
    3.  Create financial ratios (e.g., profit margin).
    4.  Handle skew and missing data.
    5.  Select and clean the final feature set.
    """
    if features_df.empty:
        print("Skipping training data preparation due to empty dataframe.")
        return pd.DataFrame()

    print("\nPreparing data for 'next year' prediction model...")

    # Sort data chronologically for each company
    features_df = features_df.sort_values(['Orgnr', 'Year']).copy()

    # 1. Create the specific target variable: Deal_Next_Year
    # The transaction year is Transaction Date.dt.year. The row for the year *before* that should be the positive case.
    features_df['Deal_Year'] = features_df['Transaction Date'].dt.year
    features_df['Deal_Next_Year'] = (features_df['Year'] == features_df['Deal_Year'] - 1).astype(int)

    # 2. Financial Feature Engineering (YoY Growth)
    financial_cols = ['Sum driftsinnt.', 'Driftsres.', 'Vareforbr.', 'Lønnskostnader', 'Avskr. varige driftsmidl.', 'Andre driftskostnader', 'Sum driftskostn.']
    
    for col in financial_cols:
        # Use pct_change after grouping by company to get YoY growth
        growth_col = f'{col}_YoY_Growth'
        features_df[growth_col] = features_df.groupby('Orgnr')[col].pct_change(fill_method=None) * 100
        # Replace infinite values that occur from division by zero (e.g., 0 to 100)
        features_df[growth_col] = features_df[growth_col].replace([np.inf, -np.inf], 0)
        
    # 3. Create Financial Ratios
    # Use np.divide to handle division by zero gracefully, resulting in NaN
    features_df['ProfitMargin'] = np.divide(features_df['Driftsres.'], features_df['Sum driftsinnt.']) * 100

    # 4. Handle Skew and Missing Data
    # First, fill any NaNs in the source financial columns before transformations.
    features_df[financial_cols] = features_df[financial_cols].fillna(0)
    
    # Log-transform skewed financial columns to make them more normally distributed
    for col in financial_cols:
        features_df[col] = np.sign(features_df[col]) * np.log1p(np.abs(features_df[col]))

    # Final catch-all: Replace any remaining inf/NaN in all feature columns
    # This ensures that no invalid values are passed to the model.
    feature_cols_for_cleaning = [f'{c}_YoY_Growth' for c in financial_cols] + ['ProfitMargin'] + financial_cols
    for col in feature_cols_for_cleaning:
        if col in features_df.columns:
            features_df[col] = features_df[col].replace([np.inf, -np.inf], 0)
            features_df[col] = features_df[col].fillna(0)

    # Fill NaNs that resulted from pct_change (for the first year of data) or division by zero
    # Simple zero-filling is a start; more complex imputation could be used later.
    cols_to_fill = [f'{col}_YoY_Growth' for col in financial_cols] + ['ProfitMargin']
    features_df[cols_to_fill] = features_df[cols_to_fill].fillna(0)

    # 5. Select and Clean Final Feature Set
    # We assume 'NACE-bransjekode' is the NACE code column.
    # It might have a different name, which would need to be adjusted.
    feature_columns = [
        'Year', 'BoardChanges_LastYear'
    ] + financial_cols + [f'{c}_YoY_Growth' for c in financial_cols] + ['ProfitMargin']
    
    # Check if NACE code column exists and add it
    nace_col_name = 'NACE-bransjekode'
    if nace_col_name not in features_df.columns:
        print(f"Warning: NACE code column '{nace_col_name}' not found. It will be excluded from features.")
    else:
        feature_columns.append(nace_col_name)
        # Clean NACE codes: treat as category, fill missing with a placeholder
        features_df[nace_col_name] = features_df[nace_col_name].astype(str).fillna('Missing').str.strip()


    target_column = 'Deal_Next_Year'
    
    final_cols = ['Orgnr', 'Juridisk selskapsnavn', 'Deal_Year', target_column] + feature_columns
    
    # Ensure all selected columns exist in the dataframe before trying to select them
    final_cols_exist = [col for col in final_cols if col in features_df.columns]
    ml_ready_df = features_df[final_cols_exist].copy()

    # Drop any rows that are from the year of a deal or after, as we can't use them for predicting that deal
    if 'Deal_Year' in ml_ready_df.columns:
        ml_ready_df = ml_ready_df[ (ml_ready_df['Deal_Year'].isna()) | (ml_ready_df['Year'] < ml_ready_df['Deal_Year']) ]
        # The Deal_Year column is a helper and not needed for the model, so drop it.
        ml_ready_df = ml_ready_df.drop(columns=['Deal_Year'])

    print("Training data preparation complete.")
    print(f"Positive cases (Deal_Next_Year=1): {ml_ready_df[target_column].sum()}")
    print(f"Dataset shape for training: {ml_ready_df.shape}")
    
    return ml_ready_df

if __name__ == '__main__':
    companies_df, board_changes_df, transactions_df = load_data()

    # Preprocess all dataframes
    transactions_df = preprocess_transactions(transactions_df)
    companies_df = preprocess_companies(companies_df)
    board_changes_df = preprocess_board_changes(board_changes_df)

    # Perform Fuzzy Matching before merging
    transactions_df = fuzzy_match_companies(companies_df, transactions_df)

    # Merge data
    # For the full dataset creation, we need to consider the timeline of events.
    # This is a simple merge for now.
    merged_df = merge_data(companies_df, transactions_df)
    
    if not merged_df.empty:
        print("\nMerged Data Head:")
        print(merged_df.head())

    # Create the final dataset
    final_dataset = create_feature_dataset(merged_df, board_changes_df)

    if not final_dataset.empty:
        print("\nFinal Dataset Head:")
        print(final_dataset.head())
        
        # Prepare the data for machine learning
        ml_dataset = prepare_for_training(final_dataset)
        
        # Save the dataset to a new file
        output_path = os.path.join('data', 'ml_ready_dataset.csv')
        ml_dataset.to_csv(output_path, index=False)
        print(f"\nMachine learning-ready dataset saved to {output_path}")

    # Next steps will be implemented here
    # 1. Preprocess data (e.g., clean names, convert dates) - In Progress
    # 2. Match transactions to companies - In Progress
    # 3. Create labeled dataset with historical features - Done
    print("\nData loading and initial preprocessing complete.") 
import pandas as pd
import re
from pathlib import Path

# Paths to input files (relative to repository root)
TRANSACTIONS_PATH = Path("data/transactions/combined_transactions.xlsx")
ENHETER_PATH = Path("data/transactions/enheter_2025-06-16T04-23-21.823559761.csv")
OUTPUT_PATH = Path("data/transactions/combined_transactions_with_orgnr.xlsx")


def clean_company_name(name: str) -> str:
    """Standardise a company name for matching.

    1. Remove percentage ownership strings such as "(100%)", "(45%)".
    2. Strip leading/trailing whitespace.
    3. Convert to lowercase for case-insensitive matching.
    """
    if pd.isna(name):
        return ""

    # Remove ANY parenthetical clause like "(100%)", "(45% Stake)", "(Rental operations)" etc.
    cleaned = re.sub(r"\s*\([^)]*\)", "", str(name))

    return cleaned.strip().lower()


def load_enheter_mapping(enheter_path: Path) -> dict:
    """Load the enheter CSV and build a mapping from cleaned navn to organisasjonsnummer."""
    # Read only the columns we need to minimise memory usage
    enheter_df = pd.read_csv(
        enheter_path,
        usecols=["organisasjonsnummer", "navn"],
        dtype={"organisasjonsnummer": str, "navn": str},
        low_memory=False,
        encoding="utf-8",
    )

    # Clean company names
    enheter_df["clean_name"] = enheter_df["navn"].map(clean_company_name)

    # Drop rows where clean_name is empty
    enheter_df = enheter_df[enheter_df["clean_name"] != ""]

    # Build dictionary (if duplicates exist, keep the first occurrence)
    mapping = (
        enheter_df.drop_duplicates("clean_name")
        .set_index("clean_name")["organisasjonsnummer"]
        .to_dict()
    )

    return mapping


def main():
    # Load transactions data
    trans_df = pd.read_excel(TRANSACTIONS_PATH)

    # Normalise column names to lower case for flexible matching
    trans_df.columns = [col.lower().strip() for col in trans_df.columns]

    if "target company" not in trans_df.columns:
        raise KeyError(
            "Could not find a 'target company' column in combined_transactions.xlsx.\n"
            "Available columns after normalisation: " + ", ".join(trans_df.columns)
        )

    # Clean target company names
    trans_df["clean_name"] = trans_df["target company"].map(clean_company_name)

    # Load mapping from enheter
    print("Loading enheter dataset (this may take a while)…")
    mapping = load_enheter_mapping(ENHETER_PATH)

    # Map organisasjonsnummer
    trans_df["organisasjonsnummer"] = trans_df["clean_name"].map(mapping)

    # Optionally, flag rows that did not match
    trans_df["orgnr_match_found"] = ~trans_df["organisasjonsnummer"].isna()

    # Drop helper column before saving
    trans_df = trans_df.drop(columns=["clean_name"])

    # Save to Excel
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    trans_df.to_excel(OUTPUT_PATH, index=False)
    print(f"Saved updated transactions with organisasjonsnummer to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main() 
import pandas as pd
import os

# Define the path to the directory containing the Excel files
data_dir = 'data/transactions'

# Get a list of all .xlsx files in the directory
excel_files = [f for f in os.listdir(data_dir) if f.endswith('.xlsx') and not f.startswith('combined')]

# Create a list to hold the dataframes
all_data = []

# Loop through the files and read them into pandas dataframes
for file in excel_files:
    file_path = os.path.join(data_dir, file)
    try:
        df = pd.read_excel(file_path)
        all_data.append(df)
        print(f"Successfully read {file}")
    except Exception as e:
        print(f"Could not read file {file}. Error: {e}")

# Concatenate all the dataframes
if all_data:
    combined_df = pd.concat(all_data, ignore_index=True)

    # Define the output file path
    output_file = os.path.join(data_dir, 'combined_transactions.xlsx')

    # Write the combined dataframe to a new Excel file
    try:
        combined_df.to_excel(output_file, index=False)
        print(f"\nSuccessfully combined {len(all_data)} files into {output_file}")
        print(f"The combined file has {len(combined_df)} rows.")
    except Exception as e:
        print(f"\nCould not write to file {output_file}. Error: {e}")
else:
    print("No dataframes to concatenate.") 
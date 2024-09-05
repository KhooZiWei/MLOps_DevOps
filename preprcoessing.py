import pandas as pd

salesDF = pd.read_csv(r'C:\Users\ZKhoo\OneDrive - SLB\MLOps_DevOps\data\superstore_dataset.csv', encoding='latin1')

# Data Preprocessing
def remove_unnecessary_columns(df, columns_to_keep):
    # Select only the necessary columns from the DataFrame.
    df = df[columns_to_keep]
    return df

def rename_column(df, old_column_name, new_column_name):
    # Rename the specified column.
    df = df.rename(columns={old_column_name: new_column_name})
    return df

def parse_date(date_str):
    # Attempt to parse a date string with multiple formats.
    formats = ['%d/%m/%Y', '%d-%m-%Y']
    for fmt in formats:
        try:
            return pd.to_datetime(date_str, format=fmt)
        except ValueError:
            continue
    return pd.NaT

def transform_date_column(df, column_name):
    # Apply the parse_date function to each entry in the column.
    df[column_name] = df[column_name].apply(parse_date)
    return df

# Remove unnecessary columns
columns_to_keep = ['Order Date', 'Sales']
salesDF = remove_unnecessary_columns(salesDF, columns_to_keep)

# Rename 'Order Date' to 'Date'
salesDF = rename_column(salesDF, 'Order Date', 'Date')

# Transform the 'Date' column
salesDF = transform_date_column(salesDF, 'Date')

# saving the dataframe
salesDF.to_csv(r'C:\Users\ZKhoo\OneDrive - SLB\MLOps_DevOps\data\cleaned_superstore.csv', index=False)


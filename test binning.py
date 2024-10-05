# test binning

import pandas as pd
from src.binning import DataBinner
# Sample DataFrame
data = {
    'float_col': list(range(100)),
    'category_col': ['A'] * 50 + ['B'] * 30 + ['C'] * 20,
    'int_col': list(range(100)),
    'datetime_col': pd.date_range(start='2021-01-01', periods=100, freq='D')
}

df = pd.DataFrame(data)
df['category_col'] = df['category_col'].astype('category')

# Define bin counts
bin_dict = {
    'float_col': 100,   # Exact number of unique values
    'category_col': 2,  # Reduce from 3 categories to 2
    'int_col': 50,      # Fewer bins than unique values
    'datetime_col': 10  # Example bin count for datetime
}

# Initialize DataBinner
binner = DataBinner(df, method='equal width')

# Perform binning
binned_df, binned_columns = binner.bin_columns(bin_dict)

print("Binned DataFrame:")
print(binned_df.head())

print("\nBinned Columns:")
print(binned_columns)

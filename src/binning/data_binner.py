# data_binner.py

import pandas as pd
from typing import Tuple, Dict, List
import math

class DataBinner:
    """
    A class to bin specified columns in a Pandas DataFrame based on provided bin counts and binning methods.
    """
    
    def __init__(self, Data: pd.DataFrame, method: str = 'equal width'):
        """
        Initializes the DataBinner with the original DataFrame and binning method.
        """
        self.original_df = Data.copy()
        self.binned_df = pd.DataFrame()
        self.binned_columns = {
            'datetime': [],
            'integer': [],
            'float': [],
            'category_grouped': [],
            'unsupported': []
        }
        self.method = method.lower()
        self._validate_method()
    
    def _validate_method(self):
        """
        Validates the binning method. Raises a ValueError if the method is unsupported.
        """
        supported_methods = ['equal width', 'quantile']
        if self.method not in supported_methods:
            raise ValueError(f"Unsupported binning method '{self.method}'. Supported methods are: {supported_methods}")
    
    def bin_columns(
        self,
        bin_dict: Dict[str, int]
    ) -> Tuple[pd.DataFrame, Dict[str, List[str]]]:
        """
        Bins specified columns in the DataFrame based on the provided bin counts and binning method.
        """
        # Initialize dictionary to categorize binned columns
        self.binned_columns = {
            'datetime': [],
            'integer': [],
            'float': [],
            'category_grouped': [],
            'unsupported': []
        }
        # Create a copy of the DataFrame to avoid modifying the original data
        Bin_Data = self.original_df.copy()

        for col, bins in bin_dict.items():
            if col not in Bin_Data.columns:
                print(f"⚠️ Column '{col}' does not exist in the DataFrame. Skipping.")
                continue

            try:
                if pd.api.types.is_datetime64_any_dtype(Bin_Data[col]):
                    # Binning datetime columns using pd.cut or pd.qcut based on method
                    Bin_Data[col] = self._bin_column(Bin_Data[col], bins, self.method)
                    self.binned_columns['datetime'].append(col)

                elif pd.api.types.is_integer_dtype(Bin_Data[col]):
                    Bin_Data[col] = self._bin_column(Bin_Data[col], bins, self.method)
                    self.binned_columns['integer'].append(col)

                elif pd.api.types.is_float_dtype(Bin_Data[col]):
                    Bin_Data[col] = self._bin_column(Bin_Data[col], bins, self.method)
                    self.binned_columns['float'].append(col)
                
                elif pd.api.types.is_categorical_dtype(Bin_Data[col]) or pd.api.types.is_object_dtype(Bin_Data[col]):
                    # Group categorical columns into specified number of bins
                    Bin_Data[col] = self._bin_categorical_column(Bin_Data[col], bins)
                    self.binned_columns['category_grouped'].append(col)
                else:
                    print(f"Column '{col}' has unsupported dtype '{Bin_Data[col].dtype}'. Skipping.")
                    self.binned_columns['unsupported'].append(col)

            except Exception as e:
                # Detailed error messages based on column type
                if pd.api.types.is_datetime64_any_dtype(Bin_Data[col]):
                    print(f"Failed to bin datetime column '{col}': {e}")
                elif pd.api.types.is_integer_dtype(Bin_Data[col]):
                    print(f"Failed to bin integer column '{col}': {e}")
                elif pd.api.types.is_float_dtype(Bin_Data[col]):
                    print(f"Failed to bin float column '{col}': {e}")
                elif pd.api.types.is_categorical_dtype(Bin_Data[col]) or pd.api.types.is_object_dtype(Bin_Data[col]):
                    print(f"Failed to bin category column '{col}': {e}")
                else:
                    print(f"Failed to bin column '{col}': {e}")
                self.binned_columns['unsupported'].append(col)

        # Retain only the successfully binned columns
        successfully_binned = (
            self.binned_columns['datetime'] +
            self.binned_columns['integer'] +
            self.binned_columns['float'] +
            self.binned_columns['category_grouped']
        )
        self.binned_df = Bin_Data[successfully_binned]

        return self.binned_df, self.binned_columns

    def _bin_column(self, series: pd.Series, bins: int, method: str) -> pd.Series:
        """
        Bins a single column using the specified method and returns integer labels as categorical.

        Parameters:
            series (pd.Series): The column to bin.
            bins (int): The number of bins.
            method (str): The binning method ('equal width' or 'quantile').

        Returns:
            pd.Series: The binned column as a categorical Series.
        """
        unique_values = series.nunique(dropna=True)
        print(f"Column '{series.name}': {unique_values} unique values requested {bins} bins.")

        if bins > unique_values:
            print(f"⚠️ Requested bin count {bins} exceeds unique values {unique_values} for column '{series.name}'. Adjusting bin count to {unique_values}.")
            bins = unique_values

        if bins == unique_values:
            # Assign each unique value to its own bin
            # This ensures each unique float value is in a separate bin
            print(f"Assigning each unique value in column '{series.name}' to its own bin.")
            # Create a mapping from unique value to a unique bin number
            sorted_unique = series.dropna().unique()
            sorted_unique.sort()
            value_to_bin = {value: idx for idx, value in enumerate(sorted_unique)}
            binned_series = series.map(value_to_bin).astype('category')
            return binned_series
        else:
            if method == 'equal width':
                binned = pd.cut(
                    series,
                    bins=bins,
                    labels=False,
                    duplicates='drop'
                )
            elif method == 'quantile':
                # Handle case where number of unique values is less than bins for quantile
                if unique_values < bins:
                    print(f"⚠️ Not enough unique values for quantile binning on column '{series.name}'. Using {unique_values} bins instead of {bins}.")
                    bins = unique_values
                binned = pd.qcut(
                    series,
                    q=bins,
                    labels=False,
                    duplicates='drop'
                )
            else:
                # This should not happen due to validation in __init__
                raise ValueError(f"Unsupported binning method '{method}'.")

            # After binning, check if the number of unique bins matches the requested bins
            actual_bins = binned.nunique(dropna=True)
            print(f"Column '{series.name}': Requested {bins} bins, but got {actual_bins} unique binned values.")

            return binned.astype('category')

    def _bin_categorical_column(self, series: pd.Series, bins: int) -> pd.Series:
        """
        Groups categorical column into specified number of bins based on frequency.

        Parameters:
            series (pd.Series): The categorical column to bin.
            bins (int): The desired number of bins (groups).

        Returns:
            pd.Series: The binned categorical column.
        """
        if not pd.api.types.is_categorical_dtype(series):
            series = series.astype('category')

        category_counts = series.value_counts().sort_values(ascending=False)
        unique_categories = len(category_counts)

        if bins >= unique_categories:
            print(f"⚠️ Requested bin count {bins} is greater than or equal to unique categories {unique_categories} for column '{series.name}'. No binning applied.")
            return series

        # Calculate the number of categories per bin
        categories_per_bin = math.ceil(unique_categories / bins)
        grouped_categories = []
        current_bin = 1
        for i, category in enumerate(category_counts.index):
            if (i > 0) and (i % categories_per_bin == 0) and (current_bin < bins):
                current_bin += 1
            grouped_categories.append((category, current_bin))

        # Create a mapping from category to bin number
        category_to_bin = dict(grouped_categories)
        binned_series = series.map(category_to_bin).astype('category')
        binned_series.name = series.name

        return binned_series

    def get_binned_data(self) -> pd.DataFrame:
        """
        Retrieves the binned DataFrame.
        """
        return self.binned_df.copy()

    def get_binned_columns(self) -> Dict[str, List[str]]:
        """
        Retrieves the categorization of binned columns by data type.
        """
        return self.binned_columns.copy()

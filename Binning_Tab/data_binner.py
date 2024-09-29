# data_binner.py

import pandas as pd
from typing import Tuple, Dict, List


class DataBinner:
    """
    A class to bin specified columns in a Pandas DataFrame based on provided bin counts and binning methods.

    Attributes:
        original_df (pd.DataFrame): The original DataFrame.
        binned_df (pd.DataFrame): The DataFrame after binning specified columns.
        binned_columns (dict): Categorization of binned columns by data type.
            Example:
                {
                    'datetime': ['Date1', 'Date2'],
                    'integer': ['Age', 'Salary'],
                    'float': ['Price', 'Rating'],
                    'unsupported': ['Category']
                }
        method (str): The binning method to use ('equal width' or 'quantile').
    """

    def __init__(self, Data: pd.DataFrame, method: str = 'equal width'):
        """
        Initializes the DataBinner with the original DataFrame and binning method.

        Parameters:
            Data (pd.DataFrame): The original DataFrame to be binned.
            method (str): The binning method to use. Supported methods are 'equal width' and 'quantile'.
                          Defaults to 'equal width'.
        """
        self.original_df = Data.copy()
        self.binned_df = pd.DataFrame()
        self.binned_columns = {
            'datetime': [],
            'integer': [],
            'float': [],
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

        Parameters:
            bin_dict (dict): A dictionary where keys are column names and values are the number of bins.

        Returns:
            Tuple containing:
                - binned_df (pd.DataFrame): DataFrame with binned columns.
                - binned_columns (dict): Dictionary categorizing binned columns by data type.
                  Example:
                      {
                          'datetime': ['Date1', 'Date2'],
                          'integer': ['Age', 'Salary'],
                          'float': ['Price', 'Rating'],
                          'unsupported': ['Category']
                      }
        """
        # Initialize dictionary to categorize binned columns
        self.binned_columns = {
            'datetime': [],
            'integer': [],
            'float': [],
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
                else:
                    print(f"Failed to bin column '{col}': {e}")
                self.binned_columns['unsupported'].append(col)

        # Retain only the successfully binned columns
        successfully_binned = (
            self.binned_columns['datetime'] +
            self.binned_columns['integer'] +
            self.binned_columns['float']
        )
        self.binned_df = Bin_Data[successfully_binned]

        return self.binned_df, self.binned_columns

    def _bin_column(self, series: pd.Series, bins: int, method: str) -> pd.Series:
        """
        Bins a single column using the specified method.

        Parameters:
            series (pd.Series): The column to bin.
            bins (int): The number of bins.
            method (str): The binning method ('equal width' or 'quantile').

        Returns:
            pd.Series: The binned column as a categorical type.
        """
        if method == 'equal width':
            return pd.cut(
                series,
                bins=bins,
                duplicates='drop'
            )
        elif method == 'quantile':
            return pd.qcut(
                series,
                q=bins,
                duplicates='drop'
            )
        else:
            # This should not happen due to validation in __init__
            raise ValueError(f"Unsupported binning method '{method}'.")

    def get_binned_data(self) -> pd.DataFrame:
        """
        Retrieves the binned DataFrame.

        Returns:
            pd.DataFrame: DataFrame containing only the binned columns.
        """
        return self.binned_df.copy()

    def get_binned_columns(self) -> Dict[str, List[str]]:
        """
        Retrieves the categorization of binned columns by data type.

        Returns:
            dict: Dictionary categorizing binned columns.
        """
        return self.binned_columns.copy()

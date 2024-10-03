# src/binning/k_anonymity_binner.py

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from copy import deepcopy

class KAnonymityBinner:
    """
    A class to perform automated binning of data columns to achieve k-anonymity.
    """

    def __init__(self, data: pd.DataFrame, k: int = 5, method: str = 'quantile'):
        """
        Initialize the KAnonymityBinner.

        Parameters:
            data (pd.DataFrame): The DataFrame containing columns to be binned.
            k (int): The desired level of anonymity.
            method (str): The binning method ('quantile' or 'equal width').
        """
        self.data = data.copy()
        self.k = k
        self.method = method.lower()
        self.binned_data = self.data.copy()
        self._validate_method()

    def _validate_method(self):
        supported_methods = ['quantile', 'equal width']
        if self.method not in supported_methods:
            raise ValueError(f"Unsupported binning method '{self.method}'. Supported methods are: {supported_methods}")

    def perform_binning(self) -> pd.DataFrame:
        """
        Perform automated binning to achieve k-anonymity.

        Returns:
            pd.DataFrame: The binned DataFrame.
        """
        # Initialize the binning configuration
        bin_config = {col: self._get_initial_bins(col) for col in self.data.columns}

        # Iteratively adjust bins to achieve k-anonymity
        success = False
        max_iterations = 100  # To prevent infinite loops
        iteration = 0

        while not success and iteration < max_iterations:
            # Apply current bin configuration
            temp_binned_data = self._apply_binning(bin_config)

            # Check if k-anonymity is achieved
            if self._check_k_anonymity(temp_binned_data):
                success = True
                self.binned_data = temp_binned_data
            else:
                # Adjust binning configuration
                bin_config = self._adjust_bins(bin_config)
                iteration += 1

        if not success:
            raise ValueError("Unable to achieve k-anonymity with the given data and parameters.")

        return self.binned_data

    def _get_initial_bins(self, column: str) -> int:
        """
        Determine the initial number of bins for a column.

        Parameters:
            column (str): The column name.

        Returns:
            int: The initial number of bins.
        """
        unique_values = self.data[column].nunique()
        return min(10, unique_values)  # Start with up to 10 bins

    def _apply_binning(self, bin_config: Dict[str, int]) -> pd.DataFrame:
        """
        Apply binning to the data using the current bin configuration.

        Parameters:
            bin_config (Dict[str, int]): Current binning configuration.

        Returns:
            pd.DataFrame: The binned DataFrame.
        """
        binned_data = pd.DataFrame()

        for col, bins in bin_config.items():
            series = self.data[col]
            if pd.api.types.is_numeric_dtype(series) or pd.api.types.is_datetime64_any_dtype(series):
                binned_series = self._bin_column(series, bins)
                binned_data[col] = binned_series
            else:
                # For non-numeric types, keep as is
                binned_data[col] = series

        return binned_data

    def _bin_column(self, series: pd.Series, bins: int) -> pd.Series:
        """
        Bin a single column.

        Parameters:
            series (pd.Series): The column to bin.
            bins (int): Number of bins.

        Returns:
            pd.Series: Binned column.
        """
        if self.method == 'quantile':
            binned = pd.qcut(series, q=bins, labels=False, duplicates='drop')
        elif self.method == 'equal width':
            binned = pd.cut(series, bins=bins, labels=False, duplicates='drop')
        else:
            raise ValueError(f"Unsupported binning method '{self.method}'.")

        return binned.astype('category')

    def _check_k_anonymity(self, binned_data: pd.DataFrame) -> bool:
        """
        Check if the binned data achieves k-anonymity.

        Parameters:
            binned_data (pd.DataFrame): The binned DataFrame.

        Returns:
            bool: True if k-anonymity is achieved, False otherwise.
        """
        counts = binned_data.groupby(list(binned_data.columns)).size()
        min_group_size = counts.min()
        return min_group_size >= self.k

    def _adjust_bins(self, bin_config: Dict[str, int]) -> Dict[str, int]:
        """
        Adjust the bin configuration to attempt to achieve k-anonymity.

        Parameters:
            bin_config (Dict[str, int]): Current binning configuration.

        Returns:
            Dict[str, int]: Updated binning configuration.
        """
        # Reduce the number of bins for columns with the highest cardinality
        # Sort columns by current number of bins in descending order
        sorted_columns = sorted(bin_config.items(), key=lambda x: x[1], reverse=True)

        for col, bins in sorted_columns:
            if bins > 1:
                bin_config[col] = bins - 1
                break  # Adjust one column at a time
        else:
            # If all bins are already at 1, cannot reduce further
            raise ValueError("Cannot adjust bins further to achieve k-anonymity.")

        return bin_config

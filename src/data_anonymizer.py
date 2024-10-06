# src/data_anonymizer.py

import pandas as pd
import numpy as np
from scipy.stats import entropy
import warnings

class DataAnonymizer:
    def __init__(self, original_data: pd.DataFrame, k: int, debug_callback=None):
        """
        Initializes the DataAnonymizer with the original data, k value, and an optional debug callback.

        Parameters:
        - original_data (pd.DataFrame): The original dataset.
        - k (int): The anonymity parameter.
        - debug_callback (callable, optional): Function to call for debug messages (e.g., st.write).
        """
        self.debug = debug_callback  # Assign the debug callback

        if self.debug:
            self.debug("Initializing DataAnonymizer...")
            self.debug(f"Original Data Shape: {original_data.shape}")
            self.debug(f"k value: {k}")

        if not isinstance(original_data, pd.DataFrame):
            raise TypeError("original_data must be a pandas DataFrame.")
        if not isinstance(k, int) or k < 1:
            raise ValueError("k must be a positive integer.")
        
        self.original_data = original_data.copy()
        self.k = k
        self.anonymized_data = None
        self.report = pd.DataFrame(columns=["Method", "Parameter", "Actual_Value", "Notes"])

        if self.debug:
            self.debug("DataAnonymizer initialized successfully.")

    def _generalize_column(self, df: pd.DataFrame, column: str, bin_size: int) -> pd.DataFrame:
        """
        Generalizes a numerical column by binning.
        """
        if self.debug:
            self.debug(f"Generalizing numerical column: {column} with bin size: {bin_size}")

        min_val = df[column].min()
        max_val = df[column].max()
        bins = list(range(int(min_val), int(max_val) + bin_size, bin_size))
        if bins[-1] < max_val:
            bins.append(int(max_val) + bin_size)
        labels = [f"{bins[i]}-{bins[i+1]-1}" for i in range(len(bins)-1)]
        df[column] = pd.cut(df[column], bins=bins, labels=labels, include_lowest=True)

        if self.debug:
            self.debug(f"Column {column} after generalization: {df[column].head()}")

        return df

    def _generalize_categorical(self, df: pd.DataFrame, column: str, threshold: float = 0.05) -> pd.DataFrame:
        """
        Generalizes a categorical column by grouping infrequent categories into 'Other'.
        """
        if self.debug:
            self.debug(f"Generalizing categorical column: {column} with threshold: {threshold}")

        freq = df[column].value_counts(normalize=True)
        to_replace = freq[freq < threshold].index
        df[column] = df[column].replace(to_replace, 'Other')

        if self.debug:
            self.debug(f"Column {column} after generalization: {df[column].value_counts()}")

        return df

    def _generalize_datetime(self, df: pd.DataFrame, column: str, freq: str = 'Y') -> pd.DataFrame:
        """
        Generalizes a datetime column by truncating to a specified frequency.
        """
        if self.debug:
            self.debug(f"Generalizing datetime column: {column} with frequency: {freq}")

        df[column] = pd.to_datetime(df[column]).dt.to_period(freq).astype(str)

        if self.debug:
            self.debug(f"Column {column} after generalization: {df[column].head()}")

        return df

    def _generalize_all(self, df: pd.DataFrame, quasi_identifiers: list, bin_size: int, cat_threshold: float, datetime_freq: str) -> pd.DataFrame:
        """
        Generalizes all quasi-identifiers based on their data types.
        """
        if self.debug:
            self.debug("Starting generalization of all quasi-identifiers.")

        for col in quasi_identifiers:
            if pd.api.types.is_numeric_dtype(df[col]):
                df = self._generalize_column(df, col, bin_size)
            elif pd.api.types.is_datetime64_any_dtype(df[col]):
                df = self._generalize_datetime(df, col, datetime_freq)
            elif pd.api.types.is_categorical_dtype(df[col]) or pd.api.types.is_object_dtype(df[col]):
                df = self._generalize_categorical(df, col, cat_threshold)
            else:
                warning_msg = f"Column {col} has an unsupported data type for generalization."
                warnings.warn(warning_msg)
                if self.debug:
                    self.debug(f"Warning: {warning_msg}")
        if self.debug:
            self.debug("Completed generalization of all quasi-identifiers.")
        return df

    def _calculate_k_anonymity(self, df: pd.DataFrame, quasi_identifiers: list) -> int:
        if self.debug:
            self.debug("Calculating k-anonymity.")
        
        group_counts = df.groupby(quasi_identifiers).size()
        
        if self.debug:
            self.debug(f"Group Counts:\n{group_counts}")
        
        if group_counts.empty:
            return 0
        
        min_k = group_counts.min()
        
        if self.debug:
            self.debug(f"Calculated k-anonymity: {min_k}")
        
        return min_k


    def _calculate_l_diversity(self, df: pd.DataFrame, sensitive_attribute: str, quasi_identifiers: list) -> int:
        """
        Calculates the l-diversity of the DataFrame based on quasi-identifiers.
        """
        if self.debug:
            self.debug("Calculating l-diversity.")

        if not quasi_identifiers:
            raise ValueError("quasi_identifiers list is empty.")
        diversity = df.groupby(quasi_identifiers)[sensitive_attribute].nunique()
        if diversity.empty:
            return 0
        current_l = diversity.min()

        if self.debug:
            self.debug(f"Calculated l-diversity: {current_l}")

        return current_l

    def _calculate_t_closeness(self, df: pd.DataFrame, sensitive_attribute: str, quasi_identifiers: list) -> float:
        """
        Calculates the t-closeness of the DataFrame based on quasi-identifiers.
        """
        if self.debug:
            self.debug("Calculating t-closeness.")

        if not quasi_identifiers:
            raise ValueError("quasi_identifiers list is empty.")
        overall_dist = df[sensitive_attribute].value_counts(normalize=True).sort_index()
        groups = df.groupby(quasi_identifiers)[sensitive_attribute]
        max_kl_divergence = 0
        for name, group in groups:
            group_dist = group.value_counts(normalize=True).sort_index()
            # Align the distributions
            all_categories = overall_dist.index.union(group_dist.index)
            overall_freq = overall_dist.reindex(all_categories, fill_value=0)
            group_freq = group_dist.reindex(all_categories, fill_value=0)
            # To avoid division by zero or log of zero, add a small epsilon where necessary
            epsilon = 1e-10
            group_freq = group_freq + epsilon
            overall_freq = overall_freq + epsilon
            # Normalize again to ensure sum to 1
            group_freq = group_freq / group_freq.sum()
            overall_freq = overall_freq / overall_freq.sum()
            # Calculate Kullback-Leibler divergence as a proxy for EMD
            kl_div = entropy(overall_freq, group_freq, base=2)
            if np.isnan(kl_div):
                kl_div = 0  # Handle cases where distributions are identical
            if kl_div > max_kl_divergence:
                max_kl_divergence = kl_div

        if self.debug:
            self.debug(f"Calculated t-closeness (max KL divergence): {max_kl_divergence}")

        return max_kl_divergence

    def apply_k_anonymity(self, quasi_identifiers: list, generalize_bin_size: int = 10, cat_threshold: float = 0.05, datetime_freq: str = 'Y'):
        """
        Applies k-anonymity to the DataFrame with enhanced generalization.
        """
        if self.debug:
            self.debug("Applying k-anonymity.")

        df = self.original_data.copy()
        
        # Generalize all quasi-identifiers based on their data types
        df = self._generalize_all(df, quasi_identifiers, generalize_bin_size, cat_threshold, datetime_freq)

        # Check k-anonymity
        current_k = self._calculate_k_anonymity(df, quasi_identifiers)
        if current_k < self.k:
            warning_msg = f"After generalization, k-anonymity is {current_k}, which is less than the desired {self.k}."
            warnings.warn(warning_msg)
            if self.debug:
                self.debug(f"Warning: {warning_msg}")

        self.anonymized_data = df
        new_row = {
            "Method": "k-anonymity",
            "Parameter": f"k={self.k}",
            "Actual_Value": current_k,
            "Notes": f"Generalization applied with bin size {generalize_bin_size}"
        }
        self.report = pd.concat([self.report, pd.DataFrame([new_row])], ignore_index=True)

        if self.debug:
            self.debug("k-anonymity applied and report updated.")

    def apply_l_diversity(self, quasi_identifiers: list, sensitive_attribute: str, generalize_bin_size: int = 10, cat_threshold: float = 0.05, datetime_freq: str = 'Y'):
        """
        Applies l-diversity to the DataFrame with enhanced generalization.
        """
        if self.debug:
            self.debug("Applying l-diversity.")

        df = self.original_data.copy()
        
        # Generalize all quasi-identifiers based on their data types
        df = self._generalize_all(df, quasi_identifiers, generalize_bin_size, cat_threshold, datetime_freq)
        
        # Check l-diversity
        current_l = self._calculate_l_diversity(df, sensitive_attribute, quasi_identifiers)
        if current_l < self.k:
            warning_msg = f"After generalization, l-diversity is {current_l}, which is less than the desired {self.k}."
            warnings.warn(warning_msg)
            if self.debug:
                self.debug(f"Warning: {warning_msg}")
        
        self.anonymized_data = df
        new_row = {
            "Method": "l-diversity",
            "Parameter": f"l={self.k}",
            "Actual_Value": current_l,
            "Notes": f"Generalization applied with bin size {generalize_bin_size}"
        }
        self.report = pd.concat([self.report, pd.DataFrame([new_row])], ignore_index=True)

        if self.debug:
            self.debug("l-diversity applied and report updated.")

    def apply_t_closeness(self, quasi_identifiers: list, sensitive_attribute: str, generalize_bin_size: int = 10, cat_threshold: float = 0.05, datetime_freq: str = 'Y'):
        """
        Applies t-closeness to the DataFrame with enhanced generalization.
        """
        if self.debug:
            self.debug("Applying t-closeness.")

        df = self.original_data.copy()
        
        # Generalize all quasi-identifiers based on their data types
        df = self._generalize_all(df, quasi_identifiers, generalize_bin_size, cat_threshold, datetime_freq)
        
        # Check t-closeness
        current_t = self._calculate_t_closeness(df, sensitive_attribute, quasi_identifiers)
        # Assuming t is similar to k, you might need to parameterize it
        if current_t > self.k:
            warning_msg = f"After generalization, t-closeness is {current_t}, which exceeds the desired {self.k}."
            warnings.warn(warning_msg)
            if self.debug:
                self.debug(f"Warning: {warning_msg}")
        
        self.anonymized_data = df
        new_row = {
            "Method": "t-closeness",
            "Parameter": f"t={self.k}",
            "Actual_Value": current_t,
            "Notes": f"Generalization applied with bin size {generalize_bin_size}"
        }
        self.report = pd.concat([self.report, pd.DataFrame([new_row])], ignore_index=True)

        if self.debug:
            self.debug("t-closeness applied and report updated.")

    def get_anonymized_data(self) -> pd.DataFrame:
        """
        Returns the anonymized DataFrame.
        """
        if self.anonymized_data is None:
            raise ValueError("Anonymization method has not been applied yet.")
        if self.debug:
            self.debug("Retrieving anonymized data.")
        return self.anonymized_data

    def get_report(self) -> pd.DataFrame:
        """
        Returns the report containing anonymization metrics.
        """
        if self.report.empty:
            raise ValueError("No anonymization methods have been applied yet.")
        if self.debug:
            self.debug("Retrieving anonymization report.")
        return self.report

    def anonymize(self, method: str, quasi_identifiers: list, sensitive_attribute: str = None, generalize_bin_size: int = 10, cat_threshold: float = 0.05, datetime_freq: str = 'Y'):
        """
        Applies the specified anonymization method.
        """
        if self.debug:
            self.debug(f"Starting anonymization with method: {method}")

        method = method.lower()
        if method == 'k-anonymity':
            self.apply_k_anonymity(quasi_identifiers, generalize_bin_size, cat_threshold, datetime_freq)
        elif method in ['l-diversity', 'ℓ-diversity']:
            if not sensitive_attribute:
                raise ValueError("sensitive_attribute must be provided for l-diversity.")
            self.apply_l_diversity(quasi_identifiers, sensitive_attribute, generalize_bin_size, cat_threshold, datetime_freq)
        elif method == 't-closeness':
            if not sensitive_attribute:
                raise ValueError("sensitive_attribute must be provided for t-closeness.")
            self.apply_t_closeness(quasi_identifiers, sensitive_attribute, generalize_bin_size, cat_threshold, datetime_freq)
        else:
            error_msg = "Unsupported anonymization method. Choose from 'k-anonymity', 'l-diversity', 't-closeness'."
            if self.debug:
                self.debug(f"Error: {error_msg}")
            raise ValueError(error_msg)

        if self.debug:
            self.debug(f"Anonymization with method {method} completed.")

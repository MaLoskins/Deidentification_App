# src/data_anonymizer.py

import pandas as pd
import numpy as np
from scipy.stats import entropy
import warnings

# Import the DataBinner class
from src.binning.data_binner import DataBinner

class DataAnonymizer:
    def __init__(self, original_data: pd.DataFrame, debug_callback=None):
        # Initialization code remains the same
        self.debug = debug_callback
        if self.debug:
            self.debug("Initializing DataAnonymizer...")
            self.debug(f"Original Data Shape: {original_data.shape}")

        if not isinstance(original_data, pd.DataFrame):
            raise TypeError("original_data must be a pandas DataFrame.")

        self.original_data = original_data.copy()
        self.anonymized_data = None
        self.report = pd.DataFrame(columns=["Method", "Parameter", "Actual_Value", "Notes"])

        if self.debug:
            self.debug("DataAnonymizer initialized successfully.")

    def _generalize_all(self, quasi_identifiers: list, max_iterations: int = 10) -> pd.DataFrame:
        if self.debug:
            self.debug("Starting iterative generalization of all quasi-identifiers.")

        # Initialize generalization parameters
        generalization_params = {}
        datetime_cols = []

        for col in quasi_identifiers:
            unique_values = self.original_data[col].nunique()
            if pd.api.types.is_numeric_dtype(self.original_data[col]):
                generalization_params[col] = {'bins': min(30, unique_values), 'type': 'numeric'}
            elif pd.api.types.is_datetime64_any_dtype(self.original_data[col]):
                generalization_params[col] = {'bins': min(30, unique_values), 'type': 'datetime'}
                datetime_cols.append(col)
            elif pd.api.types.is_categorical_dtype(self.original_data[col]) or pd.api.types.is_object_dtype(self.original_data[col]):
                generalization_params[col] = {'bins': min(30, unique_values), 'type': 'categorical'}
            else:
                generalization_params[col] = {'type': 'unsupported'}

        base_df = self.original_data.copy()

        # Convert datetime columns to numeric timestamps for binning
        for col in datetime_cols:
            base_df[col] = pd.to_datetime(base_df[col]).astype('int64')  # Convert to nanoseconds

        for iteration in range(max_iterations):
            if self.debug:
                self.debug(f"Generalization Iteration {iteration + 1}")

            # Make a copy of base_df
            df = base_df.copy()

            # Prepare bin_dict for DataBinner
            bin_dict = {}

            for col in quasi_identifiers:
                params = generalization_params[col]
                if 'max_generalization_reached' in params and params['max_generalization_reached']:
                    df[col] = 'All'  # Set to 'All' if maximum generalization reached
                    continue  # Skip further generalization

                if params['type'] in ['numeric', 'categorical', 'datetime']:
                    bin_count = params['bins']
                    bin_dict[col] = bin_count
                else:
                    warning_msg = f"Column '{col}' has an unsupported data type for generalization."
                    warnings.warn(warning_msg)
                    if self.debug:
                        self.debug(f"Warning: {warning_msg}")

            # Use DataBinner for numeric and categorical columns
            if bin_dict:
                binner = DataBinner(df, method='equal width')  # You can choose 'equal width' or 'quantile'
                binned_df, binned_columns = binner.bin_columns(bin_dict)
                # Update df with binned columns
                for binned_col in binned_df.columns:
                    df[binned_col] = binned_df[binned_col]

            # Convert binned datetime columns back to readable date ranges
            for col in datetime_cols:
                if col in df.columns and df[col].dtype.name == 'category':
                    # Parse bin labels to readable date ranges
                    bin_labels = df[col].cat.categories
                    new_labels = []
                    for label in bin_labels:
                        # Extract lower and upper bounds
                        bounds = label.strip('[]()').split('->')
                        if len(bounds) == 2:
                            lower, upper = bounds
                            try:
                                lower_date = pd.to_datetime(float(lower.strip()), unit='ns')
                                upper_date = pd.to_datetime(float(upper.strip()), unit='ns')
                                lower_str = lower_date.strftime('%Y-%m-%d') if pd.notnull(lower_date) else ''
                                upper_str = upper_date.strftime('%Y-%m-%d') if pd.notnull(upper_date) else ''
                                new_label = f"[{lower_str} -> {upper_str})"
                            except Exception:
                                new_label = label
                        else:
                            new_label = label
                        new_labels.append(new_label)
                    # Update the categories with new labels
                    df[col] = df[col].cat.rename_categories(new_labels)

            yield df  # Yield the generalized DataFrame

            # Update generalization parameters for next iteration
            for col in quasi_identifiers:
                params = generalization_params[col]
                if params['type'] in ['numeric', 'categorical', 'datetime']:
                    if params['bins'] <= 1:
                        params['max_generalization_reached'] = True
                        df[col] = 'All'
                        if self.debug:
                            self.debug(f"Maximum generalization reached for column '{col}'. All values grouped into 'All'.")
                    else:
                        # Decrease bins more gradually
                        decrement = max(1, int(params['bins'] * 0.1))  # Decrease bins by 10%
                        params['bins'] = max(1, params['bins'] - decrement)
                        if self.debug:
                            self.debug(f"Decreasing bin count for column '{col}' to {params['bins']}")
                else:
                    continue  # Unsupported types are skipped

        else:
            # Maximum iterations reached without achieving desired anonymity
            warning_msg = f"Maximum generalization iterations ({max_iterations}) reached without achieving desired anonymity."
            warnings.warn(warning_msg)
            if self.debug:
                self.debug(f"Warning: {warning_msg}")

        if self.debug:
            self.debug("Completed iterative generalization of all quasi-identifiers.")
        return df

    def _calculate_k_anonymity(self, df: pd.DataFrame, quasi_identifiers: list) -> int:
        """
        Calculates the k-anonymity of the DataFrame based on quasi-identifiers.

        Parameters:
        - df (pd.DataFrame): The DataFrame to evaluate.
        - quasi_identifiers (list): List of quasi-identifier column names.

        Returns:
        - int: The calculated k-anonymity value.
        """
        if self.debug:
            self.debug("Calculating k-anonymity.")
            self.debug(f"Quasi-identifiers used for grouping: {quasi_identifiers}")

        if not quasi_identifiers:
            raise ValueError("quasi_identifiers list is empty.")

        # Ensure all quasi-identifiers are present in the DataFrame
        missing_cols = [col for col in quasi_identifiers if col not in df.columns]
        if missing_cols:
            error_msg = f"Quasi-identifiers missing in DataFrame: {missing_cols}"
            if self.debug:
                self.debug(f"Error: {error_msg}")
            raise ValueError(error_msg)

        group_counts = df.groupby(quasi_identifiers).size()

        if self.debug:
            self.debug(f"Group Counts:\n{group_counts}")

        if group_counts.empty:
            self.debug("No groups found. Returning k-anonymity as 0.")
            return 0

        # Exclude groups with zero counts (if any, though ideally none after conversion)
        group_counts = group_counts[group_counts > 0]

        min_k = group_counts.min()

        if self.debug:
            self.debug(f"Calculated k-anonymity: {min_k}")

        return min_k

    def _calculate_l_diversity(self, df: pd.DataFrame, sensitive_attribute: str, quasi_identifiers: list) -> int:
        """
        Calculates the l-diversity of the DataFrame based on quasi-identifiers.

        Parameters:
        - df (pd.DataFrame): The DataFrame to evaluate.
        - sensitive_attribute (str): The sensitive attribute column name.
        - quasi_identifiers (list): List of quasi-identifier column names.

        Returns:
        - int: The calculated l-diversity value.
        """
        if self.debug:
            self.debug("Calculating l-diversity.")

        if not quasi_identifiers:
            raise ValueError("quasi_identifiers list is empty.")
        if sensitive_attribute not in df.columns:
            error_msg = f"Sensitive attribute '{sensitive_attribute}' not found in DataFrame."
            if self.debug:
                self.debug(f"Error: {error_msg}")
            raise ValueError(error_msg)

        diversity = df.groupby(quasi_identifiers)[sensitive_attribute].nunique()

        if self.debug:
            self.debug(f"Diversity counts per group:\n{diversity}")

        if diversity.empty:
            return 0

        current_l = diversity.min()

        if self.debug:
            self.debug(f"Calculated l-diversity: {current_l}")

        return current_l

    def _calculate_t_closeness(self, df: pd.DataFrame, sensitive_attribute: str, quasi_identifiers: list) -> float:
        """
        Calculates the t-closeness of the DataFrame based on quasi-identifiers.

        Parameters:
        - df (pd.DataFrame): The DataFrame to evaluate.
        - sensitive_attribute (str): The sensitive attribute column name.
        - quasi_identifiers (list): List of quasi-identifier column names.

        Returns:
        - float: The calculated t-closeness value.
        """
        if self.debug:
            self.debug("Calculating t-closeness.")

        if not quasi_identifiers:
            raise ValueError("quasi_identifiers list is empty.")
        if sensitive_attribute not in df.columns:
            raise ValueError(f"Sensitive attribute '{sensitive_attribute}' not found in DataFrame.")

        overall_dist = df[sensitive_attribute].value_counts(normalize=True).sort_index()
        groups = df.groupby(quasi_identifiers)[sensitive_attribute]
        max_distance = 0

        for name, group in groups:
            group_dist = group.value_counts(normalize=True).sort_index()
            # Align the distributions
            all_categories = overall_dist.index.union(group_dist.index)
            overall_freq = overall_dist.reindex(all_categories, fill_value=0)
            group_freq = group_dist.reindex(all_categories, fill_value=0)

            # Calculate total variation distance
            distance = 0.5 * np.sum(np.abs(overall_freq.values - group_freq.values))
            max_distance = max(max_distance, distance)

            if self.debug:
                self.debug(f"Group {name}: Total Variation Distance = {round(distance, 3)}")

        if self.debug:
            self.debug(f"Calculated t-closeness (max total variation distance): {max_distance}")

        return max_distance

    def _update_report(self, method: str, parameter: str, actual_value, notes: str):
        new_row = {
            "Method": method,
            "Parameter": parameter,
            "Actual_Value": actual_value,
            "Notes": notes
        }
        self.report = pd.concat([self.report, pd.DataFrame([new_row])], ignore_index=True)

    def apply_k_anonymity(self, quasi_identifiers: list, k: int, max_iterations: int = 10):
        """
        Applies k-anonymity to the DataFrame with automated generalization.

        Parameters:
        - quasi_identifiers (list): List of quasi-identifier column names.
        - k (int): The k-anonymity parameter.
        - max_iterations (int): Maximum number of generalization iterations.
        """
        if self.debug:
            self.debug("Applying k-anonymity.")

        if not quasi_identifiers:
            raise ValueError("quasi_identifiers list is empty.")
        if k is None or k < 1:
            raise ValueError("k value must be a positive integer for k-anonymity.")

        # Perform iterative generalization
        generalized_df_generator = self._generalize_all(quasi_identifiers, max_iterations=max_iterations)
        for generalized_df in generalized_df_generator:
            current_k = self._calculate_k_anonymity(generalized_df, quasi_identifiers)
            if current_k >= k:
                break  # Desired k-anonymity achieved

        # Final k-anonymity check
        if current_k < k:
            warning_msg = f"After generalization, k-anonymity is {current_k}, which is less than the desired {k}."
            warnings.warn(warning_msg)
            if self.debug:
                self.debug(f"Warning: {warning_msg}")
        else:
            if self.debug:
                self.debug(f"k-anonymity requirement met: {current_k} >= {k}")

        self.anonymized_data = generalized_df
        self._update_report("k-anonymity", f"k={k}", current_k, "Automated generalization applied")

        if self.debug:
            self.debug("k-anonymity applied and report updated.")

    def apply_l_diversity(self, quasi_identifiers: list, sensitive_attribute: str, l: int, max_iterations: int = 10):
        """
        Applies l-diversity to the DataFrame with automated generalization.

        Parameters:
        - quasi_identifiers (list): List of quasi-identifier column names.
        - sensitive_attribute (str): The sensitive attribute column name.
        - l (int): The l-diversity parameter.
        - max_iterations (int): Maximum number of generalization iterations.
        """
        if self.debug:
            self.debug("Applying l-diversity.")

        if not quasi_identifiers:
            raise ValueError("quasi_identifiers list is empty.")
        if not sensitive_attribute:
            raise ValueError("sensitive_attribute must be provided for l-diversity.")
        if l is None or l < 1:
            raise ValueError("l value must be a positive integer for l-diversity.")

        # Perform iterative generalization
        generalized_df_generator = self._generalize_all(quasi_identifiers, max_iterations=max_iterations)
        for generalized_df in generalized_df_generator:
            current_l = self._calculate_l_diversity(generalized_df, sensitive_attribute, quasi_identifiers)
            if current_l >= l:
                break  # Desired l-diversity achieved

        # Final l-diversity check
        if current_l < l:
            warning_msg = f"After generalization, l-diversity is {current_l}, which is less than the desired {l}."
            warnings.warn(warning_msg)
            if self.debug:
                self.debug(f"Warning: {warning_msg}")
        else:
            if self.debug:
                self.debug(f"l-diversity requirement met: {current_l} >= {l}")

        self.anonymized_data = generalized_df
        self._update_report("l-diversity", f"l={l}", current_l, "Automated generalization applied")

        if self.debug:
            self.debug("l-diversity applied and report updated.")

    def apply_t_closeness(self, quasi_identifiers: list, sensitive_attribute: str, t: float, max_iterations: int = 10):
        """
        Applies t-closeness to the DataFrame with automated generalization.

        Parameters:
        - quasi_identifiers (list): List of quasi-identifier column names.
        - sensitive_attribute (str): The sensitive attribute column name.
        - t (float): The t-closeness parameter.
        - max_iterations (int): Maximum number of generalization iterations.
        """
        if self.debug:
            self.debug("Applying t-closeness.")

        if not quasi_identifiers:
            raise ValueError("quasi_identifiers list is empty.")
        if not sensitive_attribute:
            raise ValueError("sensitive_attribute must be provided for t-closeness.")
        if t is None or t < 0:
            raise ValueError("t value must be a non-negative number for t-closeness.")

        generalized_df_generator = self._generalize_all(quasi_identifiers, max_iterations=max_iterations)
        for generalized_df in generalized_df_generator:
            current_t = self._calculate_t_closeness(generalized_df, sensitive_attribute, quasi_identifiers)

            if self.debug:
                self.debug(f"Current t-closeness: {current_t} for t={t}")

            if current_t <= t:
                break  # Desired t-closeness achieved

        # Final t-closeness check
        if current_t > t:
            warning_msg = f"After generalization, t-closeness is {current_t}, which exceeds the desired {t}."
            warnings.warn(warning_msg)
            if self.debug:
                self.debug(f"Warning: {warning_msg}")
        else:
            if self.debug:
                self.debug(f"t-closeness requirement met: {current_t} <= {t}")

        self.anonymized_data = generalized_df
        self._update_report("t-closeness", f"t={t}", current_t, "Automated generalization applied")

        if self.debug:
            self.debug("t-closeness applied and report updated.")

    def get_anonymized_data(self) -> pd.DataFrame:
        """
        Returns the anonymized DataFrame.

        Returns:
        - pd.DataFrame: The anonymized dataset.

        Raises:
        - ValueError: If no anonymization method has been applied yet.
        """
        if self.anonymized_data is None:
            raise ValueError("Anonymization method has not been applied yet.")
        if self.debug:
            self.debug("Retrieving anonymized data.")
        return self.anonymized_data

    def get_report(self) -> pd.DataFrame:
        """
        Returns the report containing anonymization metrics.

        Returns:
        - pd.DataFrame: The anonymization report.

        Raises:
        - ValueError: If no anonymization methods have been applied yet.
        """
        if self.report.empty:
            raise ValueError("No anonymization methods have been applied yet.")
        if self.debug:
            self.debug("Retrieving anonymization report.")
        return self.report

    def anonymize(self, method: str, quasi_identifiers: list, k: int = None, l: int = None, t: float = None, sensitive_attribute: str = None, max_iterations: int = 10):
        """
        Applies the specified anonymization method with automated generalization.

        Parameters:
        - method (str): The anonymization method to apply ('k-anonymity', 'l-diversity', 't-closeness').
        - quasi_identifiers (list): List of quasi-identifier column names.
        - k (int, optional): The k-anonymity parameter.
        - l (int, optional): The l-diversity parameter.
        - t (float, optional): The t-closeness parameter.
        - sensitive_attribute (str, optional): The sensitive attribute column name. Required for 'l-diversity' and 't-closeness'.
        - max_iterations (int, optional): Maximum number of generalization iterations. Defaults to 10.

        Raises:
        - ValueError: If an unsupported anonymization method is specified or required parameters are missing.
        """
        if self.debug:
            self.debug(f"Starting anonymization with method: {method}")

        method = method.lower()
        if method == 'k-anonymity':
            if k is None or k < 1:
                raise ValueError("k value must be a positive integer for k-anonymity.")
            self.apply_k_anonymity(quasi_identifiers, k, max_iterations=max_iterations)
        elif method in ['l-diversity', 'ℓ-diversity']:
            if not sensitive_attribute:
                raise ValueError("sensitive_attribute must be provided for l-diversity.")
            if l is None or l < 1:
                raise ValueError("l value must be a positive integer for l-diversity.")
            self.apply_l_diversity(quasi_identifiers, sensitive_attribute, l, max_iterations=max_iterations)
        elif method == 't-closeness':
            if not sensitive_attribute:
                raise ValueError("sensitive_attribute must be provided for t-closeness.")
            if t is None or t < 0:
                raise ValueError("t parameter must be a non-negative number for t-closeness.")
            self.apply_t_closeness(quasi_identifiers, sensitive_attribute, t, max_iterations=max_iterations)
        else:
            error_msg = "Unsupported anonymization method. Choose from 'k-anonymity', 'l-diversity', 't-closeness'."
            if self.debug:
                self.debug(f"Error: {error_msg}")
            raise ValueError(error_msg)

        if self.debug:
            self.debug(f"Anonymization with method {method} completed.")

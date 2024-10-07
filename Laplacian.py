import pandas as pd
import numpy as np
import hashlib
import random
from scipy.stats import pearsonr

class DifferentialPrivacyLaplacian:
    def __init__(self, epsilon, sensitivity=1, seed=None, categorical_threshold=20, text_columns=None):
        """
        Initialize the Differential Privacy Laplacian Noise Generator.

        Parameters:
        - epsilon (float): Privacy parameter. Smaller values offer stronger privacy.
        - sensitivity (float or dict): Sensitivity per column. If dict, keys are column names.
        - seed (int, optional): Seed for reproducibility.
        - categorical_threshold (int): Maximum number of unique values to treat object dtype as categorical.
        - text_columns (list, optional): List of column names to treat as text.
        """
        self.epsilon = epsilon
        self.sensitivity = sensitivity
        self.categorical_threshold = categorical_threshold
        self.text_columns = text_columns if text_columns else []
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)

    def _get_sensitivity(self, column):
        """
        Retrieve sensitivity for a given column.
        """
        if isinstance(self.sensitivity, dict):
            return self.sensitivity.get(column, 1)
        return self.sensitivity

    def _is_categorical(self, series, column):
        """
        Determine if a series should be treated as categorical based on unique values and text_columns.
        """
        if column in self.text_columns:
            return False
        return pd.api.types.is_categorical_dtype(series) or (pd.api.types.is_object_dtype(series) and series.nunique() <= self.categorical_threshold)

    def _add_laplacian_noise(self, series, column):
        """
        Add Laplacian noise to a numerical series.
        """
        sensitivity = self._get_sensitivity(column)
        if sensitivity <= 0:
            raise ValueError(f"Sensitivity for column '{column}' must be greater than 0.")
        scale = sensitivity / self.epsilon
        noise = np.random.laplace(0, scale, size=series.shape)
        noisy_series = series + noise
        return noisy_series

    def _add_randomized_response(self, series, column):
        """
        Apply randomized response to a categorical series by randomly selecting a different category.
        """
        sensitivity = self._get_sensitivity(column)
        if sensitivity <= 0:
            raise ValueError(f"Sensitivity for column '{column}' must be greater than 0 for randomized response.")
        
        categories = series.unique()
        num_categories = len(categories)
        
        if num_categories <= 1:
            # Only one category exists; cannot randomize
            return series

        # Probability to keep the original value
        p = np.exp(self.epsilon / (2 * sensitivity)) / (1 + np.exp(self.epsilon / (2 * sensitivity)))

        def randomized_response(x):
            if random.random() < p:
                return x
            else:
                # Select a different category uniformly at random
                other_categories = [cat for cat in categories if cat != x]
                return random.choice(other_categories) if other_categories else x

        noisy_series = series.apply(randomized_response)
        return noisy_series

    def _flip_boolean(self, series, column):
        """
        Flip boolean values with probability derived from ε and sensitivity.
        """
        sensitivity = self._get_sensitivity(column)
        if sensitivity <= 0:
            raise ValueError(f"Sensitivity for column '{column}' must be greater than 0 for flipping.")
        
        # Calculate probability to flip
        p_flip = 1 / (1 + np.exp(self.epsilon / sensitivity))
        
        def flip(x):
            return not x if random.random() < p_flip else x

        noisy_series = series.apply(flip)
        return noisy_series

    def _redact_text(self, series, column):
        """
        Redact sensitive information from text data.
        This is a placeholder implementation. In practice, use NLP techniques.
        """
        # Simple redaction: remove digits and punctuation
        noisy_series = series.apply(
            lambda x: ''.join([char for char in x if char.isalpha() or char.isspace()]).strip()
            if isinstance(x, str) else x
        )
        return noisy_series

    def apply_noise(self, df, selected_columns):
        """
        Apply differential privacy noise to selected columns of the DataFrame.

        Parameters:
        - df (pd.DataFrame): The input DataFrame.
        - selected_columns (list): List of column names to apply noise.

        Returns:
        - pd.DataFrame: New DataFrame with noisy columns.
        """
        noisy_df = df.copy()

        for column in selected_columns:
            if column not in df.columns:
                raise ValueError(f"Column '{column}' does not exist in the DataFrame.")

            series = df[column]
            dtype = series.dtype

            if pd.api.types.is_numeric_dtype(dtype) and not pd.api.types.is_bool_dtype(dtype):
                # Numerical Data
                noisy_df[column] = self._add_laplacian_noise(series.astype(float), column)

            elif self._is_categorical(series, column):
                # Categorical Data
                noisy_df[column] = self._add_randomized_response(series.astype(str), column)

            elif pd.api.types.is_bool_dtype(dtype):
                # Boolean Data
                noisy_df[column] = self._flip_boolean(series, column)

            elif pd.api.types.is_object_dtype(dtype) or pd.api.types.is_string_dtype(dtype):
                # Text Data
                noisy_df[column] = self._redact_text(series, column)

            else:
                # For unsupported types, raise an error or handle accordingly
                raise TypeError(f"Data type of column '{column}' is not supported.")

        return noisy_df

    def diagnostics(self, original_df, noisy_df, selected_columns):
        """
        Perform diagnostic assessments to evaluate the effectiveness of de-identification.

        Parameters:
        - original_df (pd.DataFrame): The original DataFrame before noise.
        - noisy_df (pd.DataFrame): The DataFrame after noise has been applied.
        - selected_columns (list): List of column names that were noised.

        Returns:
        - dict: A dictionary containing diagnostic results.
        """
        diagnostics_results = {}

        for column in selected_columns:
            if column not in original_df.columns or column not in noisy_df.columns:
                diagnostics_results[column] = "Column missing in one of the DataFrames."
                continue

            original = original_df[column]
            noisy = noisy_df[column]
            dtype = original.dtype

            diagnostics_results[column] = {}

            if pd.api.types.is_numeric_dtype(dtype) and not pd.api.types.is_bool_dtype(dtype):
                # Numerical Comparison
                mean_original = original.mean()
                mean_noisy = noisy.mean()
                var_original = original.var()
                var_noisy = noisy.var()

                # Correlation
                if original.nunique() > 1:
                    correlation, _ = pearsonr(original, noisy)
                else:
                    correlation = np.nan  # Correlation undefined for constant series

                diagnostics_results[column]['mean_original'] = mean_original
                diagnostics_results[column]['mean_noisy'] = mean_noisy
                diagnostics_results[column]['variance_original'] = var_original
                diagnostics_results[column]['variance_noisy'] = var_noisy
                diagnostics_results[column]['correlation'] = correlation

            elif self._is_categorical(original, column):
                # Categorical Data
                freq_original = original.value_counts(normalize=True).to_dict()
                freq_noisy = noisy.value_counts(normalize=True).to_dict()
                diagnostics_results[column]['frequency_original'] = freq_original
                diagnostics_results[column]['frequency_noisy'] = freq_noisy

            elif pd.api.types.is_bool_dtype(dtype):
                # Boolean Data
                freq_original = original.value_counts(normalize=True).to_dict()
                freq_noisy = noisy.value_counts(normalize=True).to_dict()
                diagnostics_results[column]['frequency_original'] = freq_original
                diagnostics_results[column]['frequency_noisy'] = freq_noisy

            elif column in self.text_columns:
                # Text Data
                original_length = original.apply(lambda x: len(x) if isinstance(x, str) else 0)
                noisy_length = noisy.apply(lambda x: len(x) if isinstance(x, str) else 0)
                diagnostics_results[column]['mean_length_original'] = original_length.mean()
                diagnostics_results[column]['mean_length_noisy'] = noisy_length.mean()

            else:
                diagnostics_results[column]['status'] = "No diagnostics implemented for this data type."

        # Unique Combinations
        diagnostics_results['unique_combinations'] = {}
        original_unique = original_df[selected_columns].drop_duplicates().shape[0]
        noisy_unique = noisy_df[selected_columns].drop_duplicates().shape[0]
        diagnostics_results['unique_combinations']['unique_original'] = original_unique
        diagnostics_results['unique_combinations']['unique_noisy'] = noisy_unique

        # Outlier Analysis for Numerical Columns
        # Exclude boolean and categorical columns
        numerical_columns = [
            col for col in selected_columns 
            if pd.api.types.is_numeric_dtype(original_df[col].dtype) and 
               not pd.api.types.is_bool_dtype(original_df[col].dtype) and 
               not self._is_categorical(original_df[col], col)
        ]
        diagnostics_results['outlier_analysis'] = {}
        for column in numerical_columns:
            original = original_df[column]
            noisy = noisy_df[column]
            q1 = original.quantile(0.25)
            q3 = original.quantile(0.75)
            IQR = q3 - q1
            lower_bound = q1 - 1.5 * IQR
            upper_bound = q3 + 1.5 * IQR

            original_outliers = original[(original < lower_bound) | (original > upper_bound)].count()
            noisy_outliers = noisy[(noisy < lower_bound) | (noisy > upper_bound)].count()

            diagnostics_results['outlier_analysis'][column] = {
                'original_outliers': original_outliers,
                'noisy_outliers': noisy_outliers
            }

        return diagnostics_results




import pandas as pd

# Sample DataFrame
data = {
    'age': [25, 30, 45, 22, 35],
    'salary': [50000, 60000, 80000, 45000, 70000],
    'gender': ['Male', 'Female', 'Female', 'Male', 'Male'],
    'has_disease': [True, False, True, False, True],
    'comments': [
        "Patient shows improvement.",
        "No significant changes.",
        "Requires further tests.",
        "Stable condition.",
        "Discharged after recovery."
    ]
}

df = pd.DataFrame(data)

# Convert 'gender' to 'category' dtype
df['gender'] = df['gender'].astype('category')

# Define privacy parameters
epsilon = 1.0
sensitivity = {
    'age': 1,
    'salary': 1000,
    'gender': 1,
    'has_disease': 1,
    # 'comments' sensitivity is irrelevant as it's treated as text
}

# Specify text columns
text_columns = ['comments']

# Initialize the differential privacy object
dp = DifferentialPrivacyLaplacian(
    epsilon=epsilon, 
    sensitivity=sensitivity, 
    seed=42, 
    text_columns=text_columns
)

# Select columns to apply noise
selected_columns = ['age', 'salary', 'gender', 'has_disease', 'comments']

# Apply noise
noisy_df = dp.apply_noise(df, selected_columns)

print("Original DataFrame:")
print(df)
print("\nNoisy DataFrame:")
print(noisy_df)

# Perform diagnostics
diagnostics = dp.diagnostics(df, noisy_df, selected_columns)

print("\nDiagnostics Results:")
for key, value in diagnostics.items():
    print(f"\n{key}:")
    if isinstance(value, dict):
        for sub_key, sub_val in value.items():
            print(f"  {sub_key}: {sub_val}")
    else:
        print(f"  {value}")

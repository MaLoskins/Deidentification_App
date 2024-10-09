# synthetic_data_generator.py

import pandas as pd
import joblib
import os
import logging
from sdv.metadata import SingleTableMetadata
from sdv.single_table import CTGANSynthesizer, GaussianCopulaSynthesizer
import torch
print(torch.cuda.is_available())

class SyntheticDataGenerator:
    def __init__(
        self,
        dataframe,
        selected_columns,
        method='ctgan',
        model_params=None,
        missing_value_strategy='drop',
        missing_fill_value=None,
        categorical_columns=None,
        numerical_columns=None,
        datetime_columns=None  # Added datetime_columns parameter
    ):
        """
        Initialize the SyntheticDataGenerator.

        Parameters:
        - dataframe: pd.DataFrame
            Original dataframe to generate synthetic data from.
        - selected_columns: list of str
            List of columns to include in the synthetic data generation.
        - method: str
            Synthesizer method to use ('ctgan' or 'gaussian_copula').
        - model_params: dict
            Additional parameters for the synthesizer model.
        - missing_value_strategy: str
            Strategy for handling missing values ('drop', 'mean_impute', 'median_impute', 'mode_impute', 'fill').
        - missing_fill_value: any
            Value to fill missing values with if missing_value_strategy is 'fill'.
        - categorical_columns: list of str or None
            List of columns to be treated as categorical. If None, auto-detect.
        - numerical_columns: list of str or None
            List of columns to be treated as numerical. If None, auto-detect.
        - datetime_columns: list of str or None
            List of columns to be treated as datetime. If None, auto-detect.
        """
        self.dataframe = dataframe.copy()
        self.selected_columns = [col for col in selected_columns if col in self.dataframe.columns]
        self.method = method.lower()
        self.model_params = model_params if model_params is not None else {}
        self.model = None
        self.missing_value_strategy = missing_value_strategy
        self.missing_fill_value = missing_fill_value

        self.logger = logging.getLogger(__name__)

        if self.method not in ['ctgan', 'gaussian_copula']:
            raise ValueError("Unsupported method. Choose 'ctgan' or 'gaussian_copula'.")

        if not self.selected_columns:
            raise ValueError("No valid columns selected for synthetic data generation.")

        self.dataframe = self.dataframe[self.selected_columns]

        # Data type handling
        if categorical_columns is not None and numerical_columns is not None and datetime_columns is not None:
            self.categorical_columns = categorical_columns
            self.numerical_columns = numerical_columns
            self.datetime_columns = datetime_columns
        else:
            # Automatic data type detection
            self.datetime_columns = self.dataframe.select_dtypes(include=['datetime', 'datetime64']).columns.tolist()
            self.categorical_columns = self.dataframe.select_dtypes(include=['object', 'category']).columns.tolist()
            self.numerical_columns = self.dataframe.select_dtypes(include=['number']).columns.tolist()

            # Remove datetime columns from categorical and numerical if detected
            self.categorical_columns = [col for col in self.categorical_columns if col not in self.datetime_columns]
            self.numerical_columns = [col for col in self.numerical_columns if col not in self.datetime_columns]

        # Set data types accordingly
        for col in self.categorical_columns:
            if col in self.dataframe.columns:
                self.dataframe[col] = self.dataframe[col].astype('object')
            else:
                self.logger.warning(f"Categorical column '{col}' not found in dataframe.")

        for col in self.numerical_columns:
            if col in self.dataframe.columns:
                self.dataframe[col] = pd.to_numeric(self.dataframe[col], errors='coerce')
            else:
                self.logger.warning(f"Numerical column '{col}' not found in dataframe.")

        for col in self.datetime_columns:
            if col in self.dataframe.columns:
                if not pd.api.types.is_datetime64_any_dtype(self.dataframe[col]):
                    # Attempt to convert to datetime
                    self.dataframe[col] = pd.to_datetime(self.dataframe[col], errors='coerce')
            else:
                self.logger.warning(f"Datetime column '{col}' not found in dataframe.")

        # Handle missing values
        self.handle_missing_values()

        if self.dataframe.empty:
            raise ValueError("Dataframe is empty after missing value handling. Cannot proceed with model training.")

        # Create metadata
        self.metadata = SingleTableMetadata()
        self.metadata.detect_from_dataframe(self.dataframe)

        # Explicitly set column types in metadata for better accuracy
        self._update_metadata_column_types()

    def _update_metadata_column_types(self):
        """Explicitly set column types in metadata, especially for datetime columns."""
        for col in self.datetime_columns:
            if col in self.metadata.get_column_names():
                self.metadata.update_column(col, sdtype='datetime')

        # Optional: Specify additional constraints or types if needed
        # For example, setting a column as primary key, unique, etc.
        # self.metadata.update_column('id', sdtype='id', constraints={'unique': True})

    def handle_missing_values(self):
        """Handle missing values in the dataframe based on the specified strategy."""
        if self.missing_value_strategy == 'drop':
            self.dataframe.dropna(inplace=True)

        # Fill missing values with mean
        elif self.missing_value_strategy == 'mean_impute':
            for col in self.numerical_columns:
                if col in self.dataframe.columns:
                    mean_value = self.dataframe[col].mean()
                    self.dataframe[col].fillna(mean_value, inplace=True)
            for col in self.categorical_columns:
                if col in self.dataframe.columns:
                    mode_value = self.dataframe[col].mode().iloc[0]
                    self.dataframe[col].fillna(mode_value, inplace=True)

        # Fill missing values with median
        elif self.missing_value_strategy == 'median_impute':
            for col in self.numerical_columns:
                if col in self.dataframe.columns:
                    median_value = self.dataframe[col].median()
                    self.dataframe[col].fillna(median_value, inplace=True)
            for col in self.categorical_columns:
                if col in self.dataframe.columns:
                    mode_value = self.dataframe[col].mode().iloc[0]
                    self.dataframe[col].fillna(mode_value, inplace=True)

        # Fill missing values with mode
        elif self.missing_value_strategy == 'mode_impute':
            for col in self.dataframe.columns:
                mode_value = self.dataframe[col].mode().iloc[0]
                self.dataframe[col].fillna(mode_value, inplace=True)

        # Fill missing values with a specified value        
        elif self.missing_value_strategy == 'fill':
            self.dataframe.fillna(self.missing_fill_value, inplace=True)
        else:
            raise ValueError(f"Unsupported missing value strategy '{self.missing_value_strategy}'.")

    def train(self):
        """Train the synthesizer model."""
        try:
            if self.method == 'ctgan':
                # Initialize CTGANSynthesizer
                self.model = CTGANSynthesizer(metadata=self.metadata, **self.model_params, cuda=True)
                self.logger.info("Training CTGAN model...")
                self.model.fit(self.dataframe)
                self.logger.info("CTGAN model trained successfully.")
            elif self.method == 'gaussian_copula':
                # Initialize GaussianCopulaSynthesizer
                self.model = GaussianCopulaSynthesizer(metadata=self.metadata, **self.model_params)
                self.logger.info("Training Gaussian Copula model...")
                self.model.fit(self.dataframe)
                self.logger.info("Gaussian Copula model trained successfully.")
        except Exception as e:
            self.logger.error(f"Error during model training: {e}")
            raise

    def generate(self, num_samples):
        """Generate synthetic data samples."""
        if self.model is None:
            raise ValueError("Model is not trained yet. Call the 'train' method first.")
        try:
            self.logger.info(f"Generating {num_samples} synthetic samples using {self.method}...")
            synthetic_data = self.model.sample(num_samples)
            self.logger.info("Synthetic data generation completed.")
            return synthetic_data
        except Exception as e:
            self.logger.error(f"Error during data generation: {e}")
            raise

    def save_model(self, filepath):
        """Save the trained model to a file."""
        if self.model is None:
            raise ValueError("No trained model to save.")
        try:
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            joblib.dump(self.model, filepath)
            self.logger.info(f"Model saved to {filepath}.")
        except Exception as e:
            self.logger.error(f"Error saving model: {e}")
            raise

    def load_model(self, filepath):
        """Load a trained model from a file."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"No model found at {filepath}.")
        try:
            self.model = joblib.load(filepath)
            self.logger.info(f"Model loaded from {filepath}.")
        except Exception as e:
            self.logger.error(f"Error loading model: {e}")
            raise

    def get_model(self):
        """Get the trained model."""
        return self.model

    def get_dataframe(self):
        """Get the processed dataframe."""
        return self.dataframe

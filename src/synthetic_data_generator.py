# src/synthetic_data_generator.py

from sdv.metadata import SingleTableMetadata
from sdv.single_table import CTGANSynthesizer, GaussianCopulaSynthesizer
import pandas as pd
import joblib
import os

class SyntheticDataGenerator:
    def __init__(self, dataframe, selected_columns, categorical_columns, numerical_columns, method='ctgan', model_params=None):
        self.dataframe = dataframe[selected_columns].copy()
        self.selected_columns = selected_columns
        self.categorical_columns = categorical_columns
        self.numerical_columns = numerical_columns
        self.method = method.lower()
        self.model_params = model_params if model_params is not None else {}
        self.model = None

        # Validate method
        if self.method not in ['ctgan', 'gaussian_copula']:
            raise ValueError("Unsupported method. Choose 'ctgan' or 'gaussian_copula'.")

        # Set data types for categorical columns to 'object' as required
        for col in self.categorical_columns:
            if col in self.dataframe.columns:
                self.dataframe[col] = self.dataframe[col].astype('object')
        
        # Set data types for numerical columns
        for col in self.numerical_columns:
            if col in self.dataframe.columns:
                self.dataframe[col] = pd.to_numeric(self.dataframe[col], errors='coerce')
        
        # Handle missing values if necessary
        self.dataframe.dropna(inplace=True)  # or use imputation

        # Create metadata for the dataframe
        self.metadata = SingleTableMetadata()
        self.metadata.detect_from_dataframe(self.dataframe)

    def train(self):
        if self.method == 'ctgan':
            # Initialize CTGANSynthesizer using the detected metadata
            self.model = CTGANSynthesizer(metadata=self.metadata, **self.model_params)
            print("Training CTGAN model...")
            self.model.fit(self.dataframe)
            print("CTGAN model trained successfully.")
        elif self.method == 'gaussian_copula':
            self.model = GaussianCopulaSynthesizer(metadata=self.metadata, **self.model_params)
            print("Training Gaussian Copula model...")
            self.model.fit(self.dataframe)
            print("Gaussian Copula model trained successfully.")
        else:
            raise ValueError("Unsupported method. Choose 'ctgan' or 'gaussian_copula'.")

    def generate(self, num_samples):
        if self.model is None:
            raise ValueError("Model is not trained yet. Call the 'train' method first.")
        
        print(f"Generating {num_samples} synthetic samples using {self.method}...")
        synthetic_data = self.model.sample(num_samples)
        print("Synthetic data generation completed.")
        return synthetic_data
    
    def save_model(self, filepath):
        if self.model is None:
            raise ValueError("No trained model to save.")
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump(self.model, filepath)
        print(f"Model saved to {filepath}.")
    
    def load_model(self, filepath):
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"No model found at {filepath}.")
        
        self.model = joblib.load(filepath)
        print(f"Model loaded from {filepath}.")
    
    def get_model(self):
        return self.model
    
    def get_dataframe(self):
        return self.dataframe


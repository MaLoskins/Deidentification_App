from sdv.metadata import SingleTableMetadata
from sdv.single_table import CTGANSynthesizer
from sdv.single_table import GaussianCopulaSynthesizer
import pandas as pd
import joblib
import os

from sdv.metadata import SingleTableMetadata
from sdv.single_table import CTGANSynthesizer, GaussianCopulaSynthesizer
import pandas as pd

class SyntheticDataGenerator:
    def __init__(self, dataframe, selected_columns, method='ctgan', model_params=None):
        self.dataframe = dataframe[selected_columns].copy()
        self.selected_columns = selected_columns
        self.method = method.lower()
        self.model_params = model_params if model_params is not None else {}
        self.model = None

        # Define categorical and numerical columns
        self.categorical_columns = ['gender', 'department']  # Update based on your data
        self.numeric_columns = ['age', 'salary']  # Update based on your data

        # Validate method
        if self.method not in ['ctgan', 'gaussian_copula']:
            raise ValueError("Unsupported method. Choose 'ctgan' or 'gaussian_copula'.")

        # Set data types for categorical columns to 'object' as required
        for col in self.categorical_columns:
            if col in self.dataframe.columns:
                self.dataframe[col] = self.dataframe[col].astype('object')
        
        # Set data types for numerical columns
        for col in self.numeric_columns:
            if col in self.dataframe.columns:
                self.dataframe[col] = pd.to_numeric(self.dataframe[col], errors='coerce')
        
        # Handle missing values if necessary
        self.dataframe.dropna(inplace=True)  # or use imputation

        # Create metadata for the dataframe
        self.metadata = SingleTableMetadata()
        self.metadata.detect_from_dataframe(self.dataframe)  # Automatically detects the column types

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

# Sample DataFrame
data = {
    'age': [25, 32, 47, 51, 62],
    'salary': [50000, 60000, 80000, 90000, 120000],
    'gender': ['Male', 'Female', 'Female', 'Male', 'Male'],
    'department': ['Sales', 'Engineering', 'HR', 'Marketing', 'Engineering']
}

df = pd.DataFrame(data)

# Select columns for synthetic data generation
selected_columns = ['age', 'salary', 'gender', 'department']

# Initialize the generator with CTGAN
synthetic_gen_ctgan = SyntheticDataGenerator(
    dataframe=df,
    selected_columns=selected_columns,
    method='ctgan',
    model_params={
        'epochs': 10000,
        'batch_size': 500,
        'verbose': True  # Optional: Enables training logs
    }
)

# Train the CTGAN model
synthetic_gen_ctgan.train()

# Generate synthetic data
synthetic_data_ctgan = synthetic_gen_ctgan.generate(num_samples=1000)
print(synthetic_data_ctgan)

import matplotlib.pyplot as plt


def plot_distributions(real_data, synthetic_data, column):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))  # Create two subplots side by side
    
    # Plot real data density distribution on the first subplot (axes[0])
    axes[0].hist(real_data[column], bins=30, alpha=0.7, color='blue', edgecolor='black', linewidth=1.2, density=True)
    axes[0].set_title(f'Real {column} Density Distribution')
    axes[0].set_xlabel(column)
    axes[0].set_ylabel('Density')
    
    # Plot synthetic data density distribution on the second subplot (axes[1])
    axes[1].hist(synthetic_data[column], bins=30, alpha=0.7, color='orange', edgecolor='black', linewidth=1.2, density=True)
    axes[1].set_title(f'Synthetic {column} Density Distribution')
    axes[1].set_xlabel(column)
    axes[1].set_ylabel('Density')
    
    # Adjust layout and show the plot
    plt.tight_layout()
    plt.show()

# Example usage
plot_distributions(df, synthetic_data_ctgan, 'age')
plot_distributions(df, synthetic_data_ctgan, 'salary')


import seaborn as sns
import matplotlib.pyplot as plt

def convert_categories_to_integers(df, categorical_columns):
    df_copy = df.copy()
    for col in categorical_columns:
        df_copy[col] = df_copy[col].astype('category').cat.codes
    return df_copy

def compare_correlations(real_data, synthetic_data, categorical_columns):
    # Convert categorical columns to integers for both real and synthetic datasets
    real_data_int = convert_categories_to_integers(real_data, categorical_columns)
    synthetic_data_int = convert_categories_to_integers(synthetic_data, categorical_columns)

    # Now calculate the correlation across all columns, including categorical ones
    real_corr = real_data_int.corr()
    synthetic_corr = synthetic_data_int.corr()

    # Plot correlation heatmaps
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    sns.heatmap(real_corr, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
    plt.title('Real Data Correlation')

    plt.subplot(1, 2, 2)
    sns.heatmap(synthetic_corr, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
    plt.title('Synthetic Data Correlation')

    plt.show()

# Example usage
categorical_columns = ['gender', 'department']
compare_correlations(df, synthetic_data_ctgan, categorical_columns)



from scipy.stats import ks_2samp

def ks_test(real_data, synthetic_data, column):
    stat, p_value = ks_2samp(real_data[column], synthetic_data[column])
    print(f"K-S test for {column}: p-value = {p_value:.4f}")

# Example usage
ks_test(df, synthetic_data_ctgan, 'age')
ks_test(df, synthetic_data_ctgan, 'salary')


def compare_categorical(real_data, synthetic_data, column):
    real_counts = real_data[column].value_counts(normalize=True)
    synthetic_counts = synthetic_data[column].value_counts(normalize=True)
    comparison = pd.DataFrame({'Real': real_counts, 'Synthetic': synthetic_counts})
    print(comparison)

# Example usage
compare_categorical(df, synthetic_data_ctgan, 'gender')
compare_categorical(df, synthetic_data_ctgan, 'department')

# data_integrity_assessor.py

import pandas as pd
import numpy as np
from scipy.stats import entropy
import matplotlib.pyplot as plt
import os

class DataIntegrityAssessor:
    """
    A class to assess the integrity loss when reducing the bin count of categorical variables
    in a Pandas DataFrame by comparing original and binned data.

    Attributes:
        original_df (pd.DataFrame): The original DataFrame with full integrity.
        binned_df (pd.DataFrame): The binned DataFrame with reduced bin counts.
        integrity_report (pd.DataFrame): DataFrame summarizing integrity loss per variable.
        overall_loss (float): Average percentage loss across all variables.
    """

    def __init__(self, original_df: pd.DataFrame, binned_df: pd.DataFrame):
        """
        Initializes the DataIntegrityAssessor with original and binned DataFrames.

        Parameters:
            original_df (pd.DataFrame): The original DataFrame with full integrity.
            binned_df (pd.DataFrame): The binned DataFrame with reduced bin counts.
        """
        self.original_df = original_df.copy()
        self.binned_df = binned_df.copy()
        self.integrity_report = None
        self.overall_loss = None

        self._validate_dataframes()

    def _validate_dataframes(self):
        """
        Validates that the original and binned DataFrames have the same columns.
        """
        if not self.original_df.columns.equals(self.binned_df.columns):
            raise ValueError("Both DataFrames must have the same columns.")

        # Check that all columns are categorical
        for col in self.original_df.columns:
            if not pd.api.types.is_object_dtype(self.original_df[col]) and not pd.api.types.is_categorical_dtype(self.original_df[col]):
                raise TypeError(f"Column '{col}' is not categorical in the original DataFrame.")
            if not pd.api.types.is_object_dtype(self.binned_df[col]) and not pd.api.types.is_categorical_dtype(self.binned_df[col]):
                raise TypeError(f"Column '{col}' is not categorical in the binned DataFrame.")

    @staticmethod
    def calculate_entropy(series: pd.Series) -> float:
        """
        Calculate the entropy of a pandas Series.

        Parameters:
            series (pd.Series): The categorical data series.

        Returns:
            float: Entropy in bits.
        """
        counts = series.value_counts(normalize=True)
        return entropy(counts, base=2)

    def assess_integrity_loss(self):
        """
        Assess the integrity loss between the original and binned DataFrames.

        Populates:
            self.integrity_report (pd.DataFrame): Summary of integrity loss per variable.
            self.overall_loss (float): Average percentage loss across all variables.
        """
        integrity_data = {
            'Variable': [],
            'Original Entropy (bits)': [],
            'Binned Entropy (bits)': [],
            'Entropy Loss (bits)': [],
            'Percentage Loss (%)': []
        }

        for col in self.original_df.columns:
            original_entropy = self.calculate_entropy(self.original_df[col])
            binned_entropy = self.calculate_entropy(self.binned_df[col])
            entropy_loss = original_entropy - binned_entropy
            percentage_loss = (entropy_loss / original_entropy) * 100 if original_entropy != 0 else 0

            integrity_data['Variable'].append(col)
            integrity_data['Original Entropy (bits)'].append(round(original_entropy, 6))
            integrity_data['Binned Entropy (bits)'].append(round(binned_entropy, 6))
            integrity_data['Entropy Loss (bits)'].append(round(entropy_loss, 6))
            integrity_data['Percentage Loss (%)'].append(round(percentage_loss, 2))

        self.integrity_report = pd.DataFrame(integrity_data)
        self.overall_loss = round(self.integrity_report['Percentage Loss (%)'].mean(), 2)

    def generate_report(self) -> pd.DataFrame:
        """
        Generates and returns the integrity loss report.

        Returns:
            pd.DataFrame: DataFrame summarizing the integrity loss per variable.
        """
        if self.integrity_report is None:
            self.assess_integrity_loss()
        return self.integrity_report.copy()

    def save_report(self, filepath: str):
        """
        Saves the integrity loss report to a CSV file.

        Parameters:
            filepath (str): The path where the report will be saved.
        """
        if self.integrity_report is None:
            self.assess_integrity_loss()
        self.integrity_report.to_csv(filepath, index=False)
        print(f"Integrity report saved to {os.path.abspath(filepath)}")

    def plot_entropy(self, save_path: str = None, figsize: tuple = (10, 6)):
        """
        Plots the original and binned entropy for each variable.

        Parameters:
            save_path (str, optional): If provided, saves the plot to the specified path.
        """
        if self.integrity_report is None:
            self.assess_integrity_loss()

        variables = self.integrity_report['Variable']
        original_entropy = self.integrity_report['Original Entropy (bits)']
        binned_entropy = self.integrity_report['Binned Entropy (bits)']

        x = np.arange(len(variables))  # the label locations
        width = 0.35  # the width of the bars

        fig, ax = plt.subplots(figsize=figsize)
        rects1 = ax.bar(x - width/2, original_entropy, width, label='Original Entropy')
        rects2 = ax.bar(x + width/2, binned_entropy, width, label='Binned Entropy')

        # Add some text for labels, title and custom x-axis tick labels, etc.
        ax.set_ylabel('Entropy (bits)')
        ax.set_title('Original vs Binned Entropy per Variable')
        ax.set_xticks(x)
        ax.set_xticklabels(variables, rotation=45, ha='right')
        ax.legend()

        # Attach a text label above each bar in rects, displaying its height.
        def autolabel(rects):
            for rect in rects:
                height = rect.get_height()
                ax.annotate(f'{height:.2f}',
                            xy=(rect.get_x() + rect.get_width() / 2, height),
                            xytext=(0, 3),  # 3 points vertical offset
                            textcoords="offset points",
                            ha='center', va='bottom')

        autolabel(rects1)
        autolabel(rects2)

        fig.tight_layout()

        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
            print(f"Entropy plot saved to {os.path.abspath(save_path)}")
        else:
            plt.show()

    def get_overall_loss(self) -> float:
        """
        Returns the overall average percentage integrity loss across all variables.

        Returns:
            float: Overall average integrity loss percentage.
        """
        if self.overall_loss is None:
            self.assess_integrity_loss()
        return self.overall_loss

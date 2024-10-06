# data_integrity_assessor.py

import pandas as pd
import numpy as np
from scipy.stats import entropy
import matplotlib.pyplot as plt
import os
from mlxtend.frequent_patterns import apriori, association_rules

class DataIntegrityAssessor:
    def __init__(self, original_df: pd.DataFrame, binned_df: pd.DataFrame):
        self.original_df = original_df.copy()
        self.binned_df = binned_df.copy()
        self.integrity_report = None
        self.overall_loss = None
        self.association_report = None

        self._validate_dataframes()

    def _validate_dataframes(self):
        if not self.original_df.columns.equals(self.binned_df.columns):
            raise ValueError("Both DataFrames must have the same columns.")

        for col in self.original_df.columns:
            if not pd.api.types.is_object_dtype(self.original_df[col]) and not pd.api.types.is_categorical_dtype(self.original_df[col]):
                raise TypeError(f"Column '{col}' is not categorical in the original DataFrame.")
            if not pd.api.types.is_object_dtype(self.binned_df[col]) and not pd.api.types.is_categorical_dtype(self.binned_df[col]):
                raise TypeError(f"Column '{col}' is not categorical in the binned DataFrame.")

    @staticmethod
    def calculate_entropy(series: pd.Series) -> float:
        counts = series.value_counts(normalize=True)
        return entropy(counts, base=2)

    def assess_integrity_loss(self):
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
        if self.integrity_report is None:
            self.assess_integrity_loss()
        return self.integrity_report.copy()

    def save_report(self, filepath: str):
        if self.integrity_report is None:
            self.assess_integrity_loss()
        self.integrity_report.to_csv(filepath, index=False)
        print(f"Integrity report saved to {os.path.abspath(filepath)}")

    def plot_entropy(self, save_path: str = None, figsize: tuple = (10, 6)):
        if self.integrity_report is None:
            self.assess_integrity_loss()

        variables = self.integrity_report['Variable']
        original_entropy = self.integrity_report['Original Entropy (bits)']
        binned_entropy = self.integrity_report['Binned Entropy (bits)']

        x = np.arange(len(variables))  # the label locations
        width = 0.35  # the width of the bars

        fig, ax = plt.subplots(figsize=figsize)
        rects1 = ax.bar(x - width/2, original_entropy, width, label='Original Entropy', alpha=0.7, edgecolor='blue', color='skyblue')
        rects2 = ax.bar(x + width/2, binned_entropy, width, label='Binned Entropy', alpha=0.7, edgecolor='orange', color='lightcoral')

        ax.set_ylabel('Entropy (bits)')
        ax.set_title('Original vs Binned Entropy per Variable')
        ax.set_xticks(x)
        ax.set_xticklabels(variables, rotation=45, ha='right')
        ax.legend()

        def autolabel(rects):
            for rect in rects:
                height = rect.get_height()
                ax.annotate(f'{height:.2f}',
                            xy=(rect.get_x() + rect.get_width() / 2, height),
                            xytext=(0, 3),  
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

        return fig  # Return the Figure object

    def get_overall_loss(self) -> float:
        if self.overall_loss is None:
            self.assess_integrity_loss()
        return self.overall_loss

    def generate_association_rules(self, min_support: float = 0.1, min_threshold: float = 0.5) -> tuple:
        """
        Generate association rules for both original and binned DataFrames.
        
        Returns:
            association_report (pd.DataFrame): Combined rules for comparison.
            original_rules (pd.DataFrame): Rules from original DataFrame.
            binned_rules (pd.DataFrame): Rules from binned DataFrame.
        """
        # Convert to boolean data type to avoid warnings
        original_df_onehot = pd.get_dummies(self.original_df, dtype=bool)
        binned_df_onehot = pd.get_dummies(self.binned_df, dtype=bool)

        # Align the original and binned data so both have the same columns
        original_df_onehot, binned_df_onehot = original_df_onehot.align(binned_df_onehot, fill_value=False, axis=1)

        # Apply Apriori algorithm to find frequent itemsets
        original_frequent_itemsets = apriori(original_df_onehot, min_support=min_support, use_colnames=True)
        binned_frequent_itemsets = apriori(binned_df_onehot, min_support=min_support, use_colnames=True)

        if original_frequent_itemsets.empty:
            print("No frequent itemsets generated for the original data.")
        if binned_frequent_itemsets.empty:
            print("No frequent itemsets generated for the binned data.")

        # Generate association rules based on confidence metric
        original_rules = association_rules(original_frequent_itemsets, metric="confidence", min_threshold=min_threshold)
        binned_rules = association_rules(binned_frequent_itemsets, metric="confidence", min_threshold=min_threshold)

        if original_rules.empty:
            print("No association rules generated for the original data.")
        if binned_rules.empty:
            print("No association rules generated for the binned data.")

        # Combine original and binned rules for comparison
        # We'll create a unique identifier for each rule to facilitate merging
        original_rules['rule_id'] = original_rules['antecedents'].astype(str) + " -> " + original_rules['consequents'].astype(str)
        binned_rules['rule_id'] = binned_rules['antecedents'].astype(str) + " -> " + binned_rules['consequents'].astype(str)

        # Merge the rules on rule_id
        association_report = pd.merge(
            original_rules,
            binned_rules,
            on='rule_id',
            suffixes=('_original', '_binned'),
            how='outer'
        ).fillna(0)  # Fill NaN where rules don't exist in one of the datasets

        self.association_report = association_report
        return self.association_report, original_rules, binned_rules

    def summarize_association_rules(self, association_report: pd.DataFrame) -> pd.DataFrame:
        """Generate a summary DataFrame comparing original and binned rules."""
        summary_data = {
            'Rule': association_report['rule_id'],
            'Original Support': association_report['support_original'],
            'Binned Support': association_report['support_binned'],
            'Original Confidence': association_report['confidence_original'],
            'Binned Confidence': association_report['confidence_binned'],
            'Original Lift': association_report['lift_original'],
            'Binned Lift': association_report['lift_binned']
        }

        summary_df = pd.DataFrame(summary_data)

        return summary_df

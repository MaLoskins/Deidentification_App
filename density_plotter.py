# density_plotter.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Optional, Tuple
import math
import os

class DensityPlotter:
    """
    A class to generate density plots for selected categorical columns in a Pandas DataFrame.

    Supported Data Types:
        - Categorical : Plotted as density plots based on bin categories.



    Attributes:
        dataframe (pd.DataFrame): The input DataFrame to plot.
        category_columns (List[str]): List of categorical date columns to plot.
        figsize (Tuple[int, int]): Size of the overall figure (width, height).
        save_path (Optional[str]): Path to save the plot image. If None, the plot is displayed.
        plot_style (str): Seaborn style to use for the plots.
    """

    def __init__(
        self,
        dataframe: pd.DataFrame,
        category_columns: List[str],
        figsize: Tuple[int, int] = (20, 20),
        save_path: Optional[str] = None,
        plot_style: str = 'whitegrid'
    ):
        """
        Initialize the DensityPlotter with a DataFrame and selected columns.

        Parameters:
            dataframe (pd.DataFrame): The input DataFrame to plot.
            category_columns (List[str]): List of categorical date columns to plot.
            figsize (Tuple[int, int], optional): Size of the overall figure. Default is (20, 20).
            save_path (Optional[str], optional): Path to save the plot image. If None, the plot is displayed.
            plot_style (str, optional): Seaborn style for the plots. Default is 'whitegrid'.
        """
        self.dataframe = dataframe.copy()
        self.category_columns = category_columns
        self.figsize = figsize
        self.save_path = save_path
        self.plot_style = plot_style

        # Initialize Seaborn style
        sns.set_style(self.plot_style)

        # Validate columns
        self._valicategory_columns()

    def _valicategory_columns(self):
        """
        Validate that the specified columns exist in the DataFrame and have appropriate data types.
        """
        for col in self.category_columns:
            if col not in self.dataframe.columns:
                raise ValueError(f"Date column '{col}' does not exist in the DataFrame.")
            if not pd.api.types.is_categorical_dtype(self.dataframe[col]):
                raise TypeError(f"Date column '{col}' is not of categorical dtype.")


    def plot_grid(self):
        """
        Generate and display/save the grid of density plots for the selected columns.
        """
        total_plots = len(self.category_columns)
        if total_plots == 0:
            print("No columns to plot.")
            return

        # Determine grid size (rows and columns)
        cols = math.ceil(math.sqrt(total_plots))
        rows = math.ceil(total_plots / cols)

        # Create subplots
        fig, axes = plt.subplots(rows, cols, figsize=self.figsize)
        axes = axes.flatten()  # Flatten in case of multiple rows

        plot_idx = 0  # Index to track the current subplot

        # Plot Date Columns as Density Plots
        for col in self.category_columns:
            ax = axes[plot_idx]
            try:
                # Get the counts for each category
                counts = self.dataframe[col].value_counts().sort_index()
                # Create a density plot using counts
                sns.kdeplot(data=counts.values, ax=ax, fill=True, color='orange')
                ax.set_title(col)             # Add title
                ax.set_ylabel('Density')
                ax.set_xlabel('')             # Remove x label
                ax.set_xticks([])             # Remove x ticks
            except Exception as e:
                print(f"Failed to plot date column '{col}': {e}")
            plot_idx += 1


        # Remove any unused subplots
        for idx in range(plot_idx, len(axes)):
            fig.delaxes(axes[idx])

        plt.tight_layout()

        # Save or show the plot
        if self.save_path:
            # Ensure the directory exists
            save_dir = os.path.dirname(self.save_path)
            if save_dir:
                os.makedirs(save_dir, exist_ok=True)
            try:
                plt.savefig(self.save_path, dpi=300)
                print(f"Density plots saved to {self.save_path}")
            except Exception as e:
                print(f"Failed to save plot to '{self.save_path}': {e}")
        else:
            plt.show()

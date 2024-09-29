# density_plotter.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Optional, Tuple
import math
import os

class DensityPlotter:
    def __init__(
        self,
        dataframe: pd.DataFrame,
        category_columns: List[str],
        figsize: Tuple[int, int] = (20, 20),
        save_path: Optional[str] = None,
        plot_style: str = 'whitegrid'
    ):
        self.dataframe = dataframe.copy()
        self.category_columns = category_columns
        self.figsize = figsize
        self.save_path = save_path
        self.plot_style = plot_style

        sns.set_style(self.plot_style)

        self._valicategory_columns()

    def _valicategory_columns(self):
        for col in self.category_columns:
            if col not in self.dataframe.columns:
                raise ValueError(f"Date column '{col}' does not exist in the DataFrame.")
            if not pd.api.types.is_categorical_dtype(self.dataframe[col]):
                raise TypeError(f"Date column '{col}' is not of categorical dtype.")

    def plot_grid(self):
        total_plots = len(self.category_columns)
        if total_plots == 0:
            print("No columns to plot.")
            return

        cols = math.ceil(math.sqrt(total_plots))
        rows = math.ceil(total_plots / cols)

        fig, axes = plt.subplots(rows, cols, figsize=self.figsize)
        axes = axes.flatten()

        plot_idx = 0

        for col in self.category_columns:
            ax = axes[plot_idx]
            try:
                counts = self.dataframe[col].value_counts().sort_index()
                sns.kdeplot(data=counts.values, ax=ax, fill=True, color='orange')
                ax.set_title(col)
                ax.set_ylabel('Density')
                ax.set_xlabel('')
                ax.set_xticks([])
            except Exception as e:
                print(f"Failed to plot date column '{col}': {e}")
            plot_idx += 1

        for idx in range(plot_idx, len(axes)):
            fig.delaxes(axes[idx])

        plt.tight_layout()

        if self.save_path:
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
        
        if self.save_path:
            plt.savefig(self.save_path, dpi=300)
            print(f"Density plots saved to {self.save_path}")
        else:
            plt.show()
        
        return fig  # Add this line to return the Figure object

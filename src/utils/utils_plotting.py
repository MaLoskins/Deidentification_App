# src/utils/utils_plotting.py

import streamlit as st
import matplotlib.pyplot as plt
from typing import List, Optional, Tuple
import traceback
import seaborn as sns
import pandas as pd
import math
import os

def plot_entropy(assessor):
    """
    Generates an entropy plot.
    Returns:
        fig_entropy (matplotlib.figure.Figure): Entropy plot figure.
    """
    try:
        fig_entropy = assessor.plot_entropy(figsize=(15, 4))
        return fig_entropy
    except Exception as e:
        st.error(f"Error plotting entropy: {e}")
        st.error(traceback.format_exc())
        return None

def plot_density_plots_streamlit(original_df, binned_df, selected_columns):
    """Generate and display density plots for original and binned data side by side."""
    
    try:
        # Initialize DensityPlotter for original data
        fig_orig = plot_density_barplots(
            dataframe=original_df[selected_columns], 
            columns=selected_columns,
            save_path=None  # Set path if saving is needed
        )
        
        # Initialize DensityPlotter for binned data
        fig_binned = plot_density_barplots(
            dataframe=binned_df[selected_columns], 
            columns=selected_columns,
            save_path=None  # Set path if saving is needed
        )
    
        
        # Create tabs for Original and Binned plots
        tab1, tab2 = st.tabs(["Original Data", "Binned Data"])
        
        with tab1:
            st.pyplot(fig_orig)
        
        with tab2:
            st.pyplot(fig_binned)
    
    except Exception as e:
        st.error(f"Failed to generate density plots: {e}")

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import math
import os
from typing import List, Tuple, Optional

def plot_density_barplots(
    dataframe: pd.DataFrame,
    columns: List[str],
    figsize: Tuple[int, int] = (20, 10),
    save_path: Optional[str] = None,
    plot_style: str = 'whitegrid'
) -> plt.Figure:
    """
    Generate and display/save bar plots with density overlays for selected columns in a DataFrame.
    Handles numerical, categorical, and datetime types dynamically.

    Parameters:
        dataframe (pd.DataFrame): The input DataFrame to plot.
        columns (List[str]): List of columns to plot.
        figsize (Tuple[int, int], optional): Size of the overall figure. Default is (20, 10).
        save_path (Optional[str], optional): Path to save the plot image. If None, the plot is displayed.
        plot_style (str, optional): Seaborn style for the plots. Default is 'whitegrid'.

    Returns:
        fig (plt.Figure): The matplotlib Figure object containing the plots.
    """
    # Create a copy of the dataframe to avoid modifying the original
    df = dataframe.copy()

    # Set the desired Seaborn style
    sns.set_style(plot_style)

    # Validate the specified columns
    supported_dtypes = ['number', 'datetime', 'category']
    for col in columns:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' does not exist in the DataFrame.")
        if not (
            pd.api.types.is_numeric_dtype(df[col]) or
            pd.api.types.is_datetime64_any_dtype(df[col]) or
            pd.api.types.is_categorical_dtype(df[col])
        ):
            raise TypeError(
                f"Column '{col}' is not of a supported dtype (number, datetime, category)."
            )

    total_plots = len(columns)
    if total_plots == 0:
        print("No columns to plot.")
        return

    # Determine the grid size for subplots
    cols_grid = math.ceil(math.sqrt(total_plots))
    rows_grid = math.ceil(total_plots / cols_grid)

    # Create subplots
    fig, axes = plt.subplots(rows_grid, cols_grid, figsize=figsize)
    if total_plots == 1:
        axes = [axes]  # Ensure axes is iterable
    else:
        axes = axes.flatten()  # Flatten in case of multiple rows

    for plot_idx, col in enumerate(columns):
        ax = axes[plot_idx]
        try:
            if pd.api.types.is_numeric_dtype(df[col]):
                # Numerical column
                sns.histplot(
                    data=df,
                    x=col,
                    kde=True,
                    stat="density",
                    ax=ax,
                    color='blue',
                    alpha=0.6
                )
                ax.set_title(f"{col} (Numerical)")
                ax.set_xlabel(col)
                ax.set_ylabel('Density')

            elif pd.api.types.is_categorical_dtype(df[col]):
                # Categorical column
                counts = df[col].value_counts().sort_index()
                categories = counts.index.astype(str)
                sns.barplot(
                    x=categories,
                    y=counts.values / counts.values.sum(),
                    ax=ax,
                    color='blue',
                    alpha=0.6,
                    label='Bar Plot'
                )

                # Optional density approximation for categorical data
                sns.kdeplot(
                    x=range(len(counts)),
                    weights=counts.values,
                    ax=ax,
                    color='orange',
                    fill=True,
                    alpha=0.2,
                    bw_adjust=0.5,
                    label='Density'
                )

                if len(categories) <= 20:
                    ax.set_xticks(range(len(categories)))
                    ax.set_xticklabels(categories, rotation=45, ha='right')
                else:
                    ax.set_xticks([0, len(categories) - 1])
                    ax.set_xticklabels([str(categories[0]), str(categories[-1])], rotation=45, ha='right')

                ax.set_title(f"{col} (Categorical)")
                ax.set_ylabel('Proportion')
                ax.set_xlabel('')
                ax.legend()

            elif pd.api.types.is_datetime64_any_dtype(df[col]):
                # Datetime column
                # If timezone aware, convert to UTC and then to naive
                if pd.api.types.is_datetime64tz_dtype(df[col]):
                    df[col] = df[col].dt.tz_convert('UTC').dt.tz_localize(None)

                sns.histplot(
                    data=df,
                    x=col,
                    kde=True,
                    stat="density",
                    ax=ax,
                    color='blue',
                    alpha=0.6
                )
                ax.set_title(f"{col} (Datetime)")
                ax.set_xlabel(col)
                ax.set_ylabel('Density')
                fig.autofmt_xdate(rotation=45)

            else:
                # Should not reach here due to earlier validation
                continue

        except Exception as e:
            print(f"Failed to plot column '{col}': {e}")

    # Remove any unused subplots
    for idx in range(total_plots, len(axes)):
        fig.delaxes(axes[idx])

    plt.tight_layout()  # Adjust subplots to fit into the figure area

    # Save or show the plot
    if save_path:
        # Ensure the directory exists
        save_dir = os.path.dirname(save_path)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        try:
            plt.savefig(save_path, dpi=300)
            print(f"Density bar plots saved to {save_path}")
        except Exception as e:
            print(f"Failed to save plot to '{save_path}': {e}")
    else:
        plt.show()

    return fig


# src/utils/utils_plotting.py

import streamlit as st
import matplotlib.pyplot as plt
from typing import List, Optional, Tuple
import traceback
import seaborn as sns
import pandas as pd
import math
import os
import numpy as np
import datetime


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

# src/utils/utils_plotting.py

# src/utils/utils_plotting.py

def plot_distributions(original_df, synthetic_df, column):
    """Plot and compare distributions of a numerical, categorical, or datetime column."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    if pd.api.types.is_datetime64_any_dtype(original_df[column]):
        # For datetime columns, plot distributions by frequency over time
        original_counts = original_df[column].dt.to_period('M').value_counts().sort_index()
        synthetic_counts = synthetic_df[column].dt.to_period('M').value_counts().sort_index()
        
        original_counts.plot(label='Original', ax=ax)
        synthetic_counts.plot(label='Synthetic', ax=ax)
        ax.set_title(f'Date Distribution Comparison for Column: {column}')
        ax.set_xlabel('Month')
        ax.set_ylabel('Frequency')
    elif pd.api.types.is_numeric_dtype(original_df[column]):
        sns.kdeplot(original_df[column], label='Original', ax=ax)
        sns.kdeplot(synthetic_df[column], label='Synthetic', ax=ax)
        ax.set_title(f'Distribution Comparison for Numerical Column: {column}')
    else:
        original_counts = original_df[column].value_counts(normalize=True)
        synthetic_counts = synthetic_df[column].value_counts(normalize=True)
        original_counts.plot(kind='bar', alpha=0.5, label='Original', ax=ax)
        synthetic_counts.plot(kind='bar', alpha=0.5, label='Synthetic', ax=ax)
        ax.set_title(f'Distribution Comparison for Categorical Column: {column}')
    
    ax.legend()
    st.pyplot(fig)

def plot_date_distributions(original_df, synthetic_df, column):
    """Plot and compare distributions of a datetime column."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Convert to period (e.g., monthly) for aggregation
    original_period = original_df[column].dt.to_period('M').value_counts().sort_index()
    synthetic_period = synthetic_df[column].dt.to_period('M').value_counts().sort_index()
    
    original_period.plot(label='Original', ax=ax)
    synthetic_period.plot(label='Synthetic', ax=ax)
    
    ax.set_title(f'Date Distribution Comparison for Column: {column}')
    ax.set_xlabel('Month')
    ax.set_ylabel('Frequency')
    ax.legend()
    st.pyplot(fig)

def compare_correlations(original_df, synthetic_df, categorical_columns):
    """Compare correlation matrices of original and synthetic data."""
    # Compute correlation matrices
    original_corr = original_df.select_dtypes(include=['number']).corr()
    synthetic_corr = synthetic_df.select_dtypes(include=['number']).corr()

    # Plot original correlations
    fig, ax = plt.subplots(1, 2, figsize=(16, 6))
    sns.heatmap(original_corr, annot=True, fmt=".2f", cmap='coolwarm', ax=ax[0])
    ax[0].set_title('Original Data Correlation Matrix')

    # Plot synthetic correlations
    sns.heatmap(synthetic_corr, annot=True, fmt=".2f", cmap='coolwarm', ax=ax[1])
    ax[1].set_title('Synthetic Data Correlation Matrix')

    st.pyplot(fig)


def plot_distributions(real_data, synthetic_data, column):
    """Plot the distributions of a specific column in real and synthetic data."""
    try:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        if pd.api.types.is_numeric_dtype(real_data[column]):
            sns.kdeplot(real_data[column].dropna(), ax=axes[0], color='blue')
            axes[0].set_title(f'Real {column} Density Distribution')
            sns.kdeplot(synthetic_data[column].dropna(), ax=axes[1], color='orange')
            axes[1].set_title(f'Synthetic {column} Density Distribution')
        else:
            sns.countplot(x=real_data[column], ax=axes[0], color='blue')
            axes[0].set_title(f'Real {column} Distribution')
            sns.countplot(x=synthetic_data[column], ax=axes[1], color='orange')
            axes[1].set_title(f'Synthetic {column} Distribution')
        plt.tight_layout()
        st.pyplot(fig)
    except Exception as e:
        st.error(f"Error plotting distributions: {e}")
        st.error(traceback.format_exc())


def convert_categories_to_integers(df, categorical_columns):
    """Convert categorical columns to integer codes."""
    df_copy = df.copy()
    for col in categorical_columns:
        if col in df_copy.columns:
            df_copy[col] = df_copy[col].astype('category').cat.codes
    return df_copy

def compare_correlations(real_data, synthetic_data, categorical_columns):
    """Compare the correlation matrices of real and synthetic data."""
    try:
        # Convert categorical columns to integers
        real_data_int = convert_categories_to_integers(real_data, categorical_columns)
        synthetic_data_int = convert_categories_to_integers(synthetic_data, categorical_columns)
        
        # Calculate correlations
        real_corr = real_data_int.corr()
        synthetic_corr = synthetic_data_int.corr()
        
        # Create subplots for side-by-side heatmaps
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
                                            
        # Plot real data correlation heatmap
        sns.heatmap(
            real_corr,
            annot=True,
            fmt=".2f",
            cmap='coolwarm',
            vmin=-1,
            vmax=1,
            annot_kws={"size": 5},
            ax=axes[0]
        )
        axes[0].set_title('Real Data Correlation')
        
        # Plot synthetic data correlation heatmap
        sns.heatmap(
            synthetic_corr,
            annot=True,
            fmt=".2f",
            cmap='coolwarm',
            vmin=-1,
            vmax=1,
            annot_kws={"size": 5},
            ax=axes[1]
        )
        axes[1].set_title('Synthetic Data Correlation')
        
        # Adjust layout and display the plot
        plt.tight_layout()
        st.pyplot(fig)
    except Exception as e:
        st.error(f"Error comparing correlations: {e}")
        st.error(traceback.format_exc())

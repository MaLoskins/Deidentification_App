# src/utils/utils_plotting.py

import streamlit as st
import matplotlib.pyplot as plt
from src.binning import DensityPlotter
import traceback

def plot_density_plots(original_df, binned_df, selected_columns_binning):
    """
    Generates density plots for original and binned data.
    Returns:
        fig_orig (matplotlib.figure.Figure): Original data density plots.
        fig_binned (matplotlib.figure.Figure): Binned data density plots.
    """
    try:
        if len(selected_columns_binning) > 1:
            plotter_orig = DensityPlotter(
                dataframe=original_df,
                category_columns=selected_columns_binning,
                figsize=(15, 4),                     
                plot_style='ticks'
            )
            fig_orig = plotter_orig.plot_grid()

            plotter_binned = DensityPlotter(
                dataframe=binned_df,
                category_columns=selected_columns_binning,
                figsize=(15, 4),
                plot_style='ticks'
            )
            fig_binned = plotter_binned.plot_grid()
            return fig_orig, fig_binned
        else:
            st.info("🔄 **Please select more than one column to display density plots.**")
            return None, None
    except Exception as e:
        st.error(f"Error plotting density: {e}")
        st.error(traceback.format_exc())
        return None, None

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

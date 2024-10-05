# src/utils/utils_plotting.py

import streamlit as st
import matplotlib.pyplot as plt
from src.binning import DensityPlotter
import traceback

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
        orig_plotter = DensityPlotter(
            dataframe=original_df[selected_columns].astype('category'), 
            category_columns=selected_columns,
            save_path=None  # Set path if saving is needed
        )
        fig_orig = orig_plotter.plot_grid()
        
        # Initialize DensityPlotter for binned data
        binned_plotter = DensityPlotter(
            dataframe=binned_df[selected_columns].astype('category'), 
            category_columns=selected_columns,
            save_path=None  # Set path if saving is needed
        )
        fig_binned = binned_plotter.plot_grid()
        
        # Create tabs for Original and Binned plots
        tab1, tab2 = st.tabs(["Original Data", "Binned Data"])
        
        with tab1:
            st.pyplot(fig_orig)
        
        with tab2:
            st.pyplot(fig_binned)
    
    except Exception as e:
        st.error(f"Failed to generate density plots: {e}")
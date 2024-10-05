# src/utils/utils_plotting.py

import streamlit as st
import matplotlib.pyplot as plt
import os
from .utils_saving import save_dataframe
from src.binning import DensityPlotter
import traceback
from src.config import PLOTS_DIR

def plot_density_plots_and_display(original_df, binned_df, selected_columns_binning, plots_dir):
    """
    Plots the density plots for original and binned data and displays them in Streamlit.
    """
    st.markdown("### 📈 Density Plots")
    if len(selected_columns_binning) > 1:
        density_tab1, density_tab2 = st.tabs(["Original Data", "Binned Data"])
        
        with density_tab1:
            try:
                plotter_orig = DensityPlotter(
                    dataframe=original_df,
                    category_columns=selected_columns_binning,
                    figsize=(15, 4),                     
                    plot_style='ticks'
                )
                fig_orig = plotter_orig.plot_grid()
                original_density_plot_path = os.path.join(plots_dir, 'original_density_plots.png')
                fig_orig.savefig(original_density_plot_path, bbox_inches='tight')
                st.pyplot(fig_orig)
                plt.close(fig_orig)
            except Exception as e:
                st.error(f"Error plotting original data density: {e}")
                st.error(traceback.format_exc())

        with density_tab2:
            try:
                plotter_binned = DensityPlotter(
                    dataframe=binned_df,
                    category_columns=selected_columns_binning,
                    figsize=(15, 4),
                    plot_style='ticks'
                )
                fig_binned = plotter_binned.plot_grid()
                binned_density_plot_path = os.path.join(plots_dir, 'binned_density_plots.png')
                fig_binned.savefig(binned_density_plot_path, bbox_inches='tight')
                st.pyplot(fig_binned)
                plt.close(fig_binned)
            except Exception as e:
                st.error(f"Error plotting binned data density: {e}")
                st.error(traceback.format_exc())
    else:
        st.info("🔄 **Please select more than one column to display density plots.**")

def plot_entropy_and_display(assessor):
    """ 
    Plots the entropy and displays it in Streamlit.
    """
    st.markdown("### 📈 Entropy")
    try:
        # Get the figure object from assessor
        fig_entropy = assessor.plot_entropy(figsize=(15, 4))
        
        # Build the path for saving the plot
        entropy_plot_path = os.path.join(PLOTS_DIR, 'entropy_plot.png')
        
        # Save the figure using the correct path
        fig_entropy.savefig(entropy_plot_path, bbox_inches='tight')
        
        # Display the plot in Streamlit
        st.pyplot(fig_entropy)
        
        # Close the plot after saving and displaying
        plt.close(fig_entropy)
    except Exception as e:
        st.error(f"Error plotting entropy: {e}")
        st.error(traceback.format_exc())

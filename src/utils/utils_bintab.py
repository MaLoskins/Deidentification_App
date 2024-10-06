# src/utils/utils_bintab.py

import streamlit as st
import pandas as pd
import traceback
from src.binning import DataBinner, DataIntegrityAssessor
from src.config import REPORTS_DIR
import os

def get_binning_configuration(Data, selected_columns_binning):
    """
    Generates binning configuration sliders for selected columns.
    """
    bins = {}
    st.markdown("### 📏 Binning Configuration")
    for column in selected_columns_binning:
        max_bins = Data[column].nunique()
        min_bins = 1 if max_bins >= 2 else 0
        default_bins = min(10, max_bins) if max_bins >= 2 else 1

        bins[column] = st.slider(
            f'📏 {column}', 
            min_value=min_bins, 
            max_value=max_bins, 
            value=default_bins, 
            key=f'bin_slider_{column}'
        )
    
    return bins

def perform_binning(original_data, binning_method, bin_dict):
    """Perform the binning process on selected columns with enhanced feedback and error handling."""
    st.markdown("### 🔄 Binning Process")
    try:
        with st.spinner('Binning data...'):
            binner = DataBinner(original_data, method=binning_method.lower())
            binned_df, binned_columns = binner.bin_columns(bin_dict)
            
            st.success("✅ Binning completed successfully!")

            # Display binned columns categorization
            st.markdown("### 🗂️ Binned Columns Categorization")
            for dtype, cols in binned_columns.items():
                if cols:
                    st.write(f"  - **{dtype.capitalize()}**: {', '.join(cols)}")
            
            return original_data, binned_df

    except Exception as e:
        st.error(f"Error during binning: {e}")
        st.error(traceback.format_exc())
        st.stop()

def perform_association_rule_mining(original_df, binned_df, selected_columns):
    """Perform association rule mining on the selected columns of the original and binned DataFrames."""
    original_df = original_df.copy()
    binned_df = binned_df.copy()

    # Convert selected columns to categorical if they aren't already
    for col in selected_columns:
        original_df[col] = original_df[col].astype('category')
        binned_df[col] = binned_df[col].astype('category')

    st.markdown("### 📊 Association Rule Mining Results")
    try:
        # Filter to keep only the selected columns
        original_df_filtered = original_df[selected_columns]
        binned_df_filtered = binned_df[selected_columns]

        # Check if the filtered DataFrames are empty
        if original_df_filtered.empty or binned_df_filtered.empty:
            st.warning("🔍 One or both of the DataFrames do not contain any data in the selected columns.")
            return

        assessor = DataIntegrityAssessor(original_df_filtered, binned_df_filtered)

        # Generate association rules
        association_report, original_rules, binned_rules = assessor.generate_association_rules()

        # Check if rules were generated
        if association_report.empty:
            st.warning("No association rules were generated for the selected columns.")
            return

        # Summarize the association rules
        summary_df = assessor.summarize_association_rules(association_report)
        st.subheader("📋 Summary of Association Rules")
        st.dataframe(summary_df)

        # Optionally, display key metrics or visualizations
        st.markdown("#### Key Metrics Comparison")
        metrics = summary_df[['Original Support', 'Binned Support', 'Original Confidence', 'Binned Confidence', 'Original Lift', 'Binned Lift']]
        st.dataframe(metrics)

        # Visualize Support Comparison
        st.markdown("#### Support Comparison")
        st.bar_chart(summary_df[['Original Support', 'Binned Support']])

        # Visualize Confidence Comparison
        st.markdown("#### Confidence Comparison")
        st.bar_chart(summary_df[['Original Confidence', 'Binned Confidence']])

        # Visualize Lift Comparison
        st.markdown("#### Lift Comparison")
        st.bar_chart(summary_df[['Original Lift', 'Binned Lift']])

        # Save the association report using the new method in the class
        os.makedirs(REPORTS_DIR, exist_ok=True)
        save_filepath = os.path.join(REPORTS_DIR, 'association_rules_report.csv')
        association_report.to_csv(save_filepath, index=False)
        st.success(f"Association rules report saved to {os.path.abspath(save_filepath)}")

        # Save the original and binned rules
        save_original_rules = os.path.join(REPORTS_DIR, 'original_rules.csv')
        save_binned_rules = os.path.join(REPORTS_DIR, 'binned_rules.csv')
        original_rules.to_csv(save_original_rules, index=False)
        binned_rules.to_csv(save_binned_rules, index=False)
        st.success(f"Original and binned rules saved to {REPORTS_DIR} directory.")

        # Optional: Display Entropy Loss
        assessor.assess_integrity_loss()
        integrity_report = assessor.generate_report()
        st.subheader("🔍 Data Integrity Assessment")
        st.dataframe(integrity_report)

        # Plot Entropy
        st.markdown("#### Entropy Comparison Plot")
        fig = assessor.plot_entropy()
        st.pyplot(fig)

    except Exception as e:
        st.error(f"Error during association rule mining: {e}")
        st.error(traceback.format_exc())

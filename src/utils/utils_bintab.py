# src/utils/utils_bintab.py

import streamlit as st
import pandas as pd
import traceback
from src.binning import DataBinner, DataIntegrityAssessor
from src.config import REPORTS_DIR
import os

def get_binning_configuration(Data, selected_columns_binning):
    """
    Generates binning configuration sliders for selected columns in two columns.
    """
    bins = {}
    st.markdown("### 📏 Binning Configuration")
    
    # Create two columns
    col1, col2 = st.columns(2)
    
    for i, column in enumerate(selected_columns_binning):
        max_bins = Data[column].nunique()
        min_bins = 1 if max_bins >= 2 else 0
        default_bins = min(10, max_bins) if max_bins >= 2 else 1
        
        # Alternate between the two columns
        with col1 if i % 2 == 0 else col2:
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

            return original_data, binned_df, binned_columns
        
    except Exception as e:
        st.error(f"❌ An error occurred during binning: {e}")
        return original_data, None, None


def binning_summary(binned_df, binned_columns, bin_dict):
    try:
        def display_section(title, data, index_col, width=2000, no_data_msg="No data to display."):
            """
            Helper function to display a section with a title and a dataframe.
            """
            st.markdown(f"### {title}")
            if data:
                df = pd.DataFrame(data)
                df.set_index(index_col, inplace=True)
                st.dataframe(df, width=width)
            else:
                st.write(no_data_msg)

        # Section 1: Binned Columns Categorization
        categorization_data = [
            {"Data Type": dtype.capitalize(), "Columns": ", ".join(cols)}
            for dtype, cols in binned_columns.items()
            if not (dtype.endswith('_bins') or dtype.endswith('_groups')) and cols
        ]
        display_section(
            title="🗂️ Binned Columns Categorization",
            data=categorization_data,
            index_col="Data Type",
            no_data_msg="No columns were binned in this category."
        )

        # Section 2: Bin Ranges
        bin_ranges_data = [
            {"Column": col, "Bin Labels": ", ".join(binned_columns.get(f"{col}_bins", []))}
            for col in bin_dict.keys()
            if f"{col}_bins" in binned_columns
        ]
        display_section(
            title="📊 Bin Ranges",
            data=bin_ranges_data,
            index_col="Column",
            no_data_msg="No bin ranges to display."
        )

        # Section 3: Combined Categories
        combined_categories_data = [
            {"Column": col, "Groups": ", ".join(binned_columns.get(f"{col}_groups", []))}
            for col in bin_dict.keys()
            if f"{col}_groups" in binned_columns
        ]
        display_section(
            title="🍎 Combined Categories",
            data=combined_categories_data,
            index_col="Column",
            no_data_msg="No combined categories to display."
        )

    except Exception as e:
        st.error(f"❌ An error occurred during binning summary: {e}")
            




def perform_association_rule_mining(original_df, binned_df, selected_columns, min_support=0.05, min_threshold=0.05):
    """Perform association rule mining on the selected columns of the original and binned DataFrames."""
    original_df = original_df.copy()
    binned_df = binned_df.copy()

    # Convert selected columns to categorical if they aren't already
    for col in selected_columns:
        original_df[col] = original_df[col].astype('category')
        binned_df[col] = binned_df[col].astype('category')

    try:
        # Filter to keep only the selected columns
        original_df_filtered = original_df[selected_columns]
        binned_df_filtered = binned_df[selected_columns]

        # Check if the filtered DataFrames are empty
        if original_df_filtered.empty or binned_df_filtered.empty:
            st.warning("🔍 One or both of the DataFrames do not contain any data in the selected columns.")
            return

        assessor = DataIntegrityAssessor(original_df_filtered, binned_df_filtered)
        association_report, original_rules, binned_rules = assessor.generate_association_rules(min_support, min_threshold)

        # Check if rules were generated
        if association_report.empty:
            st.warning("No association rules were generated for the selected columns.")
            return

        # Summarize the association rules
        summary_df = assessor.summarize_association_rules(association_report)
        st.subheader("📋 Summary of Association Rules")
        st.dataframe(summary_df, width=2000)

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

        # Save the original and binned rules
        save_original_rules = os.path.join(REPORTS_DIR, 'original_rules.csv')
        save_binned_rules = os.path.join(REPORTS_DIR, 'binned_rules.csv')
        original_rules.to_csv(save_original_rules, index=False)
        binned_rules.to_csv(save_binned_rules, index=False)


    except Exception as e:
        st.error(f"Error during association rule mining: {e}")
        st.error(traceback.format_exc())

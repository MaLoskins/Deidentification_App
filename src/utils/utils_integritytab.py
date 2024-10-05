# src/utils/utils_integritytab.py

import streamlit as st
import traceback
from src.binning import DataIntegrityAssessor, UniqueBinIdentifier

def perform_integrity_assessment(OG_Data_BinTab, Data_BinTab, selected_columns_binning):
    """Assess data integrity after binning."""
    original_for_assessment = OG_Data_BinTab[selected_columns_binning].astype('category')
    data_for_assessment = Data_BinTab[selected_columns_binning]

    return handle_integrity_assessment(original_for_assessment, data_for_assessment)

def handle_integrity_assessment(original_df, binned_df):
    """
    Handles the integrity assessment process, including generating reports and plotting entropy.
    Returns:
        report (pd.DataFrame): Integrity loss report.
        overall_loss (float): Overall average integrity loss.
        entropy_fig (matplotlib.figure.Figure): Entropy plot figure.
    """
    try:
        assessor = DataIntegrityAssessor(original_df=original_df, binned_df=binned_df)
        assessor.assess_integrity_loss()
        report = assessor.generate_report()
        overall_loss = assessor.get_overall_loss()
        entropy_fig = assessor.plot_entropy(figsize=(15, 4))
        return report, overall_loss, entropy_fig
    except Exception as e:
        st.error(f"Error during integrity assessment: {e}")
        st.error(traceback.format_exc())
        return None, None, None

def perform_unique_identification_analysis(original_for_assessment, data_for_assessment, selected_columns_uniquetab, min_comb_size, max_comb_size):
    """Handle the Unique Identification Analysis process."""
    try:
        assessor = UniqueBinIdentifier(original_df=original_for_assessment, binned_df=data_for_assessment)
        results = assessor.find_unique_identifications(
            min_comb_size=min_comb_size, 
            max_comb_size=max_comb_size, 
            columns=selected_columns_uniquetab
        )
        return results
    except Exception as e:
        st.error(f"Error during unique identification analysis: {e}")
        st.error(traceback.format_exc())
        return None

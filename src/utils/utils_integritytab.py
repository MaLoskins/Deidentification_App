# src/utils/utils_integritytab.py

import streamlit as st
import traceback
from .utils_saving import save_dataframe
from .utils_plotting import plot_entropy_and_display
from src.binning import DataIntegrityAssessor, UniqueBinIdentifier


def perform_integrity_assessment(OG_Data_BinTab, Data_BinTab, selected_columns_binning):
    """Assess data integrity after binning."""
    original_for_assessment = OG_Data_BinTab[selected_columns_binning].astype('category')
    data_for_assessment = Data_BinTab[selected_columns_binning]

    handle_integrity_assessment(original_for_assessment, data_for_assessment)

def handle_integrity_assessment(original_df, binned_df):
    """
    Handles the integrity assessment process, including generating reports and plotting entropy.
    """
    try:
        assessor = DataIntegrityAssessor(original_df=original_df, binned_df=binned_df)
        assessor.assess_integrity_loss()
        report = assessor.generate_report()
        report_filename = 'Integrity_Loss_Report.csv'
        save_dataframe(report, 'csv', report_filename, 'reports')

        overall_loss = assessor.get_overall_loss()
        st.markdown("### 📄 Integrity Loss Report")
        st.dataframe(report)
        st.write(f"📊 **Overall Average Integrity Loss:** {overall_loss:.2f}%")
        plot_entropy_and_display(assessor)
    except Exception as e:
        st.error(f"Error during integrity assessment: {e}")
        st.error(traceback.format_exc())

def perform_unique_identification_analysis(original_for_assessment, data_for_assessment, selected_columns_uniquetab, min_comb_size, max_comb_size):
    """Handle the Unique Identification Analysis process."""
    try:
        with st.spinner('🔍 Analyzing unique identifications...'):
            progress_bar = st.progress(0)

            def update_progress(combination_counter, total_combinations):
                progress = combination_counter / total_combinations
                st.session_state.progress = min(progress, 1.0)
                progress_bar.progress(st.session_state.progress)

            identifier = UniqueBinIdentifier(original_df=original_for_assessment, binned_df=data_for_assessment)
            results = identifier.find_unique_identifications(
                min_comb_size=min_comb_size, 
                max_comb_size=max_comb_size, 
                columns=selected_columns_uniquetab,
                progress_callback=update_progress
            )
            progress_bar.empty()
            return results
    except Exception as e:
        st.error(f"Error during unique identification analysis: {e}")
        st.error(traceback.format_exc())
        return None


def display_unique_identification_results(results):
    """
    Displays the unique identification analysis results and provides download options.
    """
    if results is not None:
        st.success("✅ Unique Identification Analysis Completed!")
        st.write("📄 **Unique Identification Results:**")
        st.dataframe(results)

        unique_id_filename = 'unique_identifications.csv'
        unique_id_path = save_dataframe(results, 'csv', unique_id_filename, 'unique_identifications')
        
        csv = results.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Results as CSV",
            data=csv,
            file_name='unique_identifications.csv',
            mime='text/csv',
        )

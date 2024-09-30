# Application.py

import streamlit as st
import pandas as pd
from Binning_Tab.data_binner import DataBinner
from Binning_Tab.data_integrity_assessor import DataIntegrityAssessor
from Binning_Tab.unique_bin_identifier import UniqueBinIdentifier
import os
import traceback  # For detailed error logging

# Import utility functions and directory constants
from Binning_Tab.utils import (
    hide_streamlit_style,
    load_data,
    align_dataframes,
    save_dataframe,
    run_processing,
    PLOTS_DIR,
    PROCESSED_DATA_DIR,
    REPORTS_DIR,
    UNIQUE_IDENTIFICATIONS_DIR,
    get_binning_configuration,
    plot_entropy_and_display,
    plot_density_plots_and_display,
    handle_download_binned_data,
    handle_integrity_assessment,
    handle_unique_identification_analysis,
    display_unique_identification_results
)

def setup_page():
    """Configure the Streamlit page and apply custom styles."""
    st.set_page_config(
        page_title="🛠️ Data Processing and Binning Application",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    hide_streamlit_style()
    st.title('🛠️ Data Processing and Binning Application')

def sidebar_inputs():
    """Render the sidebar with file upload, settings, binning options, and info."""
    with st.sidebar:
        st.header("📂 Upload & Settings")
        uploaded_file = st.file_uploader("📤 Upload your dataset", type=['csv', 'pkl'])
        output_file_type = st.selectbox('📁 Select Output File Type', ['csv', 'pkl'], index=0)
        st.markdown("---")

        # Display warning if CSV is selected
        if output_file_type == 'csv':
            st.warning("⚠️ **Note:** Using CSV may result in loss of data types and categories. This will affect subsequent processes. Incompatible columns will be removed from binning as a result. Consider using Pickle for better preservation.")

        st.header("⚙️ Binning Options")
        binning_method = st.selectbox('🔧 Select Binning Method', ['Quantile', 'Equal Width'])
        if binning_method == 'Quantile':
            st.warning("⚠️ **Note:** Using Quantile binning will prevent the output of 'Original Data' Density Plots due to granularity.")
        st.markdown("---")

        st.header("ℹ️ About")
        st.info("""
            This application allows you to upload a dataset, process and bin numerical and datetime columns, 
            assess data integrity post-binning, visualize data distributions, and perform unique identification analysis.
        """)
    
    return uploaded_file, output_file_type, binning_method

def load_and_preview_data(uploaded_file, input_file_type):
    """Load the uploaded data and display a preview."""
    try:
        with st.spinner('Loading data...'):
            Data, error = load_data(input_file_type, uploaded_file)
        if error:
            st.error(error)
            st.stop()
    except Exception as e:
        st.error(f"Error loading data: {e}")
        st.error(traceback.format_exc())
        st.stop()
    
    st.session_state.Original_Data = Data.copy()
    
    st.subheader('📊 Data Preview (Original Data)')
    st.dataframe(st.session_state.Original_Data.head())

def save_raw_data(Data, output_file_type):
    """Save the raw data to a CSV file."""
    mapped_save_type = 'pickle' if output_file_type == 'pkl' else 'csv'
    data_csv_path = 'Data.csv'
    try:
        Data.to_csv(data_csv_path, index=False)
    except Exception as e:
        st.error(f"Error saving Data.csv: {e}")
        st.stop()
    return mapped_save_type, data_csv_path

def run_data_processing(mapped_save_type, output_file_type, data_csv_path):
    """Run the data processing pipeline."""
    run_processing(
        save_type=mapped_save_type,
        output_filename=f'processed_data.{output_file_type}',
        file_path=data_csv_path
    )
    

def load_processed_data(output_file_type):
    """Load the processed data."""
    try:
        if output_file_type == 'csv':
            processed_data = pd.read_csv(os.path.join(PROCESSED_DATA_DIR, f'processed_data.{output_file_type}'))
        else:
            processed_data = pd.read_pickle(os.path.join(PROCESSED_DATA_DIR, f'processed_data.{output_file_type}'))
    except Exception as e:
        st.error(f"Error loading processed data: {e}")
        st.error(traceback.format_exc())
        st.stop()
    
    st.session_state.Processed_Data = processed_data.copy()

def select_columns_to_bin(processed_data):
    """Allow user to select columns to bin."""
    COLUMNS_THAT_CAN_BE_BINNED = processed_data.select_dtypes(
        include=['number', 'datetime', 'datetime64[ns, UTC]', 'datetime64[ns]']
    ).columns.tolist()
    selected_columns = st.multiselect('🔢 Select columns to bin', COLUMNS_THAT_CAN_BE_BINNED, key='selected_columns')
    return selected_columns

def perform_binning(processed_data, selected_columns, binning_method):
    # show dtypes in app for processed_data
    """Perform the binning process on selected columns."""
    try:
        bins = get_binning_configuration(processed_data, selected_columns)
    except Exception as e:
        st.error(f"Error in binning configuration: {e}")
        st.stop()

    st.markdown("### 🔄 Binning Process")
    try:
        with st.spinner('Binning data...'):
            binner = DataBinner(processed_data, method=binning_method.lower())
            binned_df, binned_columns = binner.bin_columns(bins)
            
            # Align both DataFrames (original and binned) to have the same columns
            Data_aligned, binned_df_aligned = align_dataframes(processed_data, binned_df)
    except Exception as e:
        st.error(f"Error during binning: {e}")
        st.error(traceback.format_exc())
        st.stop()
    
    st.success("✅ Binning completed successfully!")

    # Display binned columns categorization
    st.markdown("### 🗂️ Binned Columns Categorization")
    for dtype, cols in binned_columns.items():
        if cols:
            st.write(f"  - **{dtype.capitalize()}**: {', '.join(cols)}")
    
    return Data_aligned, binned_df_aligned

def perform_integrity_assessment(Data_aligned, binned_df_aligned, selected_columns):
    """Assess data integrity after binning."""
    original_for_assessment = Data_aligned[selected_columns].astype('category')
    binned_for_assessment = binned_df_aligned[selected_columns]
    
    handle_integrity_assessment(original_for_assessment, binned_for_assessment, PLOTS_DIR)

def plot_density(original_for_assessment, binned_for_assessment, selected_columns):
    """Plot and display density plots."""
    plot_density_plots_and_display(original_for_assessment, binned_for_assessment, selected_columns, PLOTS_DIR)

def update_session_data(binned_df_aligned, selected_columns):
    """Update session state with binned data."""
    st.session_state.Data = st.session_state.Original_Data.copy()
    # Updating Data for binned selection
    for col in selected_columns:
        st.session_state.Data[col] = binned_df_aligned[col]

    st.subheader('📊 Data Preview (Updated Data)')
    st.dataframe(st.session_state.Data.head())

def download_binned_data():
    """Handle downloading of the binned data."""
    handle_download_binned_data(
        data=st.session_state.Data,
        file_type_download=st.selectbox('📁 Select Download File Type', ['csv', 'pkl'], index=0, key='download_file_type'),
        save_dataframe_func=save_dataframe,
        plots_dir=PLOTS_DIR
    )

def unique_identification_section(original_for_assessment, binned_for_assessment, selected_columns):
    """Handle the Unique Identification Analysis section."""
    st.markdown("### 🔍 Unique Identification Analysis")
    with st.expander("ℹ️ **About:**"):
        st.write("""
            This section analyzes combinations of binned columns to determine how many unique observations
            in the original dataset can be uniquely identified by each combination of bins.
            It helps in understanding the discriminative power of your binned features.
        """)

    # Use a form to group inputs and button together
    with st.form("unique_id_form"):
        st.write("#### 🧮 Configure Unique Identification Analysis")
        # Define bin columns to consider (use selected columns)

        col_count = len(selected_columns)
        col1, col2 = st.columns(2)
        with col1:
            min_comb_size = st.number_input('Minimum Combination Size', min_value=1, max_value=col_count, value=1, step=1)
        with col2:
            max_comb_size = st.number_input('Maximum Combination Size', min_value=min_comb_size, max_value=col_count, value=col_count, step=1)
        
        if max_comb_size > 5:
            st.warning("⚠️  **Note:** Combinations larger than 5 may take a long time to compute depending on bin count.")
        # Submit button
        submit_button = st.form_submit_button(label='🧮 Perform Unique Identification Analysis')

    if submit_button:
        results = handle_unique_identification_analysis(
            original_df=original_for_assessment,
            binned_df=binned_for_assessment,
            bin_columns_list=selected_columns,
            min_comb_size=min_comb_size,
            max_comb_size=max_comb_size
        )
        display_unique_identification_results(results)

def new_section():
    st.markdown("---")
    st.markdown("### 📈 Select Region Columns to retrieve latitude and longitude")
    with st.expander("ℹ️ **About:**"):
        st.write("""
            This new section provides additional data visualizations to help you gain deeper insights into your dataset.
            Explore various charts and graphs to better understand the distribution and relationships within your data.
        """)
    
    # Example visualization: Correlation Heatmap
    if st.button("Generate Correlation Heatmap"):
        try:
            with st.spinner('Generating Correlation Heatmap...'):
                correlation_matrix = st.session_state.Data.corr()
                fig = st.pyplot(correlation_matrix.style.background_gradient(cmap='coolwarm').set_precision(2).to_excel())
                st.write(correlation_matrix)
        except Exception as e:
            st.error(f"Error generating correlation heatmap: {e}")
            st.error(traceback.format_exc())

def main():
    """Main function to orchestrate the Streamlit app."""
    setup_page()
    uploaded_file, output_file_type, binning_method = sidebar_inputs()
    
    if uploaded_file is not None:
        # Determine input file type
        if uploaded_file.name.endswith('.csv'):
            input_file_type = 'csv'
        elif uploaded_file.name.endswith('.pkl'):
            input_file_type = 'pkl'
        else:
            st.error("Unsupported file type! Please upload a CSV or Pickle (.pkl) file.")
            st.stop()
        
        load_and_preview_data(uploaded_file, input_file_type)
        mapped_save_type, data_csv_path = save_raw_data(st.session_state.Original_Data, output_file_type)
        run_data_processing(mapped_save_type, output_file_type, data_csv_path)
        load_processed_data(output_file_type)
        
        selected_columns = select_columns_to_bin(st.session_state.Processed_Data)
        
        if selected_columns:
            Data_aligned, binned_df_aligned = perform_binning(st.session_state.Processed_Data, selected_columns, binning_method)
            perform_integrity_assessment(Data_aligned, binned_df_aligned, selected_columns)
            plot_density(Data_aligned[selected_columns].astype('category'), binned_df_aligned[selected_columns], selected_columns)
            update_session_data(binned_df_aligned, selected_columns)
            download_binned_data()
            unique_identification_section(Data_aligned, binned_df_aligned, selected_columns)
            new_section()  # Adding the new section here
        else:
            st.info("🔄 **Please select at least one column to bin.**")
    else:
        st.info("🔄 **Please upload a file to get started.**")

if __name__ == "__main__":
    main()

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
    get_binning_configuration,  # Ensure correct function signature
    plot_entropy_and_display,    # Newly added
    plot_density_plots_and_display,  # Newly added
    handle_download_binned_data,     # Newly added
    handle_integrity_assessment,     # Newly added
    handle_unique_identification_analysis, # Newly added
    display_unique_identification_results  # Newly added
)

# Set Streamlit page configuration
st.set_page_config(
    page_title="🛠️ Data Processing and Binning Application",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Apply custom styling
hide_streamlit_style()

# Streamlit app starts here
st.title('🛠️ Data Processing and Binning Application')

# Sidebar for inputs and options
with st.sidebar:
    st.header("📂 Upload & Settings")
    uploaded_file = st.file_uploader("📤 Upload your dataset", type=['csv', 'pkl'])
    output_file_type = st.selectbox('📁 Select Output File Type', ['csv', 'pkl'], index=0)
    st.markdown("---")

    # Display warning if csv is the selected file type
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

# Main content area
if uploaded_file is not None:
    # Determine the input file type based on the uploaded file's extension
    if uploaded_file.name.endswith('.csv'):
        input_file_type = 'csv'
    elif uploaded_file.name.endswith('.pkl'):
        input_file_type = 'pkl'
    else:
        st.error("Unsupported file type! Please upload a CSV or Pickle (.pkl) file.")
        st.stop()

    # Load the raw data using the correct input file type
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

    # Store raw data in session state
    st.session_state.Data = Data.copy()
    
    # Display original data preview
    st.subheader('📊 Data Preview (Original Data)')
    st.dataframe(st.session_state.Data.head())

    
    # Determine the correct save_type for DataProcessor
    mapped_save_type = 'pickle' if output_file_type == 'pkl' else 'csv'
    
    # Save the uploaded DataFrame to 'Data.csv'
    data_csv_path = 'Data.csv'
    try:
        Data.to_csv(data_csv_path, index=False)
    except Exception as e:
        st.error(f"Error saving Data.csv: {e}")
        st.stop()
    
    # Run processing with the correct save_type and file_path
    run_processing(
        save_type=mapped_save_type,
        output_filename=f'processed_data.{output_file_type}',
        file_path=data_csv_path
    )

    # Load the processed data
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


    COLUMNS_THAT_CAN_BE_BINNED = st.session_state.Processed_Data.select_dtypes(
        include=['number', 'datetime', 'datetime64[ns, UTC]', 'datetime64[ns]' ]
    ).columns.tolist()
    COLUMNS_THAT_CAN_BE_BINNED = [col for col in COLUMNS_THAT_CAN_BE_BINNED]

    selected_columns = st.multiselect('🔢 Select columns to bin', COLUMNS_THAT_CAN_BE_BINNED, key='selected_columns')

    if selected_columns:
        # Use the utility function to get binning configuration
        try:
            if output_file_type == 'csv':
                processed_data = pd.read_csv(os.path.join(PROCESSED_DATA_DIR, f'processed_data.{output_file_type}'))
            else:
                processed_data = pd.read_pickle(os.path.join(PROCESSED_DATA_DIR, f'processed_data.{output_file_type}'))
            bins = get_binning_configuration(processed_data, selected_columns)  # Corrected argument order
        except Exception as e:
            st.error(f"Error in binning configuration: {e}")
            st.stop()

        # Binning process
        st.markdown("### 🔄 Binning Process")
        try:
            with st.spinner('Binning data...'):
                binner = DataBinner(processed_data, method=binning_method.lower())
                binned_df, binned_columns = binner.bin_columns(bins)
                
                # Align both DataFrames (original and binned) to have the same columns
                Data_aligned, binned_df_aligned = align_dataframes(processed_data, binned_df)
                
        except Exception as e:
            st.error(f"Error during binning: {e}")
            st.error(traceback.format_exc())  # Detailed error log
            st.stop()
        
        st.success("✅ Binning completed successfully!")

        # Display binned columns categorization
        st.markdown("### 🗂️ Binned Columns Categorization")
        for dtype, cols in binned_columns.items():
            if cols:
                st.write(f"  - **{dtype.capitalize()}**: {', '.join(cols)}")

        # **Important:** Exclude identifier columns from integrity assessment
        original_for_assessment = Data_aligned[selected_columns].astype('category')  # Convert to categorical
        binned_for_assessment = binned_df_aligned[selected_columns]  # Already categorical from DataBinner

        # Integrity assessment after binning using utility function
        handle_integrity_assessment(original_for_assessment, binned_for_assessment, PLOTS_DIR)

        # Plot and display density plots using utility function
        plot_density_plots_and_display(original_for_assessment, binned_for_assessment, selected_columns, PLOTS_DIR)
        
        # for columns in selected_columns extract from Data_aligned and replace columns in sessionstate.Data
        for col in selected_columns:
            st.session_state.Data[col] = binned_df_aligned[col]
        
        # Display updated session state data
        st.subheader('📊 Data Preview (Updated Data)')
        st.dataframe(st.session_state.Data.head())
     


        st.markdown("---")

        # Download binned data using utility function
        handle_download_binned_data(
            data=st.session_state.Data,
            file_type_download=st.selectbox('📁 Select Download File Type', ['csv', 'pkl'], index=0, key='download_file_type'),
            save_dataframe_func=save_dataframe,
            plots_dir=PLOTS_DIR
        )

        # Unique Identification Analysis
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
            # Define bin columns to consider (use all binned columns)
            bin_columns_list = list(binned_for_assessment.columns)
            
            # Set combination sizes with input validation
            col_count = len(bin_columns_list)
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
                bin_columns_list=bin_columns_list,
                min_comb_size=min_comb_size,
                max_comb_size=max_comb_size
            )
            display_unique_identification_results(results)
    else:
        st.info("🔄 **Please select at least one column to bin.**")
else:
    st.info("🔄 **Please upload a file to get started.**")

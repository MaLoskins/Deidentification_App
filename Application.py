# Application.py

import streamlit as st
import pandas as pd
from Binning_Tab.data_binner import DataBinner
from Binning_Tab.density_plotter import DensityPlotter
from Binning_Tab.data_integrity_assessor import DataIntegrityAssessor
from Binning_Tab.unique_bin_identifier import UniqueBinIdentifier
import matplotlib.pyplot as plt
import os
import tempfile
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
    get_binning_configuration  # Ensure correct function signature
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
    file_type = st.selectbox('📁 Select Output File Type', ['csv', 'pkl'], index=0)
    st.markdown("---")

    # Display warning if csv is the selected file type
    if file_type == 'csv':
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
    # Load the raw data
    try:
        with st.spinner('Loading data...'):
            Data, error = load_data(file_type, uploaded_file)
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

    # Select columns to bin (only numeric and datetime, excluding identifiers)
    run_processing(save_type=file_type, output_filename=f'processed_data.{file_type}', file_path=f'Data.csv')

    # Load the processed data
    if file_type == 'csv':
        processed_data = pd.read_csv(os.path.join(PROCESSED_DATA_DIR, f'processed_data.{file_type}'))
    else:
        processed_data = pd.read_pickle(os.path.join(PROCESSED_DATA_DIR, f'processed_data.{file_type}'))
    
    st.session_state.Processed_Data = processed_data.copy()


    COLUMNS_THAT_CAN_BE_BINNED = st.session_state.Processed_Data.select_dtypes(
        include=['number', 'datetime', 'datetime64[ns, UTC]', 'datetime64[ns]' ]
    ).columns.tolist()
    COLUMNS_THAT_CAN_BE_BINNED = [col for col in COLUMNS_THAT_CAN_BE_BINNED]

    selected_columns = st.multiselect('🔢 Select columns to bin', COLUMNS_THAT_CAN_BE_BINNED, key='selected_columns')

    if selected_columns:
        # copy of data frame with only selected columns temp saved as Data.csv or Data.pkl
            
        # Use the utility function to get binning configuration
        if file_type == 'csv':
            processed_data = pd.read_csv(os.path.join(PROCESSED_DATA_DIR, f'processed_data.{file_type}'))
        else:
            processed_data = pd.read_pickle(os.path.join(PROCESSED_DATA_DIR, f'processed_data.{file_type}'))
        try:
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
                
                # Extract the list of binned columns
                binned_columns_list = [col for cols in binned_columns.values() for col in cols]
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

        # Integrity assessment after binning
        st.markdown("### 📄 Integrity Loss Report")
        try:
            assessor = DataIntegrityAssessor(original_df=original_for_assessment, binned_df=binned_for_assessment)
            assessor.assess_integrity_loss()
            report = assessor.generate_report()
            
            # Save the report to the 'reports' directory
            report_filename = 'Integrity_Loss_Report.csv'
            report_path = save_dataframe(report, 'csv', report_filename, 'reports')
            
            st.dataframe(report)
            
            overall_loss = assessor.get_overall_loss()
            st.write(f"📊 **Overall Average Integrity Loss:** {overall_loss:.2f}%")
            
            st.markdown("### 📈 Entropy")
            try:
                fig_entropy = assessor.plot_entropy(figsize=(15, 4))  # Create the entropy plot
                # Save the entropy plot
                entropy_plot_path = os.path.join(PLOTS_DIR, 'entropy_plot.png')
                fig_entropy.savefig(entropy_plot_path, bbox_inches='tight')
                plt.close(fig_entropy)  # Close the figure to free memory
                # Display in Streamlit
                st.pyplot(fig_entropy)  # to display
            except Exception as e:
                st.error(f"Error plotting entropy: {e}")
                st.error(traceback.format_exc())  # Detailed error log
        except Exception as e:
            st.error(f"Error during integrity assessment: {e}")
            st.error(traceback.format_exc())  # Detailed error log

        # Tabs for Original and Binned Density Plots
        st.markdown("### 📈 Density Plots")
        if len(selected_columns) > 1:
            density_tab1, density_tab2 = st.tabs(["Original Data", "Binned Data"])
            
            with density_tab1:
                try:
                    plotter_orig = DensityPlotter(
                        dataframe=original_for_assessment,  # Use original categorical data
                        category_columns=selected_columns,
                        figsize=(15, 4),                     
                        save_path=None,  # We'll handle saving manually
                        plot_style='ticks'
                    )
                    
                    # Save the original density plot
                    original_density_plot_path = os.path.join(PLOTS_DIR, 'original_density_plots.png')
                    fig_orig = plotter_orig.plot_grid()
                    fig_orig.savefig(original_density_plot_path, bbox_inches='tight')
                    plt.close(fig_orig)  # Close the figure to free memory
                    st.pyplot(fig_orig)
                except Exception as e:
                    st.error(f"Error plotting original data density: {e}")
                    st.error(traceback.format_exc())  # Detailed error log
            
            with density_tab2:
                try:
                    plotter_binned = DensityPlotter(
                        dataframe=binned_for_assessment,  # Use user-specified binned data
                        category_columns=selected_columns,
                        figsize=(15, 4),                     
                        save_path=None,  # We'll handle saving manually
                        plot_style='ticks'
                    )
                    
                    # Save the binned density plot
                    fig_binned = plotter_binned.plot_grid()  # Get the figure object
                    binned_density_plot_path = os.path.join(PLOTS_DIR, 'binned_density_plots.png')
                    fig_binned.savefig(binned_density_plot_path, bbox_inches='tight')
                    plt.close(fig_binned)  # Close the figure to free memory
                    st.pyplot(fig_binned)
                except Exception as e:
                    st.error(f"Error plotting binned data density: {e}")
                    st.error(traceback.format_exc())  # Detailed error log
        else:
            # Print a message if only one column is selected
            st.info("🔄 **Please select more than one column to display density plots.**")
        
        st.markdown("---")

        # Download binned data
        st.markdown("### 💾 Download Binned Data")
        # Choose file type again for download of bin data
        file_type_download = st.selectbox('📁 Select Download File Type', ['csv', 'pkl'], index=0, key='download_file_type')
        try:
            if file_type_download == 'csv':
                binned_csv = binned_df_aligned.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Download Binned Data as CSV",
                    data=binned_csv,
                    file_name='binned_data.csv',
                    mime='text/csv',
                )
            elif file_type_download == 'pkl':
                # Save pickle to the 'processed_data' directory
                pickle_filename = 'binned_data.pkl'
                pickle_path = save_dataframe(binned_df_aligned, 'pkl', pickle_filename, 'processed_data')
                
                with open(pickle_path, 'rb') as f:
                    binned_pkl = f.read()
                
                st.download_button(
                    label="📥 Download Binned Data as Pickle",
                    data=binned_pkl,
                    file_name='binned_data.pkl',
                    mime='application/octet-stream',
                )
        except Exception as e:
            st.error(f"Error during data download: {e}")
            st.error(traceback.format_exc())  # Detailed error log

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
            try:
                with st.spinner('🔍 Analyzing unique identifications... This may take a while for large datasets.'):
                    # Initialize the UniqueBinIdentifier
                    identifier = UniqueBinIdentifier(original_df=original_for_assessment, binned_df=binned_for_assessment)
                    
                    # Perform unique identification analysis
                    results = identifier.find_unique_identifications(
                        min_comb_size=min_comb_size, 
                        max_comb_size=max_comb_size, 
                        columns=bin_columns_list
                    )
                    
                    # Display the results
                    st.success("✅ Unique Identification Analysis Completed!")
                    st.write("📄 **Unique Identification Results:**")
                    st.dataframe(results.head(20))  # Show top 20 for brevity
                    
                    # Save the results to 'unique_identifications' directory
                    unique_id_filename = 'unique_identifications.csv'
                    unique_id_path = save_dataframe(results, 'csv', unique_id_filename, 'unique_identifications')
                    
                    # Allow user to download the results
                    csv = results.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="📥 Download Results as CSV",
                        data=csv,
                        file_name='unique_identifications.csv',
                        mime='text/csv',
                    )
            except Exception as e:
                st.error(f"Error during unique identification analysis: {e}")
                st.error(traceback.format_exc())  # Detailed error log
    else:
        st.info("🔄 **Please select at least one column to bin.**")
else:
    st.info("🔄 **Please upload a file to get started.**")

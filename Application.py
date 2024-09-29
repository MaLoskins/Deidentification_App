import streamlit as st
import pandas as pd
from Binning_Tab.data_binner import DataBinner
from Binning_Tab.density_plotter import DensityPlotter
from Binning_Tab.data_integrity_assessor import DataIntegrityAssessor
from Binning_Tab.unique_bin_identifier import UniqueBinIdentifier
import matplotlib.pyplot as plt
import os
import tempfile

# Import utility functions and directory constants
from Binning_Tab.utils import (
    hide_streamlit_style,
    load_data,
    align_dataframes,
    save_dataframe,
    PLOTS_DIR,
    PROCESSED_DATA_DIR,
    REPORTS_DIR,
    UNIQUE_IDENTIFICATIONS_DIR
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
    with st.spinner('Loading and processing data...'):
        Data, error = load_data(file_type, uploaded_file)
    if error:
        st.error(error)
    else:
        # Display data preview
        st.subheader('📊 Data Preview (Post Processing)')
        st.dataframe(Data.head())

        # Select columns to bin (only numeric and datetime)
        COLUMNS_THAT_CAN_BE_BINNED = Data.select_dtypes(include=['int64', 'int8', 'float64', 'datetime64[ns]']).columns.tolist()
        selected_columns = st.multiselect('🔢 Select columns to bin', COLUMNS_THAT_CAN_BE_BINNED)

        if selected_columns:
            # Create a dictionary to hold the number of bins for each selected column
            bins = {}
            st.markdown("### 📏 Binning Configuration")

            # Create a grid layout by using columns
            cols_per_row = 2  # Define how many sliders you want per row
            num_cols = len(selected_columns)
            rows = num_cols // cols_per_row + (num_cols % cols_per_row > 0)

            current_col = 0

            for row in range(rows):
                # Create a new row of columns
                cols = st.columns(cols_per_row)
                
                for col_idx in range(cols_per_row):
                    if current_col < num_cols:
                        column = selected_columns[current_col]
                        max_bins = Data[column].nunique()
                        min_bins = 2 if max_bins >= 2 else 1  # At least 2 bins if possible
                        default_bins = min(10, max_bins) if max_bins >= 2 else 1  # Default to 10 or max_unique if lower

                        with cols[col_idx]:
                            if pd.api.types.is_datetime64_any_dtype(Data[column]):
                                default_bins = min(6, max_bins) if max_bins >= 2 else 1
                                bins[column] = st.slider(
                                    f'📏 {column} (Datetime)', 
                                    min_value=1, 
                                    max_value=max_bins,
                                    value=default_bins,
                                    key=column
                                )
                            elif pd.api.types.is_integer_dtype(Data[column]):
                                if max_bins > 2:
                                    bins[column] = st.slider(
                                        f'📏 {column} (Integer)', 
                                        min_value=2, 
                                        max_value=max_bins, 
                                        value=default_bins,
                                        key=column
                                    )
                                elif max_bins == 2:
                                    st.write(f'📏 **{column} (Integer):** 2 (Fixed)')
                                    bins[column] = 2
                                else:
                                    st.write(f'📏 **{column} (Integer):** {max_bins} (Fixed)')
                                    bins[column] = max_bins
                            elif pd.api.types.is_float_dtype(Data[column]):
                                if max_bins > 2:
                                    bins[column] = st.slider(
                                        f'📏 {column} (Float)', 
                                        min_value=2, 
                                        max_value=max_bins, 
                                        value=default_bins,
                                        key=column
                                    )
                                elif max_bins == 2:
                                    st.write(f'📏 **{column} (Float):** 2 (Fixed)')
                                    bins[column] = 2
                                else:
                                    st.write(f'📏 **{column} (Float):** {max_bins} (Fixed)')
                                    bins[column] = max_bins
                            else:
                                if max_bins > 1:
                                    bins[column] = st.slider(
                                        f'📏 {column}', 
                                        min_value=1, 
                                        max_value=max_bins, 
                                        value=default_bins,
                                        key=column
                                    )
                                elif max_bins == 1:
                                    st.write(f'📏 **{column}:** 1 (Fixed)')
                                    bins[column] = 1
                                else:
                                    st.write(f'📏 **{column}:** {max_bins} (Fixed)')
                                    bins[column] = max_bins

                    current_col += 1


            # Binning process
            st.markdown("### 🔄 Binning Process")
            try:
                with st.spinner('Binning data...'):
                    binner = DataBinner(Data, method=binning_method.lower())
                    binned_df, binned_columns = binner.bin_columns(bins)
                    
                    # Ensure the original DataFrame shape
                    unique_values = Data[selected_columns].nunique()
                    unique_values_list = unique_values.tolist()
                    unique_values_dict = dict(zip(selected_columns, unique_values_list))
                    Data, _ = binner.bin_columns(unique_values_dict)
            except Exception as e:
                st.error(f"Error during binning: {e}")
                st.stop()
            
            st.success("✅ Binning completed successfully!")

            # Display binned columns categorization
            st.markdown("### 🗂️ Binned Columns Categorization")
            for dtype, cols in binned_columns.items():
                if cols:
                    st.write(f"  - **{dtype.capitalize()}**: {', '.join(cols)}")

            # Align both DataFrames (original and binned) to have the same columns
            Data_aligned, binned_df_aligned = align_dataframes(Data, binned_df)

            # Integrity assessment after binning
            st.markdown("### 📄 Integrity Loss Report")
            try:
                assessor = DataIntegrityAssessor(original_df=Data_aligned, binned_df=binned_df_aligned)
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
                    st.pyplot(fig_entropy) # to display
                except Exception as e:
                    st.error(f"Error plotting entropy: {e}")
            except Exception as e:
                st.error(f"Error during integrity assessment: {e}")

            # Tabs for Original and Binned Density Plots
            st.markdown("### 📈 Density Plots")
            if len(selected_columns) > 1:
                density_tab1, density_tab2 = st.tabs(["Original Data", "Binned Data"])
                
                with density_tab1:
                    try:
                        plotter_orig = DensityPlotter(
                            dataframe=Data_aligned,
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
                
                with density_tab2:
                    try:
                        plotter_binned = DensityPlotter(
                            dataframe=binned_df_aligned,
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
                bin_columns_list = list(binned_df_aligned.columns)
                
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
                        identifier = UniqueBinIdentifier(original_df=Data_aligned, binned_df=binned_df_aligned)
                        
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
        else:
            st.info("🔄 **Please select at least one column to bin.**")
else:
    st.info("🔄 **Please upload a file to get started.**")

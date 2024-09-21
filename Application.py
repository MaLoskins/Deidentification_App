import streamlit as st
import pandas as pd
from data_binner import DataBinner
from density_plotter import DensityPlotter
from data_integrity_assessor import DataIntegrityAssessor
from unique_bin_identifier import UniqueBinIdentifier
from Process_Data import DataProcessor
import matplotlib.pyplot as plt
import tempfile
import os

# Set Streamlit page configuration
st.set_page_config(
    page_title="🛠️ Data Processing and Binning Application",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Function to hide Streamlit style elements for a cleaner look
def hide_streamlit_style():
    hide_streamlit_style = """
                <style>
                #MainMenu {visibility: hidden;}
                footer {visibility: hidden;}
                </style>
                """
    st.markdown(hide_streamlit_style, unsafe_allow_html=True)

hide_streamlit_style()

# Preprocessing for RawData
def run_processing(save_type='csv', output_filepath='Processed_Data.csv', file_path='Data.csv'):
    try:
        processor = DataProcessor(
            input_filepath=file_path,
            output_filepath=output_filepath,
            report_path='Type_Conversion_Report.csv',
            return_category_mappings=True,
            mapping_directory='Category_Mappings',
            parallel_processing=False,
            date_threshold=0.6,
            numeric_threshold=0.9,
            factor_threshold_ratio=0.2,
            factor_threshold_unique=500,
            dayfirst=True,
            log_level='INFO',
            log_file=None,
            convert_factors_to_int=True,
            date_format=None,  # Keep as None to retain datetime dtype
            save_type=save_type
        )
        processor.process()
    except Exception as e:
        st.error(f"Error during data processing: {e}")
        st.stop()

# File upload and processing function with caching
@st.cache_data(show_spinner=False)
def load_data(file_type, uploaded_file):
    if uploaded_file is None:
        return None, "No file uploaded!"

    try:
        # Determine the appropriate file extension
        file_extension = {
            "pkl": "pkl",
            "csv": "csv"
        }.get(file_type, "csv")  # Default to 'csv' if type is unrecognized

        # Create a temporary file with the correct extension
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_extension}") as tmp_file:
            tmp_file.write(uploaded_file.getbuffer())
            temp_file_path = tmp_file.name

        if file_type == "pkl":
            save_type = 'pickle'
            output_filepath = 'Processed_Data.pkl'
            run_processing(save_type=save_type, output_filepath=output_filepath, file_path=temp_file_path)
            Data = pd.read_pickle('Processed_Data.pkl')
        elif file_type == "csv":
            save_type = 'csv'
            output_filepath = 'Processed_Data.csv'
            run_processing(save_type=save_type, output_filepath=output_filepath, file_path=temp_file_path)
            Data = pd.read_csv('Processed_Data.csv')  
        else:
            return None, "Unsupported file type!"

        # Clean up temporary file
        os.remove(temp_file_path)

        return Data, None
    except Exception as e:
        return None, f"Error loading data: {e}"

# Ensure both DataFrames have the same columns
def align_dataframes(original_df, binned_df):
    try:
        # Identify columns that exist in the original DataFrame but not in the binned DataFrame
        missing_in_binned = original_df.columns.difference(binned_df.columns)
        
        # Retain all original columns that were not binned in the binned DataFrame
        for column in missing_in_binned:
            binned_df[column] = original_df[column]
        
        # Ensure columns are ordered the same way
        binned_df = binned_df[original_df.columns]
        
        return original_df, binned_df
    except Exception as e:
        st.error(f"Error aligning dataframes: {e}")
        st.stop()

# Streamlit app starts here
st.title('🛠️ Data Processing and Binning Application')

# Sidebar for inputs and options
with st.sidebar:
    st.header("📂 Upload & Settings")
    uploaded_file = st.file_uploader("📤 Upload your dataset", type=['csv'])
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

# Main content area organized into tabs
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
                
                st.dataframe(report)
                
                overall_loss = assessor.get_overall_loss()
                st.write(f"📊 **Overall Average Integrity Loss:** {overall_loss:.2f}%")
                
                st.markdown("### 📈 Entropy")
                assessor.plot_entropy(figsize=(15, 4))
                st.pyplot(plt.gcf())
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
                            save_path=None,  # Set to None to display in the app
                            plot_style='ticks'
                        )
                        plotter_orig.plot_grid()
                        st.pyplot(plt.gcf())
                    except Exception as e:
                        st.error(f"Error plotting original data density: {e}")
                
                with density_tab2:
                    try:
                        plotter_binned = DensityPlotter(
                            dataframe=binned_df_aligned,
                            category_columns=selected_columns,
                            figsize=(15, 4),                     
                            save_path=None,  # Set to None to display in the app
                            plot_style='ticks'
                        )
                        plotter_binned.plot_grid()
                        st.pyplot(plt.gcf())
                    except Exception as e:
                        st.error(f"Error plotting binned data density: {e}")
            else:
                # Print a message if only one column is selected
                st.info("🔄 **Please select more than one column to display density plots.**")
            
            st.markdown("---")

            # Download binned data
            st.markdown("### 💾 Download Binned Data")
            # Choose file type again for download of bin data
            file_type_download = st.selectbox('📁 Select Download File Type', ['pkl', 'csv'], index=0)
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
                        binned_data = binned_df_aligned.to_pickle('binned_data.pkl')
                        with open('binned_data.pkl', 'rb') as f:
                            binned_pkl = f.read()
                        st.download_button(
                            label="📥 Download Binned Data as Pickle",
                            data=binned_pkl,
                            file_name='binned_data.pkl',
                            mime='application/octet-stream',
                        )
                        os.remove('binned_data.pkl')
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
                    with st.spinner('🔍 Analysing unique identifications... This may take a while for large datasets.'):
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
            st.info("🔄 **Please select at least one non-binary column to bin.**")
else:
    st.info("🔄 **Please upload a file to get started.**")
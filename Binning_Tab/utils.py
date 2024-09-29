# utils.py

import streamlit as st
import pandas as pd
import tempfile
import os
from Binning_Tab.Process_Data import DataProcessor

# Define the root output directory
OUTPUT_DIR = "outputs"

# Define subdirectories
PROCESSED_DATA_DIR = os.path.join(OUTPUT_DIR, "processed_data")
REPORTS_DIR = os.path.join(OUTPUT_DIR, "reports")
PLOTS_DIR = os.path.join(OUTPUT_DIR, "plots")
UNIQUE_IDENTIFICATIONS_DIR = os.path.join(OUTPUT_DIR, "unique_identifications")

def create_output_directories():
    """
    Creates the necessary output directories if they don't exist.
    """
    directories = [
        PROCESSED_DATA_DIR,
        REPORTS_DIR,
        PLOTS_DIR,
        UNIQUE_IDENTIFICATIONS_DIR
    ]
    for directory in directories:
        os.makedirs(directory, exist_ok=True)

def hide_streamlit_style():
    """
    Hides Streamlit's default menu and footer for a cleaner interface.
    """
    hide_style = """
        <style>
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        </style>
        """
    st.markdown(hide_style, unsafe_allow_html=True)

def run_processing(save_type='csv', output_filename='Processed_Data.csv', file_path='Data.csv'):
    """
    Initializes and runs the data processor, saving outputs to the designated directories.
    """
    try:
        # Ensure output directories exist
        create_output_directories()
        
        # Define output file paths
        output_filepath = os.path.join(PROCESSED_DATA_DIR, output_filename)
        report_path = os.path.join(REPORTS_DIR, 'Type_Conversion_Report.csv')
        
        processor = DataProcessor(
            input_filepath=file_path,
            output_filepath=output_filepath,
            report_path=report_path,
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
        return output_filepath
    except Exception as e:
        st.error(f"Error during data processing: {e}")
        st.stop()

def load_data(file_type, uploaded_file):
    """
    Loads the uploaded file into a Pandas DataFrame without any processing.

    Parameters:
        file_type (str): Type of the uploaded file ('csv' or 'pkl').
        uploaded_file (UploadedFile): The uploaded file object.

    Returns:
        pd.DataFrame: The loaded DataFrame.
        str: Error message if any, else None.
    """
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

        # Read the data into a DataFrame
        if file_type == "pkl":
            Data = pd.read_pickle(temp_file_path)
        elif file_type == "csv":
            Data = pd.read_csv(temp_file_path)
        else:
            return None, "Unsupported file type!"

        # Clean up temporary file
        os.remove(temp_file_path)

        return Data, None
    except Exception as e:
        return None, f"Error loading data: {e}"

def align_dataframes(original_df, binned_df):
    """
    Ensures both DataFrames have the same columns.
    """
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

def save_dataframe(df, file_type, filename, subdirectory):
    """
    Saves the DataFrame to the specified file type within a subdirectory.
    """
    try:
        # Ensure output directories exist
        create_output_directories()
        
        # Determine the full path
        if subdirectory == "processed_data":
            dir_path = PROCESSED_DATA_DIR
        elif subdirectory == "reports":
            dir_path = REPORTS_DIR
        elif subdirectory == "unique_identifications":
            dir_path = UNIQUE_IDENTIFICATIONS_DIR
        elif subdirectory == "plots":
            dir_path = PLOTS_DIR
        else:
            raise ValueError("Unsupported subdirectory for saving DataFrame.")

        file_path = os.path.join(dir_path, filename)

        if file_type == 'csv':
            df.to_csv(file_path, index=False)
        elif file_type == 'pkl':
            df.to_pickle(file_path)
        else:
            raise ValueError("Unsupported file type for saving.")
        
        return file_path  # Return the path for further use if needed
    except Exception as e:
        st.error(f"Error saving DataFrame: {e}")
        st.stop()

def load_dataframe(file_path, file_type):
    """
    Loads a DataFrame from the specified file path and type.
    """
    try:
        if file_type == 'csv':
            return pd.read_csv(file_path)
        elif file_type == 'pkl':
            return pd.read_pickle(file_path)
        else:
            raise ValueError("Unsupported file type for loading.")
    except Exception as e:
        st.error(f"Error loading DataFrame: {e}")
        st.stop()

def get_binning_configuration(Data, selected_columns):
    """
    Generates binning configuration sliders for selected columns.

    Args:
        Data (pd.DataFrame): The DataFrame containing the data.
        selected_columns (list): List of columns selected for binning.

    Returns:
        dict: A dictionary with column names as keys and number of bins as values.
    """
    bins = {}
    st.markdown("### 📏 Binning Configuration")

    # Define layout parameters
    cols_per_row = 2  # Number of sliders per row
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

    return bins

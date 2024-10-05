# src/utils/utils_general.py

import streamlit as st
import os
from src.data_processing import DataProcessor
from src.config import (
    PROCESSED_DATA_DIR,
    REPORTS_DIR,
    CAT_MAPPING_DIR,
    DATA_DIR
)

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
        # Define output file paths
        output_filepath = os.path.join(PROCESSED_DATA_DIR, output_filename)
        report_path = os.path.join(REPORTS_DIR, 'Type_Conversion_Report.csv')
        
        processor = DataProcessor(
            input_filepath=os.path.join(DATA_DIR, file_path),
            output_filepath=output_filepath,
            report_path=report_path,
            return_category_mappings=True,
            mapping_directory=CAT_MAPPING_DIR,
            parallel_processing=False,  # Set to True if parallel processing is desired
            date_threshold=0.6,
            numeric_threshold=0.9,
            factor_threshold_ratio=0.4,
            factor_threshold_unique=1000,
            dayfirst=True,
            log_level='INFO',
            log_file=None,
            convert_factors_to_int=False,
            date_format=None,  # Keep as None to retain datetime dtype
            save_type=save_type
        )
        processed_data = processor.process()
        return processed_data
        
    except Exception as e:
        st.error(f"Error during data processing: {e}")
        st.stop()

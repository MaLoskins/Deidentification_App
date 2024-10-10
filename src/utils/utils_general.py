# src/utils/utils_general.py

import streamlit as st
import os
import matplotlib.pyplot as plt
from src.data_processing import DataProcessor
import pandas as pd
import numpy as np
from src.config import (
    PROCESSED_DATA_DIR,
    REPORTS_DIR,
    CAT_MAPPING_DIR,
    DATA_DIR,
    UNIQUE_IDENTIFICATIONS_DIR,
    PLOTS_DIR,
    LOGS_DIR
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

def save_dataframe(df, file_type, filename, subdirectory):
    """
    Saves the DataFrame or Figure to the specified file type within a subdirectory.
    """
    try:
        if subdirectory == "processed_data":
            dir_path = PROCESSED_DATA_DIR
        elif subdirectory == "reports":
            dir_path = REPORTS_DIR
        elif subdirectory == "unique_identifications":
            dir_path = UNIQUE_IDENTIFICATIONS_DIR
        elif subdirectory == "plots":
            dir_path = PLOTS_DIR
        else:
            raise ValueError("Unsupported subdirectory for saving.")

        os.makedirs(dir_path, exist_ok=True)  # Ensure directory exists
        file_path = os.path.join(dir_path, filename)
        if file_type == 'csv':
            df.to_csv(file_path, index=False)
        elif file_type == 'pkl':
            df.to_pickle(file_path)
        elif file_type == 'png':
            # Handle saving plots
            if isinstance(df, plt.Figure):
                df.savefig(file_path, bbox_inches='tight')
            else:
                raise ValueError("Unsupported data type for saving as PNG.")
        else:
            raise ValueError("Unsupported file type for saving.")

        return file_path
    except Exception as e:
        st.error(f"Error saving file `{filename}`: {e}")
        st.stop()

def initialize_session_state():
    """Initialize all necessary session state variables."""
    default_session_state = {
        # Original Data
        'UPLOADED_ORIGINAL_DATA': pd.DataFrame(),
        'ORIGINAL_DATA': pd.DataFrame(),
        'GLOBAL_DATA': pd.DataFrame(),
        
        # Binning Session States
        'Binning_Selected_Columns': [],
        'Binning_Method': 'Quantile',  # Default value
        'Binning_Configuration': {},
        
        # Location Granularizer Session States
        'Location_Selected_Columns': [],
        'geocoded_data': pd.DataFrame(),
        'geocoded_dict': {},
        
        # Unique Identification Analysis Session States
        'Unique_ID_Results': {},
        
        # Anonymization Session States
        'ANONYMIZED_DATA': pd.DataFrame(),
        'ANONYMIZATION_REPORT': pd.DataFrame(),
        
        # Progress Indicators
        'geocoding_progress': 0,
        'granular_location_progress': 0,
        
        # Flags for Processing Steps 
        'is_binning_done': False,
        'is_geocoding_done': False,
        'is_granular_location_done': False,
        'is_unique_id_done': False,

        # Logging
        'log_file': os.path.join(LOGS_DIR, 'app.log'),
        'session_state_logs': [],
        'show_logs': False,

        # Data Processing Settings (Newly Added)
        'date_threshold': 0.6,
        'numeric_threshold': 0.9,
        'factor_threshold_ratio': 0.4,
        'factor_threshold_unique': 1000,
        'dayfirst': True,
        'convert_factors_to_int': False,
        'date_format': None
    }
    
    for key, value in default_session_state.items():
        if key not in st.session_state:
            st.session_state[key] = value

def update_session_state(key: str, value):
    """
    Update a session state variable and log the update.

    Args:
        key (str): The key of the session state variable.
        value: The value to set for the session state variable.
    """
    st.session_state[key] = value
    log_message = f"🔄 **Session State Updated:** `{key}` has been set/updated."
    st.session_state['session_state_logs'].append(log_message)
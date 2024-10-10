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

help_info = {
    "sidebar_inputs": {
        "uploaded_file": "Upload your dataset in CSV or Pickle format. This is your primary data input.",
        "output_file_type": "Select the desired output file format for processed data: CSV or Pickle.",
        "binning_method": "Choose the binning method: 'Quantile' for equal-sized bins or 'Equal Width' for bins of equal range."
    },
    "binning_tab": {
        "selected_columns_binning": "Select the columns you wish to bin. This is required to perform manual binning.",
        "start_dynamic_binning": "Check this option to initiate the dynamic binning process.",
        "min_support": "Set the minimum support threshold for association rule mining. This controls the minimum frequency of itemsets.",
        "min_threshold": "Set the minimum confidence threshold for association rule mining. This determines the minimum confidence level for the rules generated."
    },
    "location_granulariser_tab": {
        "selected_geo_column": "Choose a column that contains geographical data to perform geocoding.",
        "granularity": "Select the level of granularity for location identification (e.g., address, city, state, etc.).",
        "start_geocoding": "Initiate the geocoding process to convert geographical locations into standardized formats.",
        "generate_granular_location": "Click to start the location granularization process.",
        "load_map_button": "Click to load the map with the geocoded data."
    },
    "unique_identification_analysis_tab": {
        "selected_columns_uniquetab": "Select columns to analyze for unique identification. The analysis reveals potential identifiers in the data.",
        "min_comb_size": "Specify the minimum size for combinations of columns to consider during the uniqueness analysis.",
        "max_comb_size": "Specify the maximum size for combinations of columns to consider during the uniqueness analysis."
    },
    "data_anonymization_tab": {
        "anonymization_method": "Select the method for data anonymization: k-anonymity, l-diversity, or t-closeness.",
        "quasi_identifiers": "Choose the columns to generalize for anonymity. These are the quasi-identifiers.",
        "sensitive_attribute": "Select a sensitive attribute to protect during the anonymization process (if applicable).",
        "max_iterations": "Set the maximum number of iterations for the anonymization process. This controls the complexity of generalization."
    },
    "synthetic_data_generation_tab": {
        "selected_columns": "Choose which columns from the original dataset will be included in the synthetic data generation.",
        "missing_value_strategy": "Select a strategy for handling missing values: Drop, Mean, Median, Mode, or Fill with a specific value.",
        "num_samples": "Specify the number of synthetic samples to generate from the model.",
        "method": "Select the synthetic data generation method: CTGAN or Gaussian Copula.",
        "ctgan_epochs": "Set the number of epochs for training the CTGAN model.",
        "ctgan_batch_size": "Specify the batch size for training the CTGAN model.",
        "generate_synthetic_data": "Click to start the synthetic data generation process.",
        "compare_distributions": "Select a column to compare the distribution of synthetic data against the original data."
    },
    "data_processing_settings": {
        "date_detection_threshold": "Set the threshold for date detection in the dataset.",
        "numeric_detection_threshold": "Set the threshold for numeric detection in the dataset.",
        "factor_threshold_ratio": "Adjust the factor threshold ratio for detecting categorical data.",
        "factor_threshold_unique": "Specify the minimum unique value count for factors.",
        "day_first": "Check this if dates are in day-first format.",
        "convert_factors_to_int": "Choose whether to convert categorical factors to integer type.",
        "date_format": "Specify the date format if applicable."
    },
    "about_application": "This application helps with data processing, anonymization, and synthetic data generation."
}


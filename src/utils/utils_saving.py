# src/utils/utils_saving.py

import streamlit as st
import pandas as pd
import os
from src.config import (
    PROCESSED_DATA_DIR,
    REPORTS_DIR,
    UNIQUE_IDENTIFICATIONS_DIR,
    PLOTS_DIR
)

def save_dataframe(df, file_type, filename, subdirectory):
    """
    Saves the DataFrame to the specified file type within a subdirectory.
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
            raise ValueError("Unsupported subdirectory for saving DataFrame.")

        file_path = os.path.join(dir_path, filename)
        if file_type == 'csv':
            df.to_csv(file_path, index=False)
        elif file_type == 'pkl':
            df.to_pickle(file_path)
        else:
            raise ValueError("Unsupported file type for saving.")
        
        return file_path
    except Exception as e:
        st.error(f"Error saving DataFrame: {e}")
        st.stop()

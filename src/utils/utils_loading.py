# src/utils/utils_loading.py

import streamlit as st
import pandas as pd
import tempfile
import os

def load_data(file_type, uploaded_file):
    """
    Loads the uploaded file into a Pandas DataFrame without any processing.
    """
    if uploaded_file is None:
        return None, "No file uploaded!"

    try:
        file_extension = {"pkl": "pkl", "csv": "csv"}.get(file_type, "csv")
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_extension}") as tmp_file:
            tmp_file.write(uploaded_file.getbuffer())
            temp_file_path = tmp_file.name

        if file_type == "pkl":
            Data = pd.read_pickle(temp_file_path)
        elif file_type == "csv":
            Data = pd.read_csv(temp_file_path)
        else:
            return None, "Unsupported file type!"

        os.remove(temp_file_path)
        return Data, None
    except Exception as e:
        return None, f"Error loading data: {e}"

def align_dataframes(original_df, binned_df):
    """
    Ensures both DataFrames have the same columns.
    """
    try:
        missing_in_binned = original_df.columns.difference(binned_df.columns)
        for column in missing_in_binned:
            binned_df[column] = original_df[column]
        binned_df = binned_df[original_df.columns]
        return original_df, binned_df
    except Exception as e:
        st.error(f"Error aligning dataframes: {e}")
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

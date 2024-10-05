# src/utils/utils_download.py

import streamlit as st
import pandas as pd
import traceback

def download_binned_data(data_full, data, file_type_download='csv'):
    """Handle downloading of the binned data."""
    if data is not None and isinstance(data, pd.DataFrame):
        # Add Streamlit option to select original or full data
        download_choice = st.radio(
            "Choose data to download:",
            options=["Only Binned Columns", "Full Data"],
            index=0,
            key='download_choice'
        )
        data_to_download = data_full if download_choice == "Full Data" else data
    else:
        data_to_download = None

    if data_to_download is not None:
        handle_download_binned_data(
            data=data_to_download,
            file_type_download=file_type_download
        )

def handle_download_binned_data(data, file_type_download='csv'):
    """
    Handles the download functionality for binned data.
    """
    st.markdown("### 💾 Download Binned Data")
    try:
        if file_type_download == 'csv':
            binned_csv = data.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Binned Data as CSV",
                data=binned_csv,
                file_name='binned_data.csv',
                mime='text/csv',
            )
        elif file_type_download == 'pkl':
            # Serialize DataFrame to pickle within this function
            binned_pkl = data.to_pickle(None)  # Using None returns bytes
            st.download_button(
                label="📥 Download Binned Data as Pickle",
                data=binned_pkl,
                file_name='binned_data.pkl',
                mime='application/octet-stream',
            )
    except Exception as e:
        st.error(f"Error during data download: {e}")
        st.error(traceback.format_exc())

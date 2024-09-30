# Application.py

import streamlit as st
import pandas as pd
from Binning_Tab.data_binner import DataBinner
from Binning_Tab.data_integrity_assessor import DataIntegrityAssessor
from Binning_Tab.unique_bin_identifier import UniqueBinIdentifier
import os
import traceback  # For detailed error logging

# Import utility functions for Binning
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
    get_binning_configuration,
    plot_entropy_and_display,
    plot_density_plots_and_display,
    handle_download_binned_data,
    handle_integrity_assessment,
    handle_unique_identification_analysis,
    display_unique_identification_results
)

# Import utility functions for Location Granulariser

from Location_Granulariser.geocoding import (
    extract_gpe_entities,
    interpret_location,
    geocode_location_with_cache,
    detect_geographical_columns,
    reverse_geocode_with_cache,
    perform_geocoding,
    generate_granular_location,
    prepare_map_data
)

def setup_page():
    """Configure the Streamlit page and apply custom styles."""
    st.set_page_config(
        page_title="🛠️ Data Processing and Binning Application",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    hide_streamlit_style()
    st.title('🛠️ Data Processing and Binning Application')

def sidebar_inputs():
    """Render the sidebar with file upload, settings, binning options, and info."""
    with st.sidebar:
        st.header("📂 Upload & Settings")
        uploaded_file = st.file_uploader("📤 Upload your dataset", type=['csv', 'pkl'])
        output_file_type = st.selectbox('📁 Select Output File Type', ['csv', 'pkl'], index=0)
        st.markdown("---")

        # Display warning if CSV is selected
        if output_file_type == 'csv':
            st.warning("⚠️ **Note:** Using CSV will result in the loss of some meta-data regarding data types. This will not affect the application's functionality.")

        st.header("⚙️ Binning Options")
        binning_method = st.selectbox('🔧 Select Binning Method', ['Quantile', 'Equal Width'])
        if binning_method == 'Equal Width':
            st.warning("⚠️ **Note:** Using Equal Width will drastically affect the distribution of your data. (Large integrity loss)")  
        
        st.markdown("---")

        st.header("ℹ️ About")
        st.info("""
            This application allows you to upload a dataset, process and bin numerical and datetime columns, 
            assess data integrity post-binning, visualize data distributions, and perform unique identification analysis.
        """)

    return uploaded_file, output_file_type, binning_method

def load_and_preview_data(uploaded_file, input_file_type):
    """Load the uploaded data and display a preview."""
    try:
        with st.spinner('Loading data...'):
            Data, error = load_data(input_file_type, uploaded_file)
        if error:
            st.error(error)
            st.stop()
    except Exception as e:
        st.error(f"Error loading data: {e}")
        st.error(traceback.format_exc())
        st.stop()

    # Clean column names by removing '\' and '/'
    Data.columns = Data.columns.str.replace(r'[\\/]', '', regex=True)

    st.session_state.Original_Data = Data.copy()
    st.session_state.Global_Data = Data.copy()  # Initialize Global_Data

    st.subheader('📊 Data Preview (Original Data)')
    st.dataframe(st.session_state.Original_Data.head())

def save_raw_data(Data, output_file_type):
    """Save the raw data to a CSV or Pickle file."""
    mapped_save_type = 'pickle' if output_file_type == 'pkl' else 'csv'
    data_csv_path = f'Data.{output_file_type}'
    try:
        if mapped_save_type == 'pickle':
            Data.to_pickle(data_csv_path)
        else:
            Data.to_csv(data_csv_path, index=False)
    except Exception as e:
        st.error(f"Error saving Data.{output_file_type}: {e}")
        st.stop()
    return mapped_save_type, data_csv_path

def run_data_processing(mapped_save_type, output_file_type, data_csv_path):
    """Run the data processing pipeline."""
    processed_data = run_processing(
        save_type=mapped_save_type,
        output_filename=f'processed_data.{output_file_type}',
        file_path=data_csv_path
    )

    st.session_state.Processed_Data = processed_data.copy()

def initialize_session_state():
    """Initialize session state variables if not already present."""
    if 'Original_Data' not in st.session_state:
        st.session_state.Original_Data = pd.DataFrame()
    if 'Global_Data' not in st.session_state:
        st.session_state.Global_Data = pd.DataFrame()
    if 'Binning_Selected_Columns' not in st.session_state:
        st.session_state.Binning_Selected_Columns = []
    if 'Location_Selected_Columns' not in st.session_state:
        st.session_state.Location_Selected_Columns = []
    if 'Granular_Location_Column_Set' not in st.session_state:
        st.session_state.Granular_Location_Column_Set = False # Different Method to Binning because of computation time
    if 'geocoded_data' not in st.session_state:
        st.session_state.geocoded_data = None
    if 'geocoded_dict' not in st.session_state:
        st.session_state.geocoded_dict = {}
    # Removed 'Manipulation_Selected_Columns' as it's no longer needed

def main():
    """Main function to orchestrate the Streamlit app."""
    setup_page()
    initialize_session_state()
    uploaded_file, output_file_type, binning_method = sidebar_inputs()

    if uploaded_file is not None:
        # Determine input file type
        if uploaded_file.name.endswith('.csv'):
            input_file_type = 'csv'
        elif uploaded_file.name.endswith('.pkl'):
            input_file_type = 'pkl'
        else:
            st.error("Unsupported file type! Please upload a CSV or Pickle (.pkl) file.")
            st.stop()

        load_and_preview_data(uploaded_file, input_file_type)
        mapped_save_type, data_csv_path = save_raw_data(st.session_state.Original_Data, output_file_type)
        run_data_processing(mapped_save_type, output_file_type, data_csv_path)
    else:
        st.info("🔄 **Please upload a file to get started.**")
        st.stop()

    # Create Tabs
    tabs = st.tabs(["📊 Binning", "📍 Location Data Geocoding Granulariser", "🔍 Unique Identification Analysis"])

    ######################
    # Binning Tab
    ######################
    with tabs[0]:
        st.header("📊 Binning")
        st.markdown("### 🔢 Select Columns to Bin")

        # Determine columns available for binning (no exclusion)
        available_columns = st.session_state.Processed_Data.select_dtypes(
            include=['number', 'datetime', 'datetime64[ns, UTC]', 'datetime64[ns]']
        ).columns.tolist()

        selected_columns_binning = st.multiselect(
            'Select columns to bin',
            options=available_columns,
            default=st.session_state.Binning_Selected_Columns,
            key='binning_columns'
        )
        st.session_state.Binning_Selected_Columns = selected_columns_binning

        if selected_columns_binning:
            Data_aligned, binned_df_aligned = perform_binning(
                st.session_state.Processed_Data,
                selected_columns_binning,
                binning_method
            )
            perform_integrity_assessment(Data_aligned, binned_df_aligned, selected_columns_binning)
            plot_density(
                Data_aligned[selected_columns_binning].astype('category'),
                binned_df_aligned[selected_columns_binning],
                selected_columns_binning
            )
            # Update Global_Data
            st.session_state.Global_Data = st.session_state.Original_Data.copy()
            for col in selected_columns_binning:
                st.session_state.Global_Data[col] = binned_df_aligned[col]

            st.subheader('📊 Data Preview (Global Data)')
            st.dataframe(st.session_state.Global_Data.head())

            download_binned_data()
        else:
            st.info("🔄 **Please select at least one column to bin.**")

    ######################
    # Location Granulariser Tab
    ######################
    with tabs[1]:
        st.header("📍 Location Data Geocoding Granulariser")
        st.markdown("### 🔢 Select Location Column to Granularise ")
        
        # Upload and display data
        data = st.session_state.Original_Data.copy()
        display_original_data(data)
        
        # Geocoding process
        geocoding_section(data)
        
        # Display geocoded data
        display_geocoded_data()
        
        # Granular location generation
        granular_location_section()
        
        # Display geocoded data with granular location
        display_geocoded_with_granular_data()
        
        # Map display
        if st.session_state.geocoded_data is not None:
            map_section()

    ######################
    # Unique Identification Analysis Tab
    ######################
    with tabs[2]:
        st.header("🔍 Unique Identification Analysis")
        st.markdown("### 🔢 Selected Columns for Analysis")

        if st.session_state.Binning_Selected_Columns is not None and st.session_state.Location_Selected_Columns is not None and st.session_state.Granular_Location_Column_Set:
            selected_columns = st.session_state.Binning_Selected_Columns + st.session_state.Location_Selected_Columns
        elif st.session_state.Binning_Selected_Columns is not None:
            selected_columns = st.session_state.Binning_Selected_Columns
        elif st.session_state.Location_Selected_Columns is not None and st.session_state.Granular_Location_Column_Set:
            selected_columns = st.session_state.Location_Selected_Columns
        else:
            selected_columns = None


    #display global data if available
    if st.session_state.Global_Data is not None:
        st.subheader('📊 Data Preview (Global Data)')
        st.dataframe(st.session_state.Global_Data.head())

        # Use only columns selected in Binning tab

        if not selected_columns:
            st.warning("⚠️ **No columns selected in Binning tab for analysis.**")
            st.info("🔄 **Please select columns in the Binning tab to perform Unique Identification Analysis.**")
        else:
            st.write(f"**Columns selected for analysis:** {', '.join(selected_columns)}")

            # Format selected columns as categorical
            analysis_df = st.session_state.Global_Data[selected_columns].astype('category')

            unique_identification_section(
                original_for_assessment=st.session_state.Original_Data[selected_columns].astype('category'),
                binned_for_assessment=analysis_df,
                selected_columns=selected_columns
            )


######################
# Binning Functions
######################

def perform_binning(processed_data, selected_columns_binning, binning_method):
    """Perform the binning process on selected columns."""
    try:
        bins = get_binning_configuration(processed_data, selected_columns_binning)
    except Exception as e:
        st.error(f"Error in binning configuration: {e}")
        st.stop()

    st.markdown("### 🔄 Binning Process")
    try:
        with st.spinner('Binning data...'):
            binner = DataBinner(processed_data, method=binning_method.lower())
            binned_df, binned_columns = binner.bin_columns(bins)

            # Align both DataFrames (original and binned) to have the same columns
            Data_aligned, binned_df_aligned = align_dataframes(processed_data, binned_df)
    except Exception as e:
        st.error(f"Error during binning: {e}")
        st.error(traceback.format_exc())
        st.stop()

    st.success("✅ Binning completed successfully!")

    # Display binned columns categorization
    st.markdown("### 🗂️ Binned Columns Categorization")
    for dtype, cols in binned_columns.items():
        if cols:
            st.write(f"  - **{dtype.capitalize()}**: {', '.join(cols)}")

    return Data_aligned, binned_df_aligned

def perform_integrity_assessment(Data_aligned, binned_df_aligned, selected_columns_binning):
    """Assess data integrity after binning."""
    original_for_assessment = Data_aligned[selected_columns_binning].astype('category')
    binned_for_assessment = binned_df_aligned[selected_columns_binning]

    handle_integrity_assessment(original_for_assessment, binned_for_assessment, PLOTS_DIR)

def plot_density(original_for_assessment, binned_for_assessment, selected_columns_binning):
    """Plot and display density plots."""
    plot_density_plots_and_display(original_for_assessment, binned_for_assessment, selected_columns_binning, PLOTS_DIR)

def download_binned_data():
    """Handle downloading of the binned data."""
    handle_download_binned_data(
        data=st.session_state.Global_Data,
        file_type_download=st.selectbox('📁 Select Download File Type', ['csv', 'pkl'], index=0, key='download_file_type_download'),
        save_dataframe_func=save_dataframe,
        plots_dir=PLOTS_DIR
    )



######################
# Unique Identification Functions
######################

def unique_identification_section(original_for_assessment, binned_for_assessment, selected_columns):
    """Handle the Unique Identification Analysis section."""
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
        # Define bin columns to consider (use selected columns)

        col_count = len(selected_columns)
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
        results = handle_unique_identification_analysis(
            original_df=original_for_assessment,
            binned_df=binned_for_assessment,
            bin_columns_list=selected_columns,
            min_comb_size=min_comb_size,
            max_comb_size=max_comb_size
        )
        display_unique_identification_results(results)


######################
# Location Granulariser Functions
######################

def display_original_data(data):
    """Display the original uploaded data."""
    st.subheader("🔍 Original Data")
    st.dataframe(data)

def geocoding_section(data):
    """Handle the geocoding process."""
    st.markdown("---")
    st.header("1️⃣ Geocoding")
    
    with st.expander("⚙️ Geocoding Options", expanded=True):
        columns_location = detect_geographical_columns(data)
        
        # Remove all columns from data that are not in columns
        data = data[[col for col in data.columns if col in columns_location]]
        
        if not columns_location:
            st.warning("No columns detected that likely contain geographical data. Try uploading a different file or renaming location columns.")
            st.stop()
        
        selected_columns_location = st.multiselect(
            "Select columns to geocode",
            options=columns_location,
            help="Choose the columns containing location data to geocode."
        )

        st.session_state.Location_Selected_Columns = selected_columns_location
        
        # Handle concatenation if exactly two columns are selected
        if len(selected_columns_location) == 2:
            combined_column_name = ' '.join(selected_columns_location)
            if combined_column_name not in data.columns:
                data[combined_column_name] = data[selected_columns_location[0]].fillna('').astype(str) + ' ' + data[selected_columns_location[1]].fillna('').astype(str)
            selected_columns_location = [combined_column_name]
            st.warning(
                f"⚠️ **Concatenation Alert:** You have selected two columns. "
                f"The application will concatenate them into a new column '{combined_column_name}' "
                f"for geocoding purposes instead of processing them individually."
            )
        
        preprocess_button = st.button("📂 Start Geocoding")
        
        if preprocess_button:
            if not selected_columns_location:
                st.warning("Please select at least one column to geocode.")
            else:
                try:
                    with st.spinner("Processing..."):
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        geocoded_df = perform_geocoding(
                            data=data,
                            selected_columns=selected_columns_location,
                            session_state=st.session_state,
                            progress_bar=progress_bar,
                            status_text=status_text
                        )
                        st.session_state.geocoded_data = geocoded_df
                    st.success("✅ Geocoding completed!")
                except ValueError as ve:
                    st.warning(str(ve))
                except Exception as e:
                    st.error(f"❌ An unexpected error occurred during geocoding: {e}")
                    st.stop()

def display_geocoded_data():
    """Display the geocoded data and provide download options."""
    if st.session_state.geocoded_data is not None:
        st.subheader("📝 Geocoded Data")
        st.dataframe(st.session_state.geocoded_data)
        
        st.download_button(
            label="💾 Download Geocoded Data",
            data=st.session_state.geocoded_data.to_csv(index=False).encode('utf-8'),
            file_name="geocoded_data.csv",
            mime="text/csv"
        )
    else:
        st.info("👉 Please perform geocoding first.")

def granular_location_section():
    """Handle the granular location generation process."""
    st.markdown("---")
    st.header("2️⃣ Granular Location Generation")
    
    with st.expander("⚙️ Granularization Options", expanded=True):
        granularity_options = ["address", "suburb", "city", "state", "country", "continent"]
        granularity = st.selectbox(
            "Select Location Granularity",
            options=granularity_options,
            help="Choose the level of granularity for location identification."
        )
        
        generate_granular_button = st.button("📈 Generate Granular Location Column")
        
        if generate_granular_button:
            if st.session_state.geocoded_data is None:
                st.warning("Please perform geocoding first.")
            else:
                try:
                    with st.spinner("Generating granular location data..."):
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        granular_df, column_name = generate_granular_location(
                            data=st.session_state.geocoded_data,
                            granularity=granularity,
                            session_state=st.session_state,
                            progress_bar=progress_bar,
                            status_text=status_text
                        )
                    
                    st.session_state.geocoded_data = granular_df
                    st.success(f"✅ Granular location column generated successfully!")
                    st.session_state.Global_Data[st.session_state.Location_Selected_Columns[0]] = st.session_state.geocoded_data[column_name]
                    st.session_state.Granular_Location_Column_Set = True
                except ValueError as ve:
                    st.warning(str(ve))
                except Exception as e:
                    st.error(f"❌ An unexpected error occurred during granular location generation: {e}")
                    st.stop()

def display_geocoded_with_granular_data():
    """Display geocoded data with granular location and provide download options."""
    if st.session_state.geocoded_data is not None:
        if any(col.startswith('Granular Location') for col in st.session_state.geocoded_data.columns):
            st.subheader("📝 Geocoded Data with Granular Location")
            st.dataframe(st.session_state.geocoded_data)
            
            st.download_button(
                label="💾 Download Data with Granular Location",
                data=st.session_state.geocoded_data.to_csv(index=False).encode('utf-8'),
                file_name="geocoded_data_with_granularity.csv",
                mime="text/csv"
            )
        else:
            st.info("👉 Granular location data not available. Please generate it first.")
    else:
        st.info("👉 Please perform geocoding and granular location generation first.")

def map_section():
    """Handle the map display functionality."""
    st.markdown("---")
    
    try:
        map_data = prepare_map_data(st.session_state.geocoded_data)
        load_map_button = st.button("🗺️ Load Map")
        
        if load_map_button:
            st.map(map_data[['lat', 'lon']], use_container_width=True, zoom=2)
    except ValueError as ve:
        st.info(str(ve))
    except Exception as e:
        st.error(f"❌ An unexpected error occurred while preparing the map: {e}")
        

if __name__ == "__main__":
    main()

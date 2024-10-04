# application.py

import os
import traceback
import pandas as pd
import streamlit as st
# Import binning modules
from src.binning import DataBinner
from src.binning import KAnonymityBinner

# Import utility functions
from src.utils import (
    hide_streamlit_style,
    load_data,
    align_dataframes,
    save_dataframe,
    run_processing,
    get_binning_configuration,
    plot_entropy_and_display,
    plot_density_plots_and_display,
    handle_download_binned_data,
    handle_download_k_binned_data,
    handle_integrity_assessment,
    handle_unique_identification_analysis,
    display_unique_identification_results
)

# Import path configurations
from src.config import (
    PLOTS_DIR,
    PROCESSED_DATA_DIR,
    REPORTS_DIR,
    UNIQUE_IDENTIFICATIONS_DIR,
    CAT_MAPPING_DIR,
    DATA_DIR,
    LOGS_DIR
)

# Import location granularizer functions
from src.location_granularizer import (
    extract_gpe_entities,
    interpret_location,
    geocode_location_with_cache,
    detect_geographical_columns,
    reverse_geocode_with_cache,
    perform_geocoding,
    generate_granular_location,
    prepare_map_data
)

# =====================================
# Helper Functions for Session State
# =====================================

def initialize_session_state():
    """Initialize all necessary session state variables."""
    default_session_state = {
        # Original and Processed Data
        'ORIGINAL_DATA': pd.DataFrame(),
        'Processed_Data': pd.DataFrame(),
        'GLOBAL_DATA': pd.DataFrame(),
        
        # Binning Session States
        'Binning_Selected_Columns': [],
        'Binning_Method': 'Quantile',  # Default value
        'Binning_Configuration': {},
        
        # Location Granularizer Session States
        'Location_Selected_Columns': [],
        'Granular_Location_Column_Set': False,
        'geocoded_data': pd.DataFrame(),
        'geocoded_dict': {},
        'Granular_Location_Configurations': {},
        
        # Unique Identification Analysis Session States
        'Unique_ID_Results': {},
        
        # Progress Indicators
        'geocoding_progress': 0,
        'granular_location_progress': 0,
        
        # Flags for Processing Steps
        'is_binning_done': False,
        'is_geocoding_done': False,
        'is_granular_location_done': False,
        'is_unique_id_done': False
    }
    
    for key, value in default_session_state.items():
        if key not in st.session_state:
            st.session_state[key] = value
    
    # Initialize the session_state_logs if not present
    if 'session_state_logs' not in st.session_state:
        st.session_state['session_state_logs'] = []

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

# =====================================
# Page Configuration and Sidebar
# =====================================

def setup_page():
    """Configure the Streamlit page and apply custom styles."""
    st.set_page_config(
        page_title="🛠️ De-Identification of Privileged Data (Generalisation Methodology)",
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
        
        # ============================
        # Special Section: Session State Info
        # ============================
        with st.expander("🔍 Session State Info"):
            st.markdown("### 📝 Session State Update Logs")
            if st.session_state['session_state_logs']:
                for log in st.session_state['session_state_logs']:
                    st.markdown(log)
            else:
                st.write("No session state updates yet.")
            
            st.markdown("### 📊 Session State Variable Types")
            session_info = []
            for key, value in st.session_state.items():
                var_type = type(value).__name__
                dtypes = ""
                if isinstance(value, pd.DataFrame):
                    dtypes = ', '.join([f"{col}: {dtype}" for col, dtype in value.dtypes.items()])
                elif isinstance(value, pd.Series):
                    dtypes = f"{value.name}: {value.dtype}"
                session_info.append({
                    "Key": key,
                    "Type": var_type,
                    "Dtypes": dtypes if dtypes else "-"
                })
            df_session_info = pd.DataFrame(session_info)
            st.dataframe(df_session_info)
        
    return uploaded_file, output_file_type, binning_method

# =====================================
# Data Loading and Saving
# =====================================

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

    update_session_state('ORIGINAL_DATA', Data.copy())
    st.subheader('📊 Data Preview (Original Data)')
    st.dataframe(st.session_state.ORIGINAL_DATA.head())


def save_raw_data(Data, output_file_type):
    """Save the raw data to a CSV or Pickle file."""
    mapped_save_type = 'pickle' if output_file_type == 'pkl' else 'csv'
    data_path = os.path.join(DATA_DIR, f'Data.{output_file_type}')
    try:
        if mapped_save_type == 'pickle':
            Data.to_pickle(data_path)
        else:
            Data.to_csv(data_path, index=False)
    except Exception as e:
        st.error(f"Error saving Data.{output_file_type}: {e}")
        st.stop()
    return mapped_save_type, data_path


def run_data_processing(mapped_save_type, output_file_type, file_path):
    """Run the data processing pipeline."""
    processed_data = run_processing(
        save_type=mapped_save_type,
        output_filename=f'processed_data.{output_file_type}',
        file_path=os.path.join(DATA_DIR, file_path)
    )

    update_session_state('Processed_Data', processed_data.copy())

# =====================================
# Binning Tab Functionality
# =====================================

def perform_binning(processed_data, selected_columns_binning, binning_method, bins):
    """Perform the binning process on selected columns."""
    st.markdown("### 🔄 Binning Process")
    try:
        with st.spinner('Binning data...'):
            binner = DataBinner(processed_data, method=binning_method.lower())
            binned_df, binned_columns = binner.bin_columns(bins)

            # Align both DataFrames (original and binned) to have the same columns
            OG_Data_BinTab, Data_BinTab = align_dataframes(processed_data, binned_df)

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

    return OG_Data_BinTab, Data_BinTab


def perform_integrity_assessment(OG_Data_BinTab, Data_BinTab, selected_columns_binning):
    """Assess data integrity after binning."""
    original_for_assessment = OG_Data_BinTab[selected_columns_binning].astype('category')
    data_for_assessment = Data_BinTab[selected_columns_binning]

    handle_integrity_assessment(original_for_assessment, data_for_assessment, PLOTS_DIR)


def download_binned_data(data_full, data):
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
            file_type_download=st.selectbox(
                '📁 Download Format', 
                ['csv', 'pkl'], 
                index=0, 
                key='download_file_type_download'
            ),
            save_dataframe_func=save_dataframe
        )


def binning_tab():
    """Render the Binning Tab in the Streamlit app."""
    st.header("📊 Manual Binning")
    
    processed_data = st.session_state.Processed_Data
    location_selected = set(st.session_state.Location_Selected_Columns)
    
    # Determine available columns by excluding those selected in Location Granulariser
    available_columns = list(
        set(processed_data.select_dtypes(
            include=['number', 'datetime', 'datetime64[ns, UTC]', 'datetime64[ns]']
        ).columns) - location_selected
    )
    
    # Multiselect widget for selecting columns to bin
    selected_columns_binning = st.multiselect(
        'Select columns to bin',
        options=available_columns,
        default=st.session_state.Binning_Selected_Columns,
        key='binning_columns_form'
    )
    update_session_state('Binning_Selected_Columns', selected_columns_binning)

    # Configure bins if columns are selected
    if selected_columns_binning:
        bins = get_binning_configuration(processed_data, selected_columns_binning)
        update_session_state('Binning_Configuration', bins)
    else:
        st.info("🔄 **Please select at least one column to bin.**")

    # Proceed to binning if bins are configured
    bins = st.session_state.get('Binning_Configuration')
    
    if bins and selected_columns_binning:
        try:
            # Perform binning operation
            OG_Data_BinTab, Data_BinTab = perform_binning(
                processed_data,
                selected_columns_binning,
                st.session_state.Binning_Method,
                bins
            )

            # Assess data integrity post-binning
            perform_integrity_assessment(OG_Data_BinTab, Data_BinTab, selected_columns_binning)
            
            # Plot density distributions for binned data
            plot_density_plots_and_display(
                OG_Data_BinTab[selected_columns_binning].astype('category'), 
                Data_BinTab[selected_columns_binning], 
                selected_columns_binning, 
                PLOTS_DIR
            )
            
            # Update GLOBAL_DATA with the binned columns
            st.session_state.GLOBAL_DATA[selected_columns_binning] = Data_BinTab[selected_columns_binning]
            update_session_state('GLOBAL_DATA', st.session_state.GLOBAL_DATA)

            # Mark binning as completed
            update_session_state('is_binning_done', True)
            st.success("✅ Binning completed successfully!")

            # Provide option to download the binned data
            download_binned_data(Data_BinTab, Data_BinTab[selected_columns_binning])

        except Exception as e:
            st.error(f"Error during binning: {e}")
    elif selected_columns_binning:
        st.info("👉 Adjust the bins using the sliders above to run binning.")

# =====================================
# Location Granulariser Tab Functionality
# =====================================

def setup_geocoding_options_ui(geocoded_data: pd.DataFrame) -> list:
    """Render the UI for selecting columns to geocode."""
    selected_geo_columns = detect_geographical_columns(geocoded_data)

    if not selected_geo_columns:
        st.warning("No columns detected that likely contain geographical data. Try uploading a different file or renaming location columns.")
        st.stop()
    
    selected_geo_columns = st.multiselect(
        "Select columns to geocode",
        options=selected_geo_columns,
        help="Choose the columns containing location data to geocode.",
        max_selections=1
    )

    return selected_geo_columns


def perform_geocoding_process(selected_geo_columns, geocoded_data):
    """Perform geocoding on the selected columns."""
    if not selected_geo_columns:
        st.warning("Please select at least one column to geocode.")
        return
    else:
        try:
            with st.spinner("Processing..."):
                progress_bar = st.progress(0)
                status_text = st.empty()
                geocoded_data = perform_geocoding(
                    data=geocoded_data,
                    selected_geo_columns=selected_geo_columns,
                    session_state=st.session_state,
                    progress_bar=progress_bar,
                    status_text=status_text
                )
            update_session_state('geocoded_data', geocoded_data)
            update_session_state('is_geocoding_done', True)
            st.success("✅ Geocoding completed!")
        except ValueError as ve:
            st.warning(str(ve))
        except Exception as e:
            st.error(f"❌ An unexpected error occurred during geocoding: {e}")
            st.error(traceback.format_exc())
            st.stop()


def perform_granular_location_generation(granularity, selected_geo_columns):
    """Perform granular location generation."""
    if st.session_state.geocoded_data.empty:
        st.warning("Please perform geocoding first.")
        return
    else:
        st.write(st.session_state.geocoded_data.shape)
        column_name = selected_geo_columns[0]
        try:
            with st.spinner("Generating granular location data..."):
                progress_bar = st.progress(0)
                status_text = st.empty()
                granular_df = generate_granular_location(
                    data=st.session_state.geocoded_data,
                    granularity=granularity,
                    session_state=st.session_state,
                    progress_bar=progress_bar,
                    status_text=status_text,
                    column=column_name
                )
            
            # Update GLOBAL_DATA with the new granular location
            st.write(granular_df.shape)
            st.session_state.GLOBAL_DATA[column_name] = granular_df[column_name]
            update_session_state('GLOBAL_DATA', st.session_state.GLOBAL_DATA)

            # Track the addition of the new column for future analysis
            if column_name not in st.session_state.Location_Selected_Columns:
                st.session_state.Location_Selected_Columns.append(column_name)
                update_session_state('Location_Selected_Columns', st.session_state.Location_Selected_Columns)
                st.success(f"🔄 '{column_name}' added to Location_Selected_Columns list.")
            else:
                st.info(f"ℹ️ '{column_name}' is already in Location_Selected_Columns list.")

            update_session_state('is_granular_location_done', True)
            st.dataframe(granular_df.head())
            
        except Exception as e:
            st.error(f"❌ Error during granular location generation: {e}")
            st.error(traceback.format_exc())


def location_granulariser_tab():
    """Render the Location Granulariser Tab in the Streamlit app."""
    st.header("📍 Location Data Geocoding Granulariser")
    
    # Geocoding process
    st.header("1️⃣ Geocoding")

    geocoded_data = st.session_state.geocoded_data.copy() if not st.session_state.geocoded_data.empty else st.session_state.ORIGINAL_DATA.copy()
    st.dataframe(geocoded_data.head())

    selected_geo_columns = setup_geocoding_options_ui(geocoded_data)

    preprocess_button = st.button("📂 Start Geocoding")
    if preprocess_button:
        perform_geocoding_process(selected_geo_columns, geocoded_data)

    # Granular location generation
    st.header("2️⃣ Granular Location Generation")

    granularity_options = ["address", "suburb", "city", "state", "country", "continent"]
    granularity = st.selectbox(
        "Select Location Granularity",
        options=granularity_options,
        help="Choose the level of granularity for location identification."
    )
    generate_granular_button = st.button("📈 Generate Granular Location Column")
    if generate_granular_button:
        perform_granular_location_generation(granularity, selected_geo_columns)
        update_session_state('Location_Selected_Columns', selected_geo_columns.copy())

    # Display geocoded data with granular location
    display_geocoded_with_granular_data()
    
    # Map display
    if not st.session_state.geocoded_data.empty:
        map_section()


def display_geocoded_with_granular_data():
    """Display geocoded data with granular location and provide download options."""
    if not st.session_state.geocoded_data.empty:
        # Check if any granular location columns exist in GLOBAL_DATA
        granular_columns_present = [col for col in st.session_state.Location_Selected_Columns if col in st.session_state.GLOBAL_DATA.columns]
        if granular_columns_present:
            # output categories in the granular location column
            categories = st.session_state.GLOBAL_DATA[granular_columns_present].apply(lambda x: x.astype('category').cat.categories)

            # Display the DataFrame of categories
            st.subheader("📝 Categories in Granular Location Column")
            st.dataframe(categories)
            
            st.download_button(
                label="💾 Download Data with Granular Location",
                data=st.session_state.GLOBAL_DATA[granular_columns_present].to_csv(index=False).encode('utf-8'),
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
        st.error(traceback.format_exc())

# =====================================
# Unique Identification Analysis Tab Functionality
# =====================================

def unique_identification_section_ui(selected_columns_uniquetab):
    """Render the UI for Unique Identification Analysis."""
    st.markdown("### 🔍 Unique Identification Analysis")
    with st.expander("ℹ️ **About:**"):
        st.write("""
            This section analyzes combinations of binned columns and granular location columns to determine 
            how many unique observations can be identified by each combination of these features.
        """)

    # Use a form to group inputs and button together
    with st.form("unique_id_form"):
    
        st.write("#### 🧮 Configure Unique Identification Analysis")

        col_count = len(selected_columns_uniquetab)
        col1, col2 = st.columns(2)
        with col1:
            min_comb_size = st.number_input('Minimum Combination Size', min_value=1, max_value=col_count, value=1, step=1)
        with col2:
            max_comb_size = st.number_input('Maximum Combination Size', min_value=min_comb_size, max_value=col_count, value=col_count, step=1)

        if max_comb_size > 5:
            st.warning("⚠️  **Note:** Combinations larger than 5 may take a long time to compute depending on bin count.")

        # Submit button
        submit_button = st.form_submit_button(label='🧮 Perform Unique Identification Analysis')

    return min_comb_size, max_comb_size, submit_button


def perform_unique_identification_analysis(original_for_assessment, data_for_assessment, selected_columns_uniquetab, min_comb_size, max_comb_size):
    """Handle the Unique Identification Analysis process."""
    try:
        results = handle_unique_identification_analysis(
            original_df=original_for_assessment,
            binned_df=data_for_assessment,
            columns_list=selected_columns_uniquetab,
            min_comb_size=min_comb_size,
            max_comb_size=max_comb_size
        )
        update_session_state('Unique_ID_Results', results)
        display_unique_identification_results(results)
        update_session_state('is_unique_id_done', True)
    except Exception as e:
        st.error(f"❌ Error during unique identification analysis: {e}")
        st.error(traceback.format_exc())


def unique_identification_analysis_tab():
    """Render the Unique Identification Analysis Tab in the Streamlit app."""
    st.header("🔍 Unique Identification Analysis")
    st.markdown("### 🔢 Selected Columns for Analysis")

    selected_columns_uniquetab = []

    # Combine selections from Binning and Location Granulariser
    if st.session_state.Binning_Selected_Columns and st.session_state.Location_Selected_Columns:
        granular_columns = st.session_state.Location_Selected_Columns
        selected_columns_uniquetab = st.session_state.Binning_Selected_Columns + granular_columns
        st.write(f"🔍 **Selected Columns for Analysis:** {selected_columns_uniquetab}")
    elif st.session_state.Binning_Selected_Columns:
        selected_columns_uniquetab = st.session_state.Binning_Selected_Columns
        st.write(f"🔍 **Selected Columns for Analysis:** {selected_columns_uniquetab}")
    elif st.session_state.Location_Selected_Columns:
        granular_columns = st.session_state.Location_Selected_Columns
        selected_columns_uniquetab = granular_columns
        st.write(f"🔍 **Selected Columns for Analysis:** {selected_columns_uniquetab}")
    else:
        selected_columns_uniquetab = None
        st.info("👉 **Please use the Binning and/or Location Granulariser tabs to select columns for analysis.**")

    # Display global data if available
    if not st.session_state.GLOBAL_DATA.empty:
        st.subheader('📊 Data Preview (Global Data)')
        st.dataframe(st.session_state.GLOBAL_DATA.head())

        if not selected_columns_uniquetab:
            st.warning("⚠️ **No columns selected in Binning or Location Granulariser tabs for analysis.**")
            st.info("🔄 **Please select columns in the Binning tab and/or generate granular location data to perform Unique Identification Analysis.**")
        else:
            # Verify that selected_columns_uniquetab exist in both ORIGINAL_DATA and GLOBAL_DATA
            existing_columns = [col for col in selected_columns_uniquetab if col in st.session_state.GLOBAL_DATA.columns and col in st.session_state.ORIGINAL_DATA.columns]
            missing_in_global = [col for col in selected_columns_uniquetab if col not in st.session_state.GLOBAL_DATA.columns]
            missing_in_original = [col for col in selected_columns_uniquetab if col not in st.session_state.ORIGINAL_DATA.columns]

            if missing_in_global or missing_in_original:
                if missing_in_global: st.error(f"The following selected columns are missing from Global Data: {', '.join(missing_in_global)}. Please check your selections.")
                if missing_in_original: st.error(f"The following selected columns are missing from Original Data: {', '.join(missing_in_original)}. Please check your selections.")
                st.stop()

            # Proceed with the unique identification analysis
            original_for_assessment = st.session_state.Processed_Data[existing_columns].copy()
            data_for_assessment = st.session_state.GLOBAL_DATA[existing_columns].copy()

            min_comb_size, max_comb_size, submit_button = unique_identification_section_ui(selected_columns_uniquetab)

            if submit_button:
                perform_unique_identification_analysis(
                    original_for_assessment=original_for_assessment.astype('category'),
                    data_for_assessment=data_for_assessment.astype('category'),
                    selected_columns_uniquetab=existing_columns,
                    min_comb_size=min_comb_size,
                    max_comb_size=max_comb_size
                )

                # Perform integrity assessment
                perform_integrity_assessment(original_for_assessment, data_for_assessment, existing_columns)

                # Plot density distributions
                plot_density_plots_and_display(
                    original_for_assessment,
                    data_for_assessment,
                    existing_columns,
                    PLOTS_DIR
                )
    else:
        st.info("🔄 **Please upload and process data to perform Unique Identification Analysis.**")

# =====================================
# K-Anonymity Binning Tab Functionality
# =====================================

def download_k_anon_binned_data(data_full, data):
    """Handle downloading of the binned data."""
    if data is not None and isinstance(data, pd.DataFrame):
        # Add Streamlit option to select original or full data
        k_download_choice = st.radio(
            "Choose data to download:",
            options=["Only Binned Columns", "Full Data"],
            index=0,
            key='k_anon_download_choice'
        )
        k_data_to_download = data_full if k_download_choice == "Full Data" else data
    else:
        k_data_to_download = None

    if k_data_to_download is not None:
        handle_download_k_binned_data(
            data=k_data_to_download,
            file_type_download=st.selectbox(
                '📁 Download Format', 
                ['csv', 'pkl'], 
                index=0, 
                key='k_anon_download_file_type_download'
            ),
            save_dataframe_func=save_dataframe
        )

def k_anonymity_binning_tab():

    """Render the K-Anonymity Binning Tab in the Streamlit app."""
    st.header("🔒 K-Anonymity Binning")

    processed_data = st.session_state.Processed_Data

    # Select columns to include in k-anonymity binning
    available_columns = processed_data.select_dtypes(
        include=['number', 'datetime', 'datetime64[ns, UTC]', 'datetime64[ns]']
    ).columns.tolist()

    selected_columns_k_anonymity = st.multiselect(
        'Select columns for k-anonymity binning',
        options=available_columns,
        default=[],
        key='k_anonymity_columns_form'
    )
    update_session_state('K_Anonymity_Selected_Columns', selected_columns_k_anonymity)

    if selected_columns_k_anonymity:
        # Input for the desired k value
        k_value = st.number_input(
            'Set the value of k for k-anonymity',
            min_value=2,
            value=5,
            step=1,
            key='k_value_input'
        )
        update_session_state('K_Value', k_value)

        if st.button("Start K-Anonymity Binning"):
            try:
                with st.spinner('Performing k-anonymity binning...'):
                    # Initialize KAnonymityBinner and perform binning
                    binner = KAnonymityBinner(
                        data=processed_data[selected_columns_k_anonymity],
                        k=k_value,
                        method=st.session_state.Binning_Method.lower()
                    )
                    binned_data = binner.perform_binning()

                    # Update GLOBAL_DATA with the binned columns
                    st.session_state.GLOBAL_DATA[selected_columns_k_anonymity] = binned_data[selected_columns_k_anonymity]
                    update_session_state('GLOBAL_DATA', st.session_state.GLOBAL_DATA)

                    st.success("✅ K-Anonymity Binning completed successfully!")
                    st.dataframe(binned_data.head())

                    # Provide option to download the k-anonymity binned data
                    download_k_anon_binned_data(binned_data, binned_data[selected_columns_k_anonymity])

                    # Update GLOBAL_DATA with the binned columns
                    st.session_state.GLOBAL_DATA[selected_columns_k_anonymity] = binned_data[selected_columns_k_anonymity]
                    update_session_state('GLOBAL_DATA', st.session_state.GLOBAL_DATA)

            except Exception as e:
                st.error(f"Error during k-anonymity binning: {e}")
                st.error(traceback.format_exc())
    else:
        st.info("🔄 **Please select at least one column for k-anonymity binning.**")

# =====================================
# Main Function
# =====================================

def main():
    """Main function to orchestrate the Streamlit app."""
    setup_page()
    initialize_session_state()
    uploaded_file, output_file_type, binning_method = sidebar_inputs()

    # Update binning method in session state
    update_session_state('Binning_Method', binning_method)

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
        mapped_save_type, file_path = save_raw_data(st.session_state.ORIGINAL_DATA, output_file_type)
        run_data_processing(mapped_save_type, output_file_type, file_path)
        
        # Reset processing flags upon new upload
        processing_flags = ['is_binning_done', 'is_geocoding_done', 'is_granular_location_done', 'is_unique_id_done']
        for flag in processing_flags:
            update_session_state(flag, False)
    else:
        st.info("🔄 **Please upload a file to get started.**")
        st.stop()

    # Create Tabs
    tabs = st.tabs([
        "🔒 K-Anonymity Binning",  
        "📊 Manual Binning", 
        "📍 Location Data Geocoding Granulariser", 
        "🔍 Unique Identification Analysis"
    ])
    
    ######################
    # K-Anonymity Binning Tab
    ######################
    with tabs[0]:
        k_anonymity_binning_tab()

    ######################
    # Binning Tab
    ######################
    with tabs[1]:
        binning_tab()

    ######################
    # Location Granulariser Tab
    ######################
    with tabs[2]:
        location_granulariser_tab()

    ######################
    # Unique Identification Analysis Tab
    ######################
    with tabs[3]:
        unique_identification_analysis_tab()

# =====================================
# Entry Point
# =====================================

if __name__ == "__main__":
    main()

# application.py

import os
import sys
import traceback
import pandas as pd
import streamlit as st
from src.utils import (
    hide_streamlit_style,
    load_data,
    align_dataframes,
    get_binning_configuration,
    download_binned_data,
    perform_binning,
    perform_integrity_assessment,
    perform_association_rule_mining,
    perform_unique_identification_analysis,
    plot_density_plots_streamlit,
    binning_summary,
    compare_correlations,
    plot_distributions,
    save_dataframe,
    initialize_session_state,
    update_session_state,
    help_info  
)
from src.config import (
    PROCESSED_DATA_DIR,
    REPORTS_DIR,
    CAT_MAPPING_DIR,
    DATA_DIR,
    LOGS_DIR
)

# Import location granularizer functions
from src.location_granularizer import (
    detect_geographical_columns,
    perform_geocoding,
    generate_granular_location,
    prepare_map_data
)

# Import SyntheticDataGenerator Class
from src.synthetic_data_generator import SyntheticDataGenerator

# Import Data Processing Class
from src.data_processing import DataProcessor

# =====================================
# Page Configuration and Sidebar
# =====================================

def setup_page():
    """Configure the Streamlit page and apply custom styles."""
    st.set_page_config(
        page_title="🛠️ Dynamic De-Identification",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    hide_streamlit_style()
    st.title('🛠️ Dynamic De-Identification')

def display_logs():
    """Display application logs within the Streamlit interface."""
    st.header("📜 Application Logs")
    log_file = st.session_state.get('log_file', os.path.join(LOGS_DIR, 'app.log'))
    
    if os.path.exists(log_file):
        with open(log_file, 'r') as f:
            logs = f.read()
        st.text_area("🔍 Logs", logs, height=300)
    else:
        st.write("No logs available.")

def sidebar_inputs():
    """Render the sidebar with file upload, settings, binning options, and info."""
    with st.sidebar:
        st.header("📂 Upload & Settings")
        uploaded_file = st.file_uploader("📤 Upload your dataset", type=['csv', 'pkl'], help=help_info['sidebar_inputs']['uploaded_file'])
        
        output_file_type = st.selectbox(
            '📁 Select Output File Type', 
            ['csv', 'pkl'], 
            index=0, 
            help=help_info['sidebar_inputs']['output_file_type']
        )
        st.markdown("---")

        st.header("⚙️ Binning Options")
        binning_method = st.selectbox(
            '🔧 Select Binning Method', 
            ['Quantile', 'Equal Width'],
            help=help_info['sidebar_inputs']['binning_method']
        )
        if binning_method == 'Equal Width':
            st.warning("⚠️ **Note:** Using Equal Width will drastically affect the distribution of your data. (Large integrity loss)")

        st.header("ℹ️ About")
        st.info(help_info['about_application'])

        st.markdown("---")

        st.session_state.show_logs = st.checkbox(
            "🖥️ Show Logs in Interface", 
            value=False, 
            help="Display application logs within the app interface."
        )
        
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

@st.cache_data
def run_processing_cached(
    save_type='csv',
    output_filename='Processed_Data.csv',
    file_path='Data.csv',
    date_threshold=0.6,
    numeric_threshold=0.9,
    factor_threshold_ratio=0.4,
    factor_threshold_unique=500,
    dayfirst=False,
    convert_factors_to_int=False,
    date_format=None
):
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
            date_threshold=date_threshold,
            numeric_threshold=numeric_threshold,
            factor_threshold_ratio=factor_threshold_ratio,
            factor_threshold_unique=factor_threshold_unique,
            dayfirst=dayfirst,
            log_level='INFO',
            log_file=None,
            convert_factors_to_int=convert_factors_to_int,
            date_format=date_format,  # Keep as None to retain datetime dtype
            save_type=save_type
        )
        # Button to process the data
        processed_data = processor.process()
        return processed_data
        
    except Exception as e:
        st.error(f"Error during data processing: {e}")
        st.stop()

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

    update_session_state('UPLOADED_ORIGINAL_DATA', Data.copy())

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

# =====================================
# Binning Tab Functionality
# =====================================

def binning_tab():
    """Render the Binning Tab in the Streamlit app."""
    st.header("📊 Manual Binning")
    original_data = st.session_state.ORIGINAL_DATA.copy()
    
    # Determine available columns by excluding those selected in Location Granulariser
    available_columns = list(set(original_data.select_dtypes(
        include=['number', 'datetime', 'datetime64[ns, UTC]', 'datetime64[ns]', 'category']
    ).columns))
    
    # Multiselect widget for selecting columns to bin
    selected_columns_binning = st.multiselect(
        'Select columns to bin',
        options=available_columns,
        default=st.session_state.Binning_Selected_Columns,
        key='binning_columns_form',
        help=help_info['binning_tab']['selected_columns_binning']
    )
    update_session_state('Binning_Selected_Columns', selected_columns_binning)

    bins = None
    # Button to start binning
    
    if selected_columns_binning:
        bins = get_binning_configuration(original_data, selected_columns_binning)
    else:
        st.info(help_info['binning_tab']['select_columns_binning'])

    # Button to process binning
    st.markdown("---")
    switch_state = st.checkbox(
        'Start Dynamic Binning', 
        key='binning_switch', 
        help=help_info['binning_tab']['start_dynamic_binning']
    )

    if bins and selected_columns_binning and switch_state:
        try:
            # Perform binning operation
            original_data, binned_df, binned_columns = perform_binning(
                original_data,
                st.session_state.Binning_Method,
                bins
            )

            binning_summary(binned_df, binned_columns, bins)
            # Align both DataFrames (original and binned) to have the same columns
            OG_Data_BinTab, Data_BinTab = align_dataframes(original_data, binned_df)
            
            st.write("---")

            if st.button("📄 Run Integrity Report"):
                # Assess data integrity post-binning
                report, overall_loss, entropy_fig = perform_integrity_assessment(
                    OG_Data_BinTab,
                    Data_BinTab,
                    selected_columns_binning
                )
                
                if report is not None:
                    # Save the integrity report
                    integrity_report_bintab_path = save_dataframe(report, 'csv', 'Integrity_Loss_Report.csv', 'reports')
                    
                    # Display the integrity report
                    st.markdown("### 📄 Integrity Loss Report")
                    st.dataframe(report)
                    st.write(f"📊 **Overall Average Integrity Loss:** {overall_loss:.2f}%")
                    
                    # Plot and display entropy
                    if entropy_fig:
                        # Save entropy plot
                        entropy_plot_path = save_dataframe(entropy_fig, 'png', 'entropy_plot.png', 'plots')
                        
                        # Display entropy plot
                        st.pyplot(entropy_fig)
            st.write("---")
            # Add association rule mining parameters
            with st.expander("🔍 Association Rule Mining Settings"):
                min_support = st.slider("Minimum Support", 0.01, 1.0, 0.05, 0.01, help=help_info['binning_tab']['min_support'])
                min_threshold = st.slider("Minimum Confidence Threshold", 0.01, 1.0, 0.05, 0.01, help=help_info['binning_tab']['min_threshold'])
            # **Add a button to run Association Rule Mining**
            if st.button("🔍 Run Association Rule Mining"):
                try:
                    perform_association_rule_mining(
                        OG_Data_BinTab, 
                        Data_BinTab, 
                        selected_columns_binning,
                        min_support=min_support,
                        min_threshold=min_threshold
                    )
                    st.success("✅ Association Rule Mining completed successfully!")
                except Exception as e:
                    st.error(f"Error during Association Rule Mining: {e}")
            st.write("---")   
            # Optionally, if you also want to control the density plots with a button:
            if st.button("📈 Plot Density Plots"):
                try:
                    plot_density_plots_streamlit(OG_Data_BinTab, Data_BinTab, selected_columns_binning)
                    st.success("✅ Density plots generated successfully!")
                except Exception as e:
                    st.error(f"Error while plotting density plots: {e}")
            
            # Update GLOBAL_DATA with the binned columns
            st.session_state.GLOBAL_DATA[selected_columns_binning] = Data_BinTab[selected_columns_binning]
            update_session_state('GLOBAL_DATA', st.session_state.GLOBAL_DATA)

            # Mark binning as completed
            update_session_state('is_binning_done', True)

            st.write("---")
            # Provide option to download the binned data
            download_binned_data(Data_BinTab, Data_BinTab[selected_columns_binning])

        except Exception as e:
            st.error(f"Error during binning: {e}")

# =====================================
# Location Granulariser Tab Functionality
# =====================================

def location_granulariser_tab():
    """Render the Location Granulariser Tab in the Streamlit app."""
    st.header("📍 Location Data Geocoding Granulariser")

    # Geocoding process
    st.header("1️⃣ Geocoding")

    geocoded_data = st.session_state.geocoded_data.copy() if not st.session_state.geocoded_data.empty else st.session_state.ORIGINAL_DATA.copy()
    st.dataframe(geocoded_data.head())

    selected_geo_columns = setup_geocoding_options_ui(geocoded_data)

    if not selected_geo_columns:
        st.info("No geographical columns available for geocoding.")
        return  # Exit the function early

    preprocess_button = st.button(
        "📂 Start Geocoding", 
        help=help_info['location_granulariser_tab']['start_geocoding']
    )
    if preprocess_button:
        perform_geocoding_process(selected_geo_columns, geocoded_data)

    # Granular location generation
    st.header("2️⃣ Granular Location Generation")

    granularity_options = ["address", "suburb", "city", "state", "country", "continent"]
    granularity = st.selectbox(
        "Select Location Granularity",
        options=granularity_options,
        help=help_info['location_granulariser_tab']['granularity']
    )
    generate_granular_button = st.button(
        "📈 Generate Granular Location Column",
        help=help_info['location_granulariser_tab']['generate_granular_location']
    )
    if generate_granular_button:
        perform_granular_location_generation(granularity, selected_geo_columns)
        update_session_state('Location_Selected_Columns', selected_geo_columns.copy())

    # Display geocoded data with granular location
    display_geocoded_with_granular_data()
    
    # Map display
    if not st.session_state.geocoded_data.empty:
        map_section()

def setup_geocoding_options_ui(geocoded_data: pd.DataFrame) -> list:
    """Render the UI for selecting a single column to geocode."""
    detected_geo_columns = detect_geographical_columns(geocoded_data)

    if not detected_geo_columns:
        st.warning("No columns detected that likely contain geographical data. Try uploading a different file or renaming location columns.")
        return []  # Return an empty list

    selected_geo_column = st.selectbox(
        "Select a column to geocode",
        options=detected_geo_columns,
        help=help_info['location_granulariser_tab']['selected_geo_column']
    )

    return [selected_geo_column]  # Return as a list for consistency

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
    elif not selected_geo_columns:
        st.warning("No geographical columns selected for granular location generation.")
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

            # Convert only the granular location column to 'category'
            st.session_state.GLOBAL_DATA[column_name] = st.session_state.GLOBAL_DATA[column_name].astype('category')

            update_session_state('is_granular_location_done', True)
            st.dataframe(granular_df.head())
            
        except Exception as e:
            st.error(f"❌ Error during granular location generation: {e}")
            st.error(traceback.format_exc())

def display_geocoded_with_granular_data():
    """Display geocoded data with granular location and provide download options."""
    if not st.session_state.geocoded_data.empty:
        # Check if any granular location columns exist in GLOBAL_DATA
        granular_columns_present = [col for col in st.session_state.Location_Selected_Columns if col in st.session_state.GLOBAL_DATA.columns]
        if granular_columns_present:
            # Output categories in the granular location column
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
        
        # Debugging: Display data types
        st.write("Data types in map_data:")
        st.write(map_data.dtypes)
        
        # Load Map Button
        load_map_button = st.button("🗺️ Load Map", help=help_info['location_granulariser_tab']['load_map_button'])
        
        if load_map_button:
            # Ensure 'lat' and 'lon' are float
            map_data['lat'] = pd.to_numeric(map_data['lat'], errors='coerce')
            map_data['lon'] = pd.to_numeric(map_data['lon'], errors='coerce')
            map_data = map_data.dropna(subset=['lat', 'lon'])
            
            st.map(map_data[['lat', 'lon']], use_container_width=True, zoom=2)
    except ValueError as ve:
        st.info(str(ve))
    except Exception as e:
        st.error(f"❌ An unexpected error occurred while preparing the map: {e}")
        st.error(traceback.format_exc())

# =====================================
# Unique Identification Analysis Tab Functionality
# =====================================

def unique_identification_analysis_tab():
    """Render the Unique Identification Analysis Tab in the Streamlit app."""
    st.header("🔍 Unique Identification Analysis")
    st.markdown("### 🔢 Selected Columns for Analysis")

    selected_columns_uniquetab = []

    # Combine selections from Binning and Location Granulariser
    if st.session_state.Binning_Selected_Columns and st.session_state.Location_Selected_Columns:
        granular_columns = st.session_state.Location_Selected_Columns
        selected_columns_uniquetab = st.session_state.Binning_Selected_Columns + granular_columns
    elif st.session_state.Binning_Selected_Columns:
        selected_columns_uniquetab = st.session_state.Binning_Selected_Columns
    elif st.session_state.Location_Selected_Columns:
        granular_columns = st.session_state.Location_Selected_Columns
        selected_columns_uniquetab = granular_columns
    else:
        selected_columns_uniquetab = None
        st.info(help_info['unique_identification_analysis_tab']['use_bins_location'])

    if selected_columns_uniquetab:
        columns_and_info = pd.DataFrame(selected_columns_uniquetab, columns=['Selected Columns'])
        for col in selected_columns_uniquetab:
            columns_and_info.loc[columns_and_info['Selected Columns'] == col, 'Original Unique Count'] = st.session_state.ORIGINAL_DATA[col].nunique()
            columns_and_info.loc[columns_and_info['Selected Columns'] == col, 'Granular Unique Count'] = st.session_state.GLOBAL_DATA[col].nunique()

        # Transpose and set first row as header
        columns_and_info = columns_and_info.set_index('Selected Columns').T
        
        st.dataframe(columns_and_info, width=1600, height=100)

    # Display global data if available
    if not st.session_state.GLOBAL_DATA.empty:
        st.subheader('📊 Data Preview (Global Data)')
        st.dataframe(st.session_state.GLOBAL_DATA.head())

        if not selected_columns_uniquetab:
            st.warning("⚠️ **No columns selected in Binning or Location Granulariser tabs for analysis.**")
            st.info(help_info['unique_identification_analysis_tab']['select_columns_unique_analysis'])
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
            original_for_assessment = st.session_state.ORIGINAL_DATA[existing_columns].astype('category').copy()
            data_for_assessment = st.session_state.GLOBAL_DATA[existing_columns].copy()

            min_comb_size, max_comb_size, submit_button = unique_identification_section_ui(selected_columns_uniquetab)

            if submit_button:
                results = perform_unique_identification_analysis(
                    original_for_assessment=original_for_assessment,
                    data_for_assessment=data_for_assessment,
                    selected_columns_uniquetab=existing_columns,
                    min_comb_size=min_comb_size,
                    max_comb_size=max_comb_size
                )
                if results is not None:
                    update_session_state('Unique_ID_Results', results)
                    
                    # Display Unique Identification Results
                    st.markdown("### 📊 Unique Identification Results")
                    st.dataframe(results)
                    
                    # Provide Download Option for Unique Identification Results
                    csv = results.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="📥 Download Unique Identification Results as CSV",
                        data=csv,
                        file_name='unique_identifications.csv',
                        mime='text/csv',
                    )
                    
                    # Save the Unique Identification Results
                    unique_id_path = save_dataframe(results, 'csv', 'unique_identifications.csv', 'unique_identifications')
                    
                    update_session_state('is_unique_id_done', True)
                
                # Perform integrity assessment
                report, overall_loss, entropy_fig = perform_integrity_assessment(original_for_assessment, data_for_assessment, existing_columns)
                if report is not None:
                    # Save the integrity report
                    integrity_report_Identification_path = save_dataframe(report, 'csv', 'Integrity_Loss_Report_Unique_ID.csv', 'reports')
                    
                    # Display the integrity report
                    st.markdown("### 📄 Integrity Loss Report for Unique Identification Analysis")
                    st.dataframe(report)
                    st.write(f"📊 **Overall Average Integrity Loss:** {overall_loss:.2f}%")
                    
                    # Plot and display entropy
                    if entropy_fig:
                        # Save entropy plot
                        entropy_plot_path = save_dataframe(entropy_fig, 'png', 'entropy_plot_unique_id.png', 'plots')
                        
                        # Display entropy plot
                        st.pyplot(entropy_fig)

                # Plot density distributions
                plot_density_plots_streamlit(
                    original_for_assessment,
                    data_for_assessment,
                    existing_columns
                )
                

def unique_identification_section_ui(selected_columns_uniquetab):
    """Render the UI for Unique Identification Analysis."""

    # Use a form to group inputs and button together
    with st.form("unique_id_form"):
    
        st.write("#### 🧮 Configure Unique Identification Analysis")

        col_count = len(selected_columns_uniquetab)
        col1, col2 = st.columns(2)
        with col1:
            min_comb_size = st.number_input(
                'Minimum Combination Size', 
                min_value=1, 
                max_value=col_count, 
                value=1, 
                step=1,
                help=help_info['unique_identification_analysis_tab']['min_comb_size']
            )
        with col2:
            max_comb_size = st.number_input(
                'Maximum Combination Size', 
                min_value=min_comb_size, 
                max_value=col_count, 
                value=col_count, 
                step=1,
                help=help_info['unique_identification_analysis_tab']['max_comb_size']
            )

        if max_comb_size > 5:
            st.warning("⚠️  **Note:** Combinations larger than 5 may take a long time to compute depending on bin count.")

        # Submit button
        submit_button = st.form_submit_button(label='🧮 Perform Unique Identification Analysis')

    return min_comb_size, max_comb_size, submit_button

# =====================================
# Data Anonymization Tab Functionality
# =====================================

# Add necessary imports at the top of your application.py
import os
import sys  # Add this import if not already present
import traceback
from src.binning_optimizer import BinningOptimizer
from src.utils import (
    save_dataframe,
    plot_fitness_history,
    plot_time_taken,
    plot_comparative_distributions
)
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
import logging

# Configure the logger
logging.basicConfig(
    level=logging.INFO,
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('application.log', encoding='utf-8')  # Use utf-8 encoding
    ],
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)

def data_anonymization_tab():
    """Render the Data Anonymization Tab in the Streamlit app."""
    st.header("🔐 Data Anonymization")

    # Check if ORIGINAL_DATA is available
    if 'ORIGINAL_DATA' not in st.session_state or st.session_state.ORIGINAL_DATA.empty:
        st.warning("⚠️ **No original data available. Please upload a dataset first.**")
        return

    original_data = st.session_state.ORIGINAL_DATA.copy()

    st.subheader("⚙️ Binning Optimizer")

    # Select Privacy Model
    privacy_model = st.selectbox(
        "Select Privacy Model",
        options=["k-anonymity", "l-diversity", "t-closeness"],
        help="Choose the privacy model to enforce during binning."
    )

    # Initialize parameters based on privacy model
    with st.expander("🔧 Configure Parameters", expanded=True):
        st.markdown("### 🛠️ Configure Privacy Model Parameters")

        # Common parameter: k-anonymity level
        k = st.number_input(
            "k (k-anonymity level)",
            min_value=1,
            value=2,
            step=1,
            help="Desired k-anonymity level."
        )
        if k < 1:
            st.error("❌ 'k' must be at least 1.")
            return

        # Initialize variables for l-diversity and t-closeness
        l = None
        t = None
        sensitive_attributes = None

        if privacy_model == "l-diversity":
            l = st.number_input(
                "l (l-diversity level)",
                min_value=1,
                value=2,
                step=1,
                help="Desired l-diversity level."
            )
            if l < 1:
                st.error("❌ 'l' must be at least 1.")
                return
            sensitive_attributes = st.multiselect(
                "Select Sensitive Attributes",
                options=original_data.columns.tolist(),
                help="Columns containing sensitive information."
            )
            if not sensitive_attributes:
                st.warning("⚠️ Please select at least one sensitive attribute for l-diversity.")

        elif privacy_model == "t-closeness":
            t = st.number_input(
                "t (t-closeness threshold)",
                min_value=0.0,
                max_value=1.0,
                value=0.05,
                step=0.01,
                help="Desired t-closeness threshold."
            )
            if t <= 0 or t > 1:
                st.error("❌ 't' must be between 0 and 1.")
                return
            sensitive_attributes = st.multiselect(
                "Select Sensitive Attributes",
                options=original_data.columns.tolist(),
                help="Columns containing sensitive information."
            )
            if not sensitive_attributes:
                st.warning("⚠️ Please select at least one sensitive attribute for t-closeness.")

        # Binning Configuration
        st.markdown("---")
        st.markdown("### 🔢 Binning Configuration")

        # Select columns to bin
        columns_to_bin = st.multiselect(
            "Select Columns to Bin",
            options=original_data.columns.tolist(),
            default=original_data.columns.tolist(),
            help="Columns to include in the binning process."
        )
        if not columns_to_bin:
            st.warning("⚠️ Please select at least one column to bin.")
            return

        # Define minimum and maximum bins per column
        st.markdown("#### 🔽 Define Minimum and Maximum Bins per Column")
        min_bins_per_column = {}
        max_bins_per_column = {}
        for col in columns_to_bin:
            col_min_bins, col_max_bins = st.columns(2)
            with col_min_bins:
                min_bins = st.number_input(
                    f"Min Bins for '{col}'",
                    min_value=1,
                    max_value=100,
                    value=3,
                    step=1,
                    key=f"min_bins_{col}",
                    help=f"Minimum number of bins for column '{col}'."
                )
            with col_max_bins:
                max_bins = st.number_input(
                    f"Max Bins for '{col}'",
                    min_value=min_bins,  # Ensure max_bins >= min_bins
                    max_value=100,
                    value=20,
                    step=1,
                    key=f"max_bins_{col}",
                    help=f"Maximum number of bins for column '{col}'."
                )
            min_bins_per_column[col] = min_bins
            max_bins_per_column[col] = max_bins

        # Select binning method and optimizer
        st.markdown("---")
        st.markdown("### 🔧 Select Binning Method and Optimizer")

        binning_method = st.selectbox(
            "Binning Method",
            options=["quantile", "equal width"],
            index=0,
            help="Choose the binning method: Quantile ensures equal-sized bins, while Equal Width divides data into bins of equal range."
        )

        optimizer = st.selectbox(
            "Optimization Method",
            options=["genetic", "simulated_annealing"],
            index=0,
            help="Choose the optimization method: Genetic Algorithm or Simulated Annealing."
        )

        st.markdown("### 🛠️ Optimization Hyperparameters")

        # Optimizer-specific hyperparameters
        if optimizer == "genetic":
            generations = st.number_input(
                "Generations",
                min_value=1,
                value=100,
                step=1,
                help="Number of generations for the Genetic Algorithm."
            )
            population_size = st.number_input(
                "Population Size",
                min_value=10,
                value=50,
                step=1,
                help="Number of individuals in each generation."
            )
            mutation_rate = st.slider(
                "Mutation Rate",
                min_value=0.0,
                max_value=1.0,
                value=0.1,
                step=0.01,
                help="Probability of mutation in the Genetic Algorithm."
            )
        elif optimizer == "simulated_annealing":
            initial_temperature = st.number_input(
                "Initial Temperature",
                min_value=0.1,
                value=1000.0,
                step=100.0,
                help="Starting temperature for Simulated Annealing."
            )
            cooling_rate = st.slider(
                "Cooling Rate",
                min_value=0.0,
                max_value=1.0,
                value=0.95,
                step=0.01,
                help="Rate at which the temperature decreases."
            )
            iterations = st.number_input(
                "Iterations",
                min_value=1,
                value=1000,
                step=1,
                help="Number of iterations for Simulated Annealing."
            )
            neighbors_per_iteration = st.number_input(
                "Neighbors per Iteration",
                min_value=1,
                value=5,
                step=1,
                help="Number of neighbor solutions to evaluate each iteration."
            )

        # Expose max_iterations to the user
        max_iterations = st.number_input(
            "Max Iterations",
            min_value=1,
            value=1000,
            step=1,
            help="Maximum number of iterations for random sampling or optimization algorithms."
        )

        # Define maximum combination size with a dynamic default value
        desired_default = 3
        max_val = len(columns_to_bin)
        default_value = desired_default if max_val >= desired_default else max_val

        max_comb_size = st.number_input(
            "Maximum Combination Size",
            min_value=1,
            max_value=max_val,
            value=default_value,
            step=1,
            help="Maximum number of columns to consider in combinations for anonymization."
        )


        # Allow user to adjust max_workers
        max_workers = st.number_input(
            "Max Workers",
            min_value=1,
            max_value=os.cpu_count() or 1,
            value=min(8, os.cpu_count() or 1),
            step=1,
            help="Maximum number of worker threads for parallel processing."
        )

    # Run Optimizer Button
    st.markdown("---")
    if st.button("🛠️ Optimize Binning"):
        if not columns_to_bin:
            st.error("❌ Please select at least one column to bin.")
        elif privacy_model in ["l-diversity", "t-closeness"] and not sensitive_attributes:
            st.error("❌ Please select at least one sensitive attribute for the chosen privacy model.")
        else:
            progress_bar = st.progress(0)  # Initialize progress bar
            progress_text = st.empty()      # Placeholder for progress text
            optimization_logs = st.expander("📄 Optimization Logs", expanded=False)
            optimization_logs.markdown("### 📝 Logs:")
            optimization_log_text = optimization_logs.empty()

            # Reset previous session state
            st.session_state['Privacy_Achieved'] = False
            st.session_state['Optimization_Logs'] = ""

            def progress_callback(progress_percentage: int, log_message: str = ""):
                progress_bar.progress(progress_percentage)
                progress_text.text(f"Progress: {progress_percentage}%")
                if log_message:
                    optimization_log_text.text(log_message)

            with st.spinner("Running Binning Optimizer..."):
                try:
                    # Initialize the BinningOptimizer
                    binning_optimizer = BinningOptimizer(
                        original_data=original_data,
                        k=int(k),
                        privacy_model=privacy_model.replace('-', '_'),
                        sensitive_attributes=sensitive_attributes if privacy_model in ["l-diversity", "t-closeness"] else None,
                        l=int(l) if privacy_model == "l-diversity" else None,
                        t=float(t) if privacy_model == "t-closeness" else None,
                        min_comb_size=1,
                        max_comb_size=int(max_comb_size),
                        columns=columns_to_bin,
                        min_bins_per_column=min_bins_per_column,
                        max_bins_per_column=max_bins_per_column,
                        max_iterations=int(max_iterations),
                        optimizer=optimizer,
                        method=binning_method,
                        logger=logger,
                        # Optimizer-specific hyperparameters
                        generations=int(generations) if optimizer == "genetic" else None,
                        population_size=int(population_size) if optimizer == "genetic" else None,
                        mutation_rate=float(mutation_rate) if optimizer == "genetic" else None,
                        initial_temperature=float(initial_temperature) if optimizer == "simulated_annealing" else None,
                        cooling_rate=float(cooling_rate) if optimizer == "simulated_annealing" else None,
                        iterations=int(iterations) if optimizer == "simulated_annealing" else None,
                        neighbors_per_iteration=int(neighbors_per_iteration) if optimizer == "simulated_annealing" else None,
                        max_workers=int(max_workers)
                    )

                    # Execute the optimization
                    with st.spinner("Finding the best binning configuration..."):
                        best_bin_dict, best_binned_df = binning_optimizer.find_best_binned_data(progress_callback=progress_callback)

                    if best_bin_dict and best_binned_df is not None and not best_binned_df.empty:
                        progress_bar.progress(100)  # Ensure progress bar is complete
                        progress_text.text("Progress: 100%")
                        st.success("✅ Binning Optimization Completed Successfully!")

                        # Perform detailed privacy checks
                        privacy_achieved, privacy_details = binning_optimizer.check_privacy(best_binned_df)
                        st.session_state['Privacy_Achieved'] = privacy_achieved

                        # Display Privacy Achievement Status
                        if privacy_achieved:
                            st.success("🎉 Desired privacy level was achieved.")
                        else:
                            st.error("⚠️ Desired privacy level was NOT achieved.")

                        # Display Best Binning Configuration
                        st.markdown("### 🎯 Best Binning Configuration:")
                        bin_config_df = pd.DataFrame(list(best_bin_dict.items()), columns=["Column", "Number of Bins"])
                        st.dataframe(bin_config_df)

                        # Display Binned Data Sample
                        st.markdown("### 📊 Binned Data Sample:")
                        st.dataframe(best_binned_df.head())

                        # Display Optimization Summary
                        summary = binning_optimizer.get_optimization_summary()
                        summary['Privacy Achieved'] = "Yes" if privacy_achieved else "No"
                        st.markdown("### 📈 Optimization Summary:")
                        summary_df = pd.DataFrame(list(summary.items()), columns=["Metric", "Value"])
                        st.dataframe(summary_df)

                        # Plot Fitness History using utils_plotting.py
                        st.markdown("### 📉 Fitness Over Iterations:")
                        fig_fitness = plot_fitness_history(binning_optimizer.fitness_history, title="Fitness Over Iterations")
                        st.pyplot(fig_fitness)

                        # Plot Time Taken per Iteration using utils_plotting.py
                        st.markdown("### ⏱️ Time Taken per Iteration:")
                        fig_time = plot_time_taken(binning_optimizer.times, title="Time Taken per Iteration")
                        st.pyplot(fig_time)

                        # Plot Comparative Distributions between Original and Binned Data
                        st.markdown("### 📊 Comparative Distributions:")
                        comparison_fig = plot_comparative_distributions(original_data, best_binned_df, columns_to_bin)
                        st.pyplot(comparison_fig)

                        # Plot Privacy Compliance Visualizations
                        if privacy_model == "k-anonymity":
                            st.markdown("### 🔒 K-Anonymity Compliance:")
                            fig_k_anonymity = binning_optimizer.plot_k_anonymity_compliance()
                            st.pyplot(fig_k_anonymity)
                        elif privacy_model == "l-diversity":
                            st.markdown("### 🔒 L-Diversity Compliance:")
                            fig_l_diversity = binning_optimizer.plot_l_diversity_compliance()
                            st.pyplot(fig_l_diversity)
                        elif privacy_model == "t-closeness":
                            st.markdown("### 🔒 T-Closeness Compliance:")
                            fig_t_closeness = binning_optimizer.plot_t_closeness_compliance()
                            st.pyplot(fig_t_closeness)

                        # Option to Download Binned Data
                        st.markdown("### 💾 Download Results:")
                        csv_binned = best_binned_df.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            label="📥 Download Binned Data as CSV",
                            data=csv_binned,
                            file_name='binned_data.csv',
                            mime='text/csv',
                            key='download_binned_data'
                        )

                        # Option to Download Binning Configuration
                        csv_bin_config = bin_config_df.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            label="📥 Download Binning Configuration as CSV",
                            data=csv_bin_config,
                            file_name='binning_configuration.csv',
                            mime='text/csv',
                            key='download_binning_config'
                        )

                        # Optionally, Download Privacy Compliance Summary
                        if privacy_achieved:
                            privacy_summary = {
                                "Privacy Achieved": "Yes",
                                **privacy_details
                            }
                        else:
                            privacy_summary = {
                                "Privacy Achieved": "No",
                                **privacy_details
                            }
                        privacy_summary_df = pd.DataFrame(list(privacy_summary.items()), columns=["Metric", "Value"])
                        csv_privacy = privacy_summary_df.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            label="📥 Download Privacy Compliance Summary as CSV",
                            data=csv_privacy,
                            file_name='privacy_compliance_summary.csv',
                            mime='text/csv',
                            key='download_privacy_summary'
                        )

                        # Store the binned data and configurations in session state for further use if needed
                        st.session_state['Binned_Data'] = best_binned_df.copy()
                        st.session_state['Binning_Configuration'] = best_bin_dict.copy()

                        # Store privacy details in session state
                        st.session_state['Privacy_Details'] = privacy_details

                        # Provide Recommendations if Privacy Not Achieved
                        if not privacy_achieved:
                            st.markdown("### 🛠️ Recommendations:")
                            recommendations = binning_optimizer.get_privacy_recommendations()
                            for rec in recommendations:
                                st.write(f"- {rec}")

                        # Option to Retry Optimization
                        if not privacy_achieved:
                            if st.button("🔄 Retry Optimization"):
                                # Reset relevant session state variables
                                st.session_state['Privacy_Achieved'] = False
                                st.session_state['Optimization_Logs'] = ""
                                # Re-run the optimizer with the same or adjusted parameters
                                st.experimental_rerun()

                        # Display Comprehensive Optimization Logs
                        st.markdown("### 📄 Comprehensive Optimization Logs:")
                        with st.expander("🔍 View Logs", expanded=False):
                            st.text_area("Logs", value=binning_optimizer.get_logs(), height=300)

                    else:
                        st.error("❌ Binning Optimization failed to find a suitable configuration.")
                        st.session_state['Privacy_Achieved'] = False
                        st.session_state['Optimization_Logs'] += "❌ Binning Optimization failed to find a suitable configuration.\n"

                except Exception as e:
                    st.error(f"❌ An error occurred during binning optimization: {e}")
                    st.error(traceback.format_exc())
                    logger.error("An error occurred during binning optimization.", exc_info=True)




# =====================================
# Synthetic Data Generation Tab Functionality
# =====================================

import streamlit as st
import pandas as pd
import logging
import traceback

# Import the updated SyntheticDataGenerator class
from src.synthetic_data_generator import SyntheticDataGenerator
from src.utils.utils_plotting import plot_distributions, compare_correlations

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def synthetic_data_generation_tab():
    """Render the Synthetic Data Generation Tab in the Streamlit app."""
    st.header("🧪 Synthetic Data Generation")

    # Check if ORIGINAL_DATA is available in the session state
    if 'ORIGINAL_DATA' not in st.session_state or st.session_state.ORIGINAL_DATA.empty:
        st.warning("⚠️ **No original data available. Please upload a dataset first.**")
        return

    original_data = st.session_state.ORIGINAL_DATA.copy()

    # Select columns to use
    st.subheader("🔢 Select Columns for Synthetic Data Generation")
    selected_columns = st.multiselect(
        "Select columns to include in synthetic data generation:",
        options=original_data.columns.tolist(),
        default=original_data.columns.tolist(),
        key="selected_columns",
        help=help_info['synthetic_data_generation_tab']['selected_columns']
    )

    if not selected_columns:
        st.warning("Please select at least one column to proceed.")
        return

    # Automatically detect data types
    selected_data = original_data[selected_columns]
    inferred_categorical_columns = selected_data.select_dtypes(include=['object', 'category']).columns.tolist()
    inferred_numerical_columns = selected_data.select_dtypes(include=['number']).columns.tolist()
    inferred_datetime_columns = selected_data.select_dtypes(include=['datetime', 'datetime64']).columns.tolist()

    st.markdown("**Detected Data Types:**")
    st.markdown(f"**Datetime Columns:** {', '.join(inferred_datetime_columns) if inferred_datetime_columns else 'None'}")
    st.markdown(f"**Categorical Columns:** {', '.join(inferred_categorical_columns) if inferred_categorical_columns else 'None'}")
    st.markdown(f"**Numerical Columns:** {', '.join(inferred_numerical_columns) if inferred_numerical_columns else 'None'}")

    # Optional: Allow user to adjust data types
    adjust_dtypes = st.checkbox("🛠️ **Adjust Column Data Types**", key="adjust_dtypes")

    if adjust_dtypes:
        st.subheader("🛠️ Adjust Column Data Types")

        # Multiselect for datetime columns
        datetime_columns = st.multiselect(
            "Select Datetime Columns:",
            options=selected_columns,
            default=inferred_datetime_columns,
            key="datetime_columns",
            help=help_info['synthetic_data_generation_tab']['select_datetime_columns']
        )

        # Multiselect for categorical columns
        categorical_columns = st.multiselect(
            "Select Categorical Columns:",
            options=[col for col in selected_columns if col not in datetime_columns],
            default=inferred_categorical_columns,
            key="categorical_columns",
            help=help_info['synthetic_data_generation_tab']['select_categorical_columns']
        )

        # Multiselect for numerical columns
        numerical_columns = st.multiselect(
            "Select Numerical Columns:",
            options=[col for col in selected_columns if col not in datetime_columns],
            default=inferred_numerical_columns,
            key="numerical_columns",
            help=help_info['synthetic_data_generation_tab']['select_numerical_columns']
        )

        # Ensure all selected columns are assigned to a category
        if set(selected_columns) != set(datetime_columns).union(set(categorical_columns)).union(set(numerical_columns)):
            st.warning("Please ensure all selected columns are assigned to a category: Datetime, Categorical, or Numerical.")
            st.stop()
    else:
        datetime_columns = inferred_datetime_columns
        categorical_columns = inferred_categorical_columns
        numerical_columns = inferred_numerical_columns

    # Handle missing values
    st.subheader("🚑 Handle Missing Values")
    missing_value_strategy = st.selectbox(
        "Select missing value handling strategy:",
        options=[
            'Drop Rows with Missing Values',
            'Mean Imputation',
            'Median Imputation',
            'Mode Imputation',
            'Fill with Specific Value'
        ],
        key="missing_value_strategy",
        help=help_info['synthetic_data_generation_tab']['missing_value_strategy']
    )
    if missing_value_strategy == 'Fill with Specific Value':
        missing_fill_value = st.text_input("Specify the value to fill missing values with:", value="", key="missing_fill_value")
    else:
        missing_fill_value = None

    # Map strategy to the parameter used in the class
    strategy_mapping = {
        'Drop Rows with Missing Values': 'drop',
        'Mean Imputation': 'mean_impute',
        'Median Imputation': 'median_impute',
        'Mode Imputation': 'mode_impute',
        'Fill with Specific Value': 'fill'
    }
    selected_strategy = strategy_mapping[missing_value_strategy]

    # Select method
    st.subheader("🔧 Select Synthetic Data Generation Method")
    method = st.selectbox(
        "Choose a method:",
        options=['CTGAN', 'Gaussian Copula'],
        index=0,
        key="method_selection",
        help=help_info['synthetic_data_generation_tab']['method']
    )

    # Input model parameters
    st.subheader("⚙️ Set Model Parameters")
    if method.lower() == 'ctgan':
        epochs = st.number_input(
            "Number of Epochs:", 
            min_value=1, 
            max_value=100000, 
            value=300, 
            step=1, 
            key="epochs_input",
            help=help_info['synthetic_data_generation_tab']['ctgan_epochs']
        )
        batch_size = st.number_input(
            "Batch Size:", 
            min_value=1, 
            max_value=10000, 
            value=500, 
            step=1, 
            key="batch_size_input",
            help=help_info['synthetic_data_generation_tab']['ctgan_batch_size']
        )
        model_params = {
            'epochs': epochs,
            'batch_size': batch_size,
            'verbose': True
        }
    else:
        model_params = {}  # No parameters for Gaussian Copula
    
    # Input number of synthetic samples to generate
    num_samples = st.number_input(
        "Number of Synthetic Samples to Generate:",
        min_value=1,
        max_value=200000,
        value=1000,
        step=1,
        key="num_samples_input",
        help=help_info['synthetic_data_generation_tab']['num_samples']
    )

    # Button to start synthetic data generation
    if st.button("🚀 Generate Synthetic Data", key="generate_button", help=help_info['synthetic_data_generation_tab']['generate_synthetic_data']):
        try:
            with st.spinner("Training the model and generating synthetic data..."):
                # Initialize the generator
                synthetic_gen = SyntheticDataGenerator(
                    dataframe=original_data,
                    selected_columns=selected_columns,
                    method=method.lower(),
                    model_params=model_params,
                    missing_value_strategy=selected_strategy,
                    missing_fill_value=missing_fill_value,
                    categorical_columns=categorical_columns if adjust_dtypes else None,
                    numerical_columns=numerical_columns if adjust_dtypes else None,
                    datetime_columns=datetime_columns if adjust_dtypes else None  # Pass datetime_columns
                )

                # Train the model
                synthetic_gen.train()

                # Generate synthetic data
                synthetic_data = synthetic_gen.generate(num_samples=num_samples)

                # Store synthetic data in session state
                st.session_state.synthetic_data = synthetic_data

            st.success("✅ Synthetic data generation completed.")

            # Display synthetic data
            st.subheader("📄 Synthetic Data Sample")
            st.dataframe(st.session_state.synthetic_data.head())

            # Provide download option
            csv = st.session_state.synthetic_data.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Synthetic Data as CSV",
                data=csv,
                file_name='synthetic_data.csv',
                mime='text/csv',
                key="download_button"
            )

        except Exception as e:
            st.error(f"An error occurred: {e}")
            logger.error(traceback.format_exc())

    # Check if synthetic data exists in session state for plotting
    if 'synthetic_data' in st.session_state:
        synthetic_data = st.session_state.synthetic_data
        plot_columns = selected_columns.copy()

        # Optionally, compare distributions
        st.subheader("📊 Compare Distributions")
        column_to_compare = st.selectbox(
            "Select a column to compare distributions:",
            options=plot_columns,
            key="column_to_compare",
            help=help_info['synthetic_data_generation_tab']['compare_distributions']
        )
        if column_to_compare:
            try:
                plot_distributions(original_data, synthetic_data, column_to_compare)
                compare_correlations(original_data[plot_columns], synthetic_data[plot_columns], categorical_columns)
            except Exception as e:
                st.error(f"Error in plotting: {e}")
                logger.error(traceback.format_exc())

def data_processing_settings():
    with st.expander("🔧 Advanced Data Processing Settings"):
        st.session_state.date_threshold = st.slider(
            "Date Detection Threshold", 
            0.0, 1.0, 
            st.session_state.get('date_threshold', 0.0), 
            0.05, 
            help=help_info['data_processing_settings']['date_detection_threshold']
        )
        st.session_state.numeric_threshold = st.slider(
            "Numeric Detection Threshold", 
            0.0, 1.0, 
            st.session_state.get('numeric_threshold', 0.0), 
            0.05, 
            help=help_info['data_processing_settings']['numeric_detection_threshold']
        )
        st.session_state.factor_threshold_ratio = st.slider(
            "Factor Threshold Ratio", 
            0.0, 1.0, 
            st.session_state.get('factor_threshold_ratio', 0.0), 
            0.05, 
            help=help_info['data_processing_settings']['factor_threshold_ratio']
        )
        st.session_state.factor_threshold_unique = st.number_input(
            "Factor Threshold Unique", 
            min_value=10, 
            max_value=10000, 
            value=st.session_state.get('factor_threshold_unique', 10), 
            step=10,
            help=help_info['data_processing_settings']['factor_threshold_unique']
        )
        st.session_state.dayfirst = st.checkbox(
            "Day First in Dates", 
            value=st.session_state.get('dayfirst', False),
            help=help_info['data_processing_settings']['day_first']
        )
        st.session_state.convert_factors_to_int = st.checkbox(
            "Convert Factors to Integers", 
            value=st.session_state.get('convert_factors_to_int', False),
            help=help_info['data_processing_settings']['convert_factors_to_int']
        )
        st.session_state.date_format = st.text_input(
            "Date Format (e.g., '%Y-%m-%d')", 
            value=st.session_state.get('date_format', ''),
            help=help_info['data_processing_settings']['date_format']
        )

# =====================================
# Help Tab
# =====================================

def help_tab():
    st.header("❓ Help & Documentation")
    st.markdown("""    
        ### How to Use This Application
        
        1. **Upload Data:** Start by uploading your dataset in CSV or Pickle format.
        2. **Data Processing:** Adjust data processing settings in the sidebar or use advanced settings for granular control.
        3. **Manual Binning:** Select columns to bin and configure binning parameters.
        4. **Location Granularizer:** Geocode location data and generate granular location information.
        5. **Unique Identification Analysis:** Analyze the uniqueness of your data based on selected columns.
        6. **Data Anonymization:** Apply anonymization techniques to protect sensitive information.
        7. **Synthetic Data Generation:** Generate synthetic datasets based on your original data.
        
        ### Understanding the Settings
        - **Binning Methods:** Quantile ensures equal-sized bins, while Equal Width divides data into bins of equal range.
        - **Anonymization Methods:** 
            - **k-anonymity:** Ensures that each record is indistinguishable from at least k-1 others.
            - **l-diversity:** Extends k-anonymity by ensuring diversity in sensitive attributes.
            - **t-closeness:** Ensures that the distribution of sensitive attributes is close to the original.
        
        ### Best Practices
        - **Start Simple:** Begin with default settings to understand the workflow before customizing.
        - **Validate Results:** Use the evaluation metrics provided to assess the quality of anonymization and synthetic data.
        - **Consult Documentation:** Refer to the README documentation for detailed explanations of methods and parameters.
    """)

# =====================================
# Main Function
# =====================================

def main():
    """Main function to orchestrate the Streamlit app."""
    setup_page()
    initialize_session_state()
    uploaded_file, output_file_type, binning_method = sidebar_inputs()
    data_processing_settings()
    
    # Update binning method in session state
    st.session_state['Binning_Method'] = binning_method

    # Adjust logging based on user selection
    log_level = st.session_state.get('log_level', 'INFO').upper()
    logging.getLogger().setLevel(getattr(logging, log_level, logging.INFO))
    
    # If log_file is not set, default to 'app.log' in LOGS_DIR
    if 'log_file' not in st.session_state:
        st.session_state['log_file'] = os.path.join(LOGS_DIR, 'app.log')
    
    # Configure logging handlers
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    
    handlers = [logging.StreamHandler(sys.stdout)]
    if st.session_state.get('log_file'):
        handlers.append(logging.FileHandler(st.session_state['log_file']))
    
    logging.basicConfig(
        level=getattr(logging, log_level, logging.INFO),
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=handlers
    )
    
    # Display logs if user opted to
    if st.session_state.show_logs:
        display_logs()

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
        mapped_save_type, file_path = save_raw_data(st.session_state.UPLOADED_ORIGINAL_DATA, output_file_type)
        
        
        processed_data = run_processing_cached(
            save_type=mapped_save_type,
            output_filename=f'processed_data.{output_file_type}',
            file_path=os.path.join(DATA_DIR, file_path),
            date_threshold=st.session_state.date_threshold,
            numeric_threshold=st.session_state.numeric_threshold,
            factor_threshold_ratio=st.session_state.factor_threshold_ratio,
            factor_threshold_unique=st.session_state.factor_threshold_unique,
            dayfirst=st.session_state.dayfirst,
            convert_factors_to_int=st.session_state.convert_factors_to_int,
            date_format=st.session_state.date_format
        )
        # Update ORIGINAL_DATA
        update_session_state('ORIGINAL_DATA', processed_data.copy())

        # **Initialize GLOBAL_DATA as a copy of ORIGINAL_DATA**
        update_session_state('GLOBAL_DATA', processed_data.copy())

        if not st.session_state.ORIGINAL_DATA.empty:
            # Display the original data's head
            st.dataframe(st.session_state.ORIGINAL_DATA.head())
            
            # Create a new dataframe to show column types
            column_types = pd.DataFrame({col: [str(dtype)] for col, dtype in st.session_state.ORIGINAL_DATA.dtypes.items()})
            
            # Display the dataframe with column types
            st.dataframe(column_types)
        # Reset processing flags upon new upload
        processing_flags = ['is_binning_done', 'is_geocoding_done', 'is_granular_location_done', 'is_unique_id_done']
        for flag in processing_flags:
            update_session_state(flag, False)
    else:
        st.info("🔄 **Please upload a file to get started.**")
        st.stop()

    # Create Tabs
    tabs = st.tabs([ 
        "📊 Manual Binning", 
        "📍 Location Data Geocoding Granulariser", 
        "🔍 Unique Identification Analysis",
        "🔐 Data Anonymization",
        "🧪 Synthetic Data Generation",  # New Tab
        "❓ Help & Documentation"  # New Tab
    ])
    

    ######################
    # Binning Tab
    ######################
    with tabs[0]:
        binning_tab()

    ######################
    # Location Granulariser Tab
    ######################
    with tabs[1]:
        location_granulariser_tab()

    ######################
    # Unique Identification Analysis Tab
    ######################
    with tabs[2]:
        unique_identification_analysis_tab()

    ######################
    # Data Anonymization Tab
    ######################
    with tabs[3]:
        data_anonymization_tab()

    ######################
    # Synthetic Data Generation Tab
    ######################
    with tabs[4]:
        synthetic_data_generation_tab()
    
    ######################
    # Help & Documentation Tab
    ######################
    with tabs[-1]:
        help_tab()

# =====================================
# Entry Point
# =====================================

if __name__ == "__main__":
    main()

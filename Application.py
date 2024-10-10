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
    perform_integrity_assessment,  # Refactored to return data
    perform_association_rule_mining,
    perform_unique_identification_analysis,
    plot_density_plots_streamlit,
    binning_summary,
    compare_correlations,
    plot_distributions,
    compare_correlations,
    plot_distributions,
    save_dataframe,
    initialize_session_state,
    update_session_state
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

# Import DataAnonymizer Class
from src.data_anonymizer import DataAnonymizer

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
        uploaded_file = st.file_uploader("📤 Upload your dataset", type=['csv', 'pkl'])
        output_file_type = st.selectbox('📁 Select Output File Type', ['csv', 'pkl'], index=0)
        st.markdown("---")


        st.header("⚙️ Binning Options")
        binning_method = st.selectbox('🔧 Select Binning Method', ['Quantile', 'Equal Width'])
        if binning_method == 'Equal Width':
            st.warning("⚠️ **Note:** Using Equal Width will drastically affect the distribution of your data. (Large integrity loss)")  

        st.header("ℹ️ About")
        st.info("""This application allows you to ... (updates will be added upon completion)""")
        
        st.markdown("---")

        st.session_state.show_logs = st.checkbox("🖥️ Show Logs in Interface", value=False, help="Display application logs within the app interface.")
        
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
        key='binning_columns_form'
    )
    update_session_state('Binning_Selected_Columns', selected_columns_binning)


    bins = None
    # Button to start binning
    
    if selected_columns_binning:
        bins = get_binning_configuration(original_data, selected_columns_binning)
    else:
        st.info("🔄 **Please select at least one column to bin.**")
    
    
    # Button to process binning
    st.markdown("---")
    switch_state = st.checkbox('Start Dynamic Binning', key='binning_switch')

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

            # Add association rule mining parameters
            with st.expander("🔍 Association Rule Mining Settings"):
                min_support = st.slider("Minimum Support", 0.01, 1.0, 0.05, 0.01)
                min_threshold = st.slider("Minimum Confidence Threshold", 0.01, 1.0, 0.05, 0.01)
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

def setup_geocoding_options_ui(geocoded_data: pd.DataFrame) -> list:
    """Render the UI for selecting a single column to geocode."""
    detected_geo_columns = detect_geographical_columns(geocoded_data)

    if not detected_geo_columns:
        st.warning("No columns detected that likely contain geographical data. Try uploading a different file or renaming location columns.")
        return []  # Return an empty list

    selected_geo_column = st.selectbox(
        "Select a column to geocode",
        options=detected_geo_columns,
        help="Choose the column containing location data to geocode."
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
        load_map_button = st.button("🗺️ Load Map")
        
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
        st.info("👉 **Please use the Binning and/or Location Granulariser tabs to select columns for analysis.**")

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
            min_comb_size = st.number_input('Minimum Combination Size', min_value=1, max_value=col_count, value=1, step=1)
        with col2:
            max_comb_size = st.number_input('Maximum Combination Size', min_value=min_comb_size, max_value=col_count, value=col_count, step=1)

        if max_comb_size > 5:
            st.warning("⚠️  **Note:** Combinations larger than 5 may take a long time to compute depending on bin count.")

        # Submit button
        submit_button = st.form_submit_button(label='🧮 Perform Unique Identification Analysis')

    return min_comb_size, max_comb_size, submit_button

# =====================================
# Data Anonymization Tab Functionality
# =====================================

def data_anonymization_tab():
    """Render the Data Anonymization Tab in the Streamlit app."""
    st.header("🔐 Data Anonymization")

    # Check if ORIGINAL_DATA is available
    if 'ORIGINAL_DATA' not in st.session_state or st.session_state.ORIGINAL_DATA.empty:
        st.warning("⚠️ **No original data available. Please upload a dataset first.**")
        return

    original_data = st.session_state.ORIGINAL_DATA.copy()

    # Display a sample of the original data
    st.subheader("📋 Original Data Sample")
    st.dataframe(original_data.head())

    # Select Anonymization Method
    anonymization_methods = ['k-anonymity', 'l-diversity', 't-closeness']
    selected_method = st.selectbox("🔧 Select Anonymization Method", options=anonymization_methods)

    # Select Quasi-Identifiers
    st.subheader("🔍 Select Quasi-Identifier Columns")
    quasi_identifiers = st.multiselect(
        "Select columns to generalize (Quasi-Identifiers)",
        options=original_data.columns.tolist(),
        default=[],
        help="Quasi-identifiers are columns that can potentially identify individuals."
    )

    # Select Sensitive Attribute (if needed)
    sensitive_attribute = None
    if selected_method in ['l-diversity', 't-closeness']:
        st.subheader("🔑 Select Sensitive Attribute")
        sensitive_attribute = st.selectbox(
            "Select the sensitive attribute column",
            options=[col for col in original_data.columns if col not in quasi_identifiers],
            help="Sensitive attribute is the column that contains sensitive information."
        )

        if sensitive_attribute in quasi_identifiers:
            st.warning("⚠️ The sensitive attribute should not be among the quasi-identifiers.")
            # Remove it from quasi_identifiers
            quasi_identifiers = [qi for qi in quasi_identifiers if qi != sensitive_attribute]

    if not quasi_identifiers:
        st.info("Please select at least one quasi-identifier to configure generalization parameters.")
        return

    # Input Parameters with Sliders and SelectBoxes
    st.subheader("⚙️ Set Anonymization Parameters")

    # Max iterations slider
    max_iterations = st.slider(
        "Set the maximum number of iterations",
        min_value=1,
        max_value=300,
        value=10,
        step=1,
        help="Maximum number of generalization iterations during anonymization."
    )

    if selected_method == 'k-anonymity':
        # k-value slider
        k_value = st.slider(
            "Set the value of k (for k-anonymity)",
            min_value=2,
            max_value=original_data.shape[0],
            value=2,
            step=1,
            help="k value determines the level of anonymity."
        )
        st.markdown(f"**Selected k-value:** {k_value}")
        l_value = None
        t_value = None
    elif selected_method == 'l-diversity':
        # l-value slider for l-diversity
        l_value = st.slider(
            "Set the value of l (for l-diversity)",
            min_value=1,
            max_value=original_data.shape[0],
            value=2,
            step=1,
            help="l value determines the level of diversity."
        )
        st.markdown(f"**Selected l-value:** {l_value}")
        k_value = None
        t_value = None
    elif selected_method == 't-closeness':
        # t-value slider for t-closeness
        t_value = st.slider(
            "Set the value of t (for t-closeness)",
            min_value=0.0,
            max_value=1.0,
            value=0.2,
            step=0.01,
            help="t value determines the closeness of distribution."
        )
        st.markdown(f"**Selected t-value:** {t_value}")
        k_value = None
        l_value = None

    # Button to perform anonymization
    if st.button("✅ Apply Anonymization"):
        try:
            if not quasi_identifiers:
                st.error("❌ Please select at least one quasi-identifier column.")
                return
            if selected_method in ['l-diversity', 't-closeness'] and not sensitive_attribute:
                st.error("❌ Please select a sensitive attribute for the chosen anonymization method.")
                return
            if sensitive_attribute and sensitive_attribute not in original_data.columns:
                st.error("❌ Selected sensitive attribute does not exist in the data.")
                return

            # Initialize DataAnonymizer with debug callback (optional)
            anonymizer = DataAnonymizer(
                original_data=original_data,
                debug_callback=None  # Pass st.write for debugging
            )

            # Apply anonymization
            anonymizer.anonymize(
                method=selected_method,
                quasi_identifiers=quasi_identifiers,
                k=k_value if selected_method == 'k-anonymity' else None,
                l=l_value if selected_method == 'l-diversity' else None,
                t=t_value if selected_method == 't-closeness' else None,
                sensitive_attribute=sensitive_attribute,
                max_iterations=max_iterations  # Pass the max_iterations from the slider
            )

            # Retrieve anonymized data and report
            anonymized_data = anonymizer.get_anonymized_data()
            report = anonymizer.get_report()

            # Display anonymized data
            st.subheader("🛡️ Anonymized Data Sample")
            st.dataframe(anonymized_data[quasi_identifiers])

            # Display report
            st.subheader("📄 Anonymization Report")
            st.dataframe(report)

            # Provide Download Options
            st.subheader("💾 Download Anonymized Data and Report")
            csv_anonymized = anonymized_data.to_csv(index=False).encode('utf-8')
            csv_report = report.to_csv(index=False).encode('utf-8')

            col1, col2 = st.columns(2)
            with col1:
                st.download_button(
                    label="📥 Download Anonymized Data as CSV",
                    data=csv_anonymized,
                    file_name='anonymized_data.csv',
                    mime='text/csv',
                )
            with col2:
                st.download_button(
                    label="📥 Download Anonymization Report as CSV",
                    data=csv_report,
                    file_name='anonymization_report.csv',
                    mime='text/csv',
                )

            # Update Session State with Anonymized Data and Report
            st.session_state.ANONYMIZED_DATA = anonymized_data
            st.session_state.ANONYMIZATION_REPORT = report

            # Check if desired anonymity was achieved
            achieved_anonymity = report.iloc[-1]['Actual_Value']
            desired_anonymity = k_value or l_value or t_value

            if selected_method == 't-closeness':
                if achieved_anonymity > desired_anonymity:
                    st.warning("⚠️ The desired t-closeness level was not achieved.")
                else:
                    st.success("✅ Data anonymization completed successfully!")
            else:
                if achieved_anonymity < desired_anonymity:
                    st.warning("⚠️ The desired anonymity level was not achieved.")
                else:
                    st.success("✅ Data anonymization completed successfully!")

        except Exception as e:
            st.error(f"❌ An error occurred during anonymization: {e}")
            st.error(traceback.format_exc())


# =====================================
# Synthetic Data Generation Tab Functionality
# =====================================

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
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
        key="selected_columns"
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
            key="datetime_columns"
        )

        # Multiselect for categorical columns
        categorical_columns = st.multiselect(
            "Select Categorical Columns:",
            options=[col for col in selected_columns if col not in datetime_columns],
            default=inferred_categorical_columns,
            key="categorical_columns"
        )

        # Multiselect for numerical columns
        numerical_columns = st.multiselect(
            "Select Numerical Columns:",
            options=[col for col in selected_columns if col not in datetime_columns],
            default=inferred_numerical_columns,
            key="numerical_columns"
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
        key="missing_value_strategy"
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
        key="method_selection"
    )

    # Input model parameters
    st.subheader("⚙️ Set Model Parameters")
    if method.lower() == 'ctgan':
        epochs = st.number_input("Number of Epochs:", min_value=1, max_value=100000, value=300, step=1, key="epochs_input")
        batch_size = st.number_input("Batch Size:", min_value=1, max_value=10000, value=500, step=1, key="batch_size_input")
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
        key="num_samples_input"
    )

    # Button to start synthetic data generation
    if st.button("🚀 Generate Synthetic Data", key="generate_button"):
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
            key="column_to_compare"
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
        st.session_state.date_threshold = st.slider("Date Detection Threshold", 0.0, 1.0, st.session_state.get('date_threshold', 0.0), 0.05)
        st.session_state.numeric_threshold = st.slider("Numeric Detection Threshold", 0.0, 1.0, st.session_state.get('numeric_threshold', 0.0), 0.05)
        st.session_state.factor_threshold_ratio = st.slider("Factor Threshold Ratio", 0.0, 1.0, st.session_state.get('factor_threshold_ratio', 0.0), 0.05)
        st.session_state.factor_threshold_unique = st.number_input("Factor Threshold Unique", 
                                                                   min_value=10, 
                                                                   max_value=10000, 
                                                                   value=st.session_state.get('factor_threshold_unique', 10), 
                                                                   step=10
                                                                   )
        st.session_state.dayfirst = st.checkbox("Day First in Dates", value=st.session_state.get('dayfirst', False))
        st.session_state.convert_factors_to_int = st.checkbox("Convert Factors to Integers", value=st.session_state.get('convert_factors_to_int', False))
        st.session_state.date_format = st.text_input("Date Format (e.g., '%Y-%m-%d')", value=st.session_state.get('date_format', ''),
                                                     help="Leave blank to retain datetime dtype"
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

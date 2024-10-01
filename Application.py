# Application.py

import streamlit as st
import pandas as pd
import os
import traceback  # For detailed error logging

# Import binning modules
from src.binning import DataBinner

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

    st.session_state.ORIGINAL_DATA = Data.copy()
    st.session_state.GLOBAL_DATA = Data.copy()  # Initialize GLOBAL_DATA

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

    st.session_state.Processed_Data = processed_data.copy()


def initialize_session_state():
    """Initialize session state variables if not already present."""
    if 'ORIGINAL_DATA' not in st.session_state:
        st.session_state.ORIGINAL_DATA = pd.DataFrame()
    if 'GLOBAL_DATA' not in st.session_state:
        st.session_state.GLOBAL_DATA = pd.DataFrame()

    ##############      SELECTION CHECKS       ##############
    if 'Binning_Selected_Columns' not in st.session_state:
        st.session_state.Binning_Selected_Columns = []


    if 'Location_Selected_Columns' not in st.session_state:
        st.session_state.Location_Selected_Columns = []
    if 'Granular_Location_Column_Set' not in st.session_state:
        st.session_state.Granular_Location_Column_Set = False 
    ##########################################################

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
        mapped_save_type, file_path = save_raw_data(st.session_state.ORIGINAL_DATA, output_file_type)
        run_data_processing(mapped_save_type, output_file_type, file_path)
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
        
        # Exclude columns selected in Location Granulariser
        available_columns = list(set(st.session_state.Processed_Data.select_dtypes(
            include=['number', 'datetime', 'datetime64[ns, UTC]', 'datetime64[ns]']
        ).columns.tolist()) - set(st.session_state.Location_Selected_Columns))
        
        with st.form("binning_form"):
            selected_columns_binning = st.multiselect(
                'Select columns to bin',
                options=available_columns,
                default=st.session_state.get('Binning_Selected_Columns', []),
                key='binning_columns_form'
            )
            st.session_state.Binning_Selected_Columns = selected_columns_binning

            if selected_columns_binning:
                bins = get_binning_configuration(st.session_state.Processed_Data, selected_columns_binning)
                st.session_state.bins = bins
            else:
                st.info("🔄 **Please select at least one column to bin.**")

            submitted = st.form_submit_button("🔄 Run Binning")

        if submitted:
            bins = st.session_state.get('bins', None)
            if not bins:
                st.error("No binning configuration found. Please configure bins before running binning.")
            else:
                overlapping_columns = set(st.session_state.Binning_Selected_Columns) & set(st.session_state.Location_Selected_Columns)
                if overlapping_columns:
                    st.error(f"❌ The following columns are selected in both Binning and Location Granulariser tabs: {', '.join(overlapping_columns)}. Please select distinct columns.")
                else:
                    try:
                        OG_Data_BinTab, Data_BinTab = perform_binning(
                            st.session_state.Processed_Data,
                            selected_columns_binning,
                            binning_method,
                            bins
                        )

                        perform_integrity_assessment(OG_Data_BinTab, Data_BinTab, selected_columns_binning)
                        plot_density(
                            OG_Data_BinTab[selected_columns_binning].astype('category'),
                            Data_BinTab[selected_columns_binning],
                            selected_columns_binning
                        )
                        
                        # **Updated: Only replace binning columns without resetting GLOBAL_DATA**
                        for col in selected_columns_binning:
                            st.session_state.GLOBAL_DATA[col] = Data_BinTab[col]

                        st.success("✅ Binning completed successfully!")

                        st.subheader('📊 Data Preview (Global Data)')
                        st.dataframe(st.session_state.GLOBAL_DATA.head())

                        download_binned_data()
                    except Exception as e:
                        st.error(f"Error during binning: {e}")
        else:
            st.info("👉 Select columns and submit the form to run binning.")

    ######################
    # Location Granulariser Tab
    ######################
    with tabs[1]:
        st.header("📍 Location Data Geocoding Granulariser")
        # Geocoding process
        st.header("1️⃣ Geocoding")


        OG_Data_GeoTab = st.session_state.ORIGINAL_DATA.copy()


        def setup_geocoding_options(OG_Data_GeoTab):
            geo_columns = detect_geographical_columns(OG_Data_GeoTab)

            selected_columns = st.multiselect(
                "Select columns to geocode",
                options=geo_columns,
                help="Choose the columns containing location data to geocode.",
                max_selections=1 
            )

            if not geo_columns:
                st.warning("No columns detected that likely contain geographical data. Try uploading a different file or renaming location columns.")
                st.stop()

            st.session_state.Location_Selected_Columns = selected_columns

            OG_Data_GeoTab = OG_Data_GeoTab[geo_columns]

            return selected_columns, OG_Data_GeoTab

        def perform_geocoding_process(selected_columns, OG_Data_GeoTab):
            """Perform geocoding on the selected columns."""
            if not selected_columns:
                st.warning("Please select at least one column to geocode.")
            else:
                try:
                    with st.spinner("Processing..."):
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        geocoded_df = perform_geocoding(
                            data=OG_Data_GeoTab,
                            selected_columns=selected_columns,
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
                    st.error(traceback.format_exc())
                    st.stop()

        location_column_selected, OG_Data_GeoTab = setup_geocoding_options(OG_Data_GeoTab)
        preprocess_button = st.button("📂 Start Geocoding")

        if preprocess_button:
            perform_geocoding_process(location_column_selected, OG_Data_GeoTab)

        # Display geocoded data
        display_geocoded_data()
        # Granular location generation
        st.header("2️⃣ Granular Location Generation")


        def perform_granular_location_generation(granularity):
            """Perform granular location generation."""
            if st.session_state.geocoded_data is None:
                st.warning("Please perform geocoding first.")
            else:
                column_name = f"Granular Location ({granularity.capitalize()})"
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
                    # Add the granular column to GLOBAL_DATA
                    st.session_state.GLOBAL_DATA[location_column_selected] = granular_df[column_name].astype('category')
                    st.session_state.Granular_Location_Column_Set = True

                    if column_name not in st.session_state.Location_Selected_Columns:
                        st.session_state.Location_Selected_Columns.append(column_name)
                        st.write(f"🔄 **Added '{column_name}' to Location_Selected_Columns list.**")
                    else:
                        st.write(f"ℹ️ **'{column_name}' is already in Location_Selected_Columns list. Skipping append.**")

                    # Display unique values for verification
                    st.write(f"**Unique values in {column_name}:**")
                    unique_values = st.session_state.GLOBAL_DATA[column_name].unique()
                    st.write(unique_values)
                    st.write(f"**Number of Unique Values:** {len(unique_values)}")

                    # Debug: Confirm addition to GLOBAL_DATA
                    st.write(f"🔍 **GLOBAL_DATA Columns After Adding Granular Location:** {st.session_state.GLOBAL_DATA.columns.tolist()}")
                except ValueError as ve:
                    st.warning(str(ve))
                except Exception as e:
                    st.error(f"❌ An unexpected error occurred during granular location generation: {e}")
                    st.error(traceback.format_exc())
                    st.stop()

        granularity_options = ["address", "suburb", "city", "state", "country", "continent"]
        granularity = st.selectbox(
            "Select Location Granularity",
            options=granularity_options,
            help="Choose the level of granularity for location identification."
        )

        generate_granular_button = st.button("📈 Generate Granular Location Column")

        if generate_granular_button:
            perform_granular_location_generation(granularity)

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

        # Determine selected columns based on session state
        if (
            st.session_state.Binning_Selected_Columns and
            st.session_state.Location_Selected_Columns and
            st.session_state.Granular_Location_Column_Set
        ):
            # Combine binning and granular location columns
            granular_columns = st.session_state.Location_Selected_Columns  # Using explicitly tracked granular columns
            selected_columns = st.session_state.Binning_Selected_Columns + granular_columns
        elif st.session_state.Binning_Selected_Columns:
            selected_columns = st.session_state.Binning_Selected_Columns
        elif st.session_state.Location_Selected_Columns and st.session_state.Granular_Location_Column_Set:
            granular_columns = st.session_state.Location_Selected_Columns
            selected_columns = granular_columns
        else:
            selected_columns = None

        # Debug: Display selected_columns
        st.write(f"🔍 **Selected Columns for Analysis:** {selected_columns}")

        # Debug: Display GLOBAL_DATA columns
        st.write(f"🔍 **GLOBAL_DATA Columns:** {st.session_state.GLOBAL_DATA.columns.tolist()}")

        # Display global data if available
        if not st.session_state.GLOBAL_DATA.empty:
            st.subheader('📊 Data Preview (Global Data)')
            st.dataframe(st.session_state.GLOBAL_DATA.head())

            # Use columns present in GLOBAL_DATA
            if not selected_columns:
                st.warning("⚠️ **No columns selected in Binning or Location Granulariser tabs for analysis.**")
                st.info("🔄 **Please select columns in the Binning tab and/or generate granular location data to perform Unique Identification Analysis.**")
            else:
                # **Fix: Remove duplicate columns from selected_columns**
                selected_columns = list(dict.fromkeys(selected_columns))
                st.write(f"🔍 **Selected Columns for Analysis (After Deduplication):** {selected_columns}")

                # Verify that selected_columns exist in GLOBAL_DATA
                existing_columns = [col for col in selected_columns if col in st.session_state.GLOBAL_DATA.columns]
                missing_columns = [col for col in selected_columns if col not in st.session_state.GLOBAL_DATA.columns]
                if missing_columns:
                    st.write(f"**Columns selected for analysis that are missing:** {', '.join(missing_columns)}")
                
                if missing_columns:
                    st.error(f"The following selected columns are missing from Global Data: {', '.join(missing_columns)}. Please check your selections.")
                    st.error(traceback.format_exc())
                    st.stop()

                st.write(f"**Columns selected for analysis:** {', '.join(existing_columns)}")

                # **Critical Fix: Pass ORIGINAL_DATA as original_for_assessment and GLOBAL_DATA as binned_for_assessment**
                try:
                    unique_identification_section(
                        original_for_assessment=st.session_state.ORIGINAL_DATA[existing_columns].astype('category'),
                        binned_for_assessment=st.session_state.GLOBAL_DATA[existing_columns].astype('category'),
                        selected_columns=existing_columns
                    )
                except KeyError as ke:
                    st.error(f"KeyError: {ke}. Please ensure all selected columns exist in both Original and Global Data.")
                    st.stop()
                except Exception as e:
                    st.error(f"Unexpected error: {e}")
                    st.error(traceback.format_exc())
                    st.stop()
        else:
            st.info("🔄 **Please upload and process data to perform Unique Identification Analysis.**")


#########################
# Binning Functions
#########################

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
    binned_for_assessment = Data_BinTab[selected_columns_binning]

    handle_integrity_assessment(original_for_assessment, binned_for_assessment, PLOTS_DIR)


def plot_density(original_for_assessment, binned_for_assessment, selected_columns_binning):
    """Plot and display density plots."""
    plot_density_plots_and_display(original_for_assessment, binned_for_assessment, selected_columns_binning, PLOTS_DIR)


def download_binned_data():
    """Handle downloading of the binned data."""
    handle_download_binned_data(
        data=st.session_state.GLOBAL_DATA,
        file_type_download=st.selectbox('📁 Select Download File Type', ['csv', 'pkl'], index=0, key='download_file_type_download'),
        save_dataframe_func=save_dataframe,
        plots_dir=PLOTS_DIR
    )


#########################
# Unique Identification Functions
#########################

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


#########################
# Location Granulariser Functions
#########################

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

def display_geocoded_with_granular_data():
    """Display geocoded data with granular location and provide download options."""
    if st.session_state.geocoded_data is not None:
        # Check if any granular location columns exist in GLOBAL_DATA
        granular_columns_present = [col for col in st.session_state.Location_Selected_Columns if col in st.session_state.GLOBAL_DATA.columns]
        if granular_columns_present:
            st.subheader("📝 Geocoded Data with Granular Location")
            st.dataframe(st.session_state.GLOBAL_DATA[granular_columns_present].head())
            
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


if __name__ == "__main__":
    main()

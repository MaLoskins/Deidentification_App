# Data Processing, Binning, and Uniquity Assessment Application


![Python Version](https://img.shields.io/badge/python-3.10.11%2B-blue.svg)

A comprehensive Python application for data preprocessing, binning, integrity assessment, and unique identification analysis. Leveraging Streamlit for an interactive web interface, this application provides robust tools for handling and analyzing datasets with ease.

## Table of Contents

- [Data Processing, Binning, and Uniquity Assessment Application](#data-processing-binning-and-uniquity-assessment-application)
  - [Table of Contents](#table-of-contents)
  - [Features](#features)
    - [Description of Key Directories and Files](#description-of-key-directories-and-files)
  - [Installation](#installation)
  - [Usage](#usage)
      - [Configure Processing Parameters](#configure-processing-parameters)
      - [If not Using the Application (For Data Processing):](#if-not-using-the-application-for-data-processing)
      - [If Using Application](#if-using-application)
  - [Start the Streamlit App](#start-the-streamlit-app)
    - [Running the Streamlit App](#running-the-streamlit-app)
      - [Interact with the Application](#interact-with-the-application)
    - [Key Components](#key-components)
      - [`Application.py`](#applicationpy)
      - [`data_binner.py`](#data_binnerpy)
      - [`density_plotter.py`](#density_plotterpy)
      - [`data_integrity_assessor.py`](#data_integrity_assessorpy)
      - [`unique_bin_identifier.py`](#unique_bin_identifierpy)
      - [`Process_Data.py`](#process_datapy)
      - [`Detect_Dtypes.py`](#detect_dtypespy)
    - [Configuration](#configuration)
      - [DataProcessor Configuration (`Process_Data.py`)](#dataprocessor-configuration-process_datapy)
      - [DtypeDetector Configuration (`Detect_Dtypes.py`)](#dtypedetector-configuration-detect_dtypespy)
      - [Logging](#logging)

## Features

- **Interactive Data Upload:** Upload datasets in CSV or Pickle.
- **Dynamic Binning:** Select and bin numerical and datetime columns with customizable bin sizes.
- **Data Integrity Assessment:** Evaluate the impact of binning on data integrity using entropy metrics.
- **Density Plots:** Visualize the distribution of original and binned data through comprehensive density plots.
- **Unique Identification Analysis:** Analyze combinations of binned columns to determine unique identifications within the dataset.
- **Automated Data Processing:** Streamlined preprocessing pipeline with configurable parameters and reporting.


### Description of Key Directories and Files

- **Application.py:** The main Streamlit application for interactive data processing and analysis.
- **Category_Mappings:** Directory containing CSV files that map categorical variables to their respective categories.
- **Data.csv:** The raw input dataset to be processed.
- **Detect_Dtypes.py:** Module for detecting and converting data types of dataset columns.
- **Process_Data.py:** Script to preprocess and process the raw dataset.
- **Processed_Data.pkl:** Serialized processed dataset saved in Pickle format.
- **Run_Processor.py:** Script to execute the data processing pipeline.
- **Type_Conversion_Report.csv:** Report detailing data type conversions performed during preprocessing.
- **__pycache__:** Python cache files (automatically generated).
- **data_binner.py:** Module containing the `DataBinner` class for binning data columns.
- **data_integrity_assessor.py:** Module containing the `DataIntegrityAssessor` class for assessing integrity loss post-binning.
- **density_plotter.py:** Module containing the `DensityPlotter` class for generating density plots.
- **requirements.txt:** List of Python dependencies required for the application.
- **unique_bin_identifier.py:** Module containing the `UniqueBinIdentifier` class for unique identification analysis.

## Installation

1. **Clone the Repository**

   ```bash
   git clone https://github.com/MaLoskins/Deidentification_App.git
   cd Deidentification_App
   ```

2. **Create a Virtual Environment**

    It's recommended to use a virtual environment to manage dependencies. Ensure you are using `python version 3.10.11`

    ```bash
    python3 -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```


3. **Install Dependencies**

    Install the required Python packages using `requirements.txt.`

    ```
    pip install -r requirements.txt
    ```


## Usage

#### Configure Processing Parameters
Modify `Process_Data.py` or `Application.py` if you need to change default processing parameters such as thresholds, file paths, or conversion settings.

#### If not Using the Application (For Data Processing):

Execute the data processing script to preprocess Data.csv.

```bash
python Run_Processor.py
```

```bash
python Process_Data.py
```
- `Processed_Data.csv` or `Processed_Data.pkl`: The processed dataset.
- `Type_Conversion_Report.csv`: Details of data type conversions performed.
- `Category_Mappings/`: Updated category mapping files for categorical variables.

This will automatically convert any data set into the processed formatting. If data types need to be retained, ensure that the file is saved as either a pickle file (`.pkl`) if you want to retain data types.

#### If Using Application 

Load original `.csv` file into the prompt in the application

## Start the Streamlit App

Start the Streamlit App

```bash
streamlit run Application.py
```

### Running the Streamlit App
Launch the interactive Streamlit application to perform further data analysis and visualization.

#### Interact with the Application
- **Upload Dataset**: Choose the file type (CSV, Pickle) and upload your processed dataset.
- **Preview Data**: View the first few rows of your dataset.
- **Select Columns to Bin**: Choose compatible columns for binning.
- **Configure Binning**: Adjust the number of bins for each selected column.
- **View Binned Data**: See the categorization of binned columns and assess data integrity loss.
- **Density Plots**: Visualize the distribution of original and binned data.
- **Unique Identification Analysis**: Analyze combinations of binned columns to determine unique identifications.

### Key Components

#### `Application.py`
The main Streamlit application that provides an interactive interface for data uploading, binning, visualization, and analysis.

**Features**:
- Upload datasets in various formats.
- Select and bin numerical and datetime columns.
- Assess data integrity loss post-binning.
- Generate density plots for original and binned data.
- Perform unique identification analysis on binned columns.

#### `data_binner.py`
Contains the DataBinner class responsible for binning specified columns in a DataFrame based on user-defined bin counts.

**Key Methods**:
- `bin_columns`: Bins the specified columns and categorizes them by data type.
- `get_binned_data`: Retrieves the binned DataFrame.
- `get_binned_columns`: Provides a categorization of binned columns.

#### `density_plotter.py`
Includes the DensityPlotter class for generating density plots of categorical columns.

**Key Methods**:
- `plot_grid`: Creates a grid of density plots for selected columns.

#### `data_integrity_assessor.py`
Features the DataIntegrityAssessor class to evaluate the impact of binning on data integrity using entropy measures.

**Key Methods**:
- `assess_integrity_loss`: Calculates entropy loss for each variable.
- `generate_report`: Generates a detailed integrity loss report.
- `plot_entropy`: Visualizes the entropy before and after binning.

#### `unique_bin_identifier.py`
Encapsulates the UniqueBinIdentifier class, which analyzes combinations of binned columns to identify unique observations.

**Key Methods**:
- `find_unique_identifications`: Analyzes combinations of bin columns for unique identifications.
- `plot_results`: Visualizes the top combinations with the highest unique identifications.

#### `Process_Data.py`
Implements the DataProcessor class to preprocess the raw dataset, detect and convert data types, and save processed data along with mapping files.

**Key Features**:
- Reads and cleans input data.
- Detects data types using `Detect_Dtypes`.
- Saves processed data in specified formats (CSV, Pickle).
- Generates category mapping files for categorical variables.

#### `Detect_Dtypes.py`
Contains the DtypeDetector class that detects and converts data types of DataFrame columns based on configurable thresholds.

**Key Features**:
- Cleans column names.
- Determines column types (int, float, date, factor, bool, string).
- Converts columns to appropriate data types.
- Generates mappings for categorical variables.

### Configuration

#### DataProcessor Configuration (`Process_Data.py`)
- **Input/Output Paths**:
  - `input_filepath`: Path to the raw data file (default: Data.csv).
  - `output_filepath`: Path to save the processed data.
  - `report_path`: Path to save the type conversion report.
  - `mapping_directory`: Directory to save category mapping files.
- **Processing Options**:
  - `return_category_mappings`: Whether to save category mappings.
  - `parallel_processing`: Enable or disable parallel processing.
  - `save_type`: Format to save processed data (csv, pickle).
- **Thresholds**:
  - `date_threshold`: Threshold for date detection.
  - `numeric_threshold`: Threshold for numeric detection.
  - `factor_threshold_ratio`: Ratio threshold for factor detection.
  - `factor_threshold_unique`: Unique value threshold for factor detection.
- **Additional Options**:
  - `dayfirst`: Interpret the first value in dates as the day.
  - `convert_factors_to_int`: Convert categorical factors to integer codes.
  - `date_format`: Desired date format for output.

#### DtypeDetector Configuration (`Detect_Dtypes.py`)
- **Thresholds**:
  - `date_threshold`
  - `numeric_threshold`
  - `factor_threshold_ratio`
  - `factor_threshold_unique`
- **Options**:
  - `dayfirst`
  - `convert_factors_to_int`
  - `date_format`

#### Logging
- **log_level**: Set logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
- **log_file**: Path to save log files.



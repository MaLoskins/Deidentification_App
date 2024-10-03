# README FILE NEEDS TO BE UPDATED DO NOT USE FOR REFERENCE

A comprehensive data processing and analysis application built with Streamlit, designed to facilitate data binning, integrity assessment, visualization, and unique identification analyses. This application provides a user-friendly web interface for uploading datasets, configuring binning options, and obtaining insightful reports and visualizations.

## Table of Contents

- [README FILE NEEDS TO BE UPDATED DO NOT USE FOR REFERENCE](#readme-file-needs-to-be-updated-do-not-use-for-reference)
  - [Table of Contents](#table-of-contents)
  - [Features](#features)
  - [Installation](#installation)
  - [Usage](#usage)
  - [Directory Structure](#directory-structure)
  - [Detailed Documentation](#detailed-documentation)
    - [1. Application.py](#1-applicationpy)
    - [2. data\_binner.py](#2-data_binnerpy)
    - [3. data\_integrity\_assessor.py](#3-data_integrity_assessorpy)
    - [4. density\_plotter.py](#4-density_plotterpy)
    - [5. Detect\_Dtypes.py](#5-detect_dtypespy)
    - [6. Process\_Data.py](#6-process_datapy)
    - [7. unique\_bin\_identifier.py](#7-unique_bin_identifierpy)
    - [8. utils.py](#8-utilspy)
  - [Additional Components](#additional-components)
    - [A. `outputs` Directory](#a-outputs-directory)
    - [B. `README.md`](#b-readmemd)
    - [C. `requirements.txt`](#c-requirementstxt)
  - [Pipeline Summary](#pipeline-summary)
  - [Requirements](#requirements)
  - [License](#license)

## Features

- **User-Friendly Interface:** Built with Streamlit for an intuitive web-based experience.
- **Flexible Binning:** Supports Quantile and Equal Width binning methods with customizable configurations.
- **Data Integrity Assessment:** Evaluates the impact of binning on data entropy.
- **Visualization:** Generates density and entropy plots for comprehensive data analysis.
- **Unique Identification Analysis:** Determines the uniqueness of record combinations based on binned data.
- **Comprehensive Reporting:** Provides detailed reports and downloadable processed data.

## Installation

1. **Clone the Repository**

   ```bash
   git clone https://github.com/yourusername/your-repo.git
   cd your-repo
   ```

2. **Create a Virtual Environment**

   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install Dependencies**

   ```bash
   pip install -r requirements.txt
   ```

4. **Run the Application**

   ```bash
   streamlit run Application.py
   ```

## Usage

1. **Upload Dataset:** Use the interface to upload a CSV or Pickle file.
2. **Configure Binning:** Select the binning method and choose the columns to bin.
3. **Process Data:** The application will process and bin the data based on your configurations.
4. **Assess Integrity:** Review the integrity loss reports and entropy plots.
5. **Visualize Data:** View density plots of original and binned data.
6. **Unique Identification:** Analyze the uniqueness of record combinations.
7. **Download Results:** Download processed data and reports as needed.

## Directory Structure

```
.
├── Application.py
├── Binning_Tab
│   ├── Data.csv
│   ├── Detect_Dtypes.py
│   ├── Process_Data.py
│   ├── data_binner.py
│   ├── data_integrity_assessor.py
│   ├── density_plotter.py
│   ├── unique_bin_identifier.py
│   └── utils.py
├── Data.csv
├── README.md
├── outputs
│   ├── category_mappings
│   ├── plots
│   ├── processed_data
│   ├── reports
│   └── unique_identifications
└── requirements.txt
```

## Detailed Documentation

### 1. Application.py

**Purpose**

`Application.py` serves as the main entry point of the application, built using Streamlit to provide a user-friendly web interface. It orchestrates the entire data processing pipeline, allowing users to upload datasets, configure binning options, assess data integrity, visualize data distributions, and perform unique identification analyses.

**Key Functionalities**

- **User Interface Setup:** Configures the Streamlit page, including layout and custom styles.
- **File Upload & Settings:** Allows users to upload datasets (.csv or .pkl) and select output formats.
- **Binning Configuration:** Enables users to choose binning methods (Quantile or Equal Width) and select columns for binning.
- **Data Processing Pipeline:** Loads, processes, and bins the data using auxiliary scripts.
- **Data Integrity Assessment:** Evaluates the impact of binning on data integrity.
- **Visualization:** Generates and displays density and entropy plots.
- **Unique Identification Analysis:** Analyzes combinations of binned columns to determine their uniqueness in identifying records.
- **Download Options:** Provides functionality to download processed data and analysis results.

**Inputs**

- **User-Uploaded Data:** CSV or Pickle files containing the dataset to be processed.
- **User Selections:**
  - Output file type (csv or pkl)
  - Binning method (Quantile or Equal Width)
  - Columns to bin
  - Binning configurations via sliders

**Outputs**

- **Processed Data:** Binned dataset saved in the chosen format.
- **Reports:** Integrity loss report and type conversion report.
- **Plots:** Binned density plots, entropy plots, and original density plots.
- **Unique Identification Results:** Analysis results indicating the uniqueness of record combinations.
- **Downloadable Files:** Options to download processed data and analysis reports.

**Pipeline Role**

Acts as the central coordinator, integrating various components like data loading, binning, integrity assessment, visualization, and analysis. It leverages scripts within the `Binning_Tab` directory to perform specific tasks and presents the results to the user through a cohesive interface.

---

### 2. data_binner.py

**Purpose**

`data_binner.py` contains the `DataBinner` class, which is responsible for binning specified columns in a Pandas DataFrame. Binning transforms continuous or categorical data into discrete bins, aiding in data analysis and visualization.

**Key Functionalities**

- **Initialization:** Sets up the original and binned DataFrames, and categorizes columns based on their data types.
- **Binning Methods:** Supports Equal Width and Quantile binning techniques.
- **Column Type Handling:** Bins columns differently based on whether they're datetime, integer, or float types.
- **Error Handling:** Skips unsupported columns and logs relevant warnings.

**Inputs**

- **DataFrame (Data):** The original dataset to be binned.
- **Binning Method (method):** Specifies the binning technique (equal width or quantile).
- **Binning Configuration (bin_dict):** A dictionary specifying the number of bins for each column.

**Outputs**

- **Binned DataFrame (binned_df):** The DataFrame containing binned columns.
- **Binned Columns Dictionary (binned_columns):** Categorizes successfully binned columns by their data types (datetime, integer, float).

**Pipeline Role**

Operates as the core component for transforming data within the pipeline. After data type detection and conversion (handled by `Detect_Dtypes.py` and `Process_Data.py`), `DataBinner` performs the binning operation, which is essential for subsequent data integrity assessments and visualizations.

---

### 3. data_integrity_assessor.py

**Purpose**

`data_integrity_assessor.py` houses the `DataIntegrityAssessor` class, which evaluates the loss of information resulting from the binning process. It quantifies how much the original data's entropy has been reduced post-binning.

**Key Functionalities**

- **Entropy Calculation:** Computes the entropy of both original and binned columns to measure data complexity.
- **Integrity Loss Assessment:** Determines the absolute and percentage loss in entropy for each column.
- **Report Generation:** Creates a detailed report summarizing the integrity loss per column and overall.
- **Visualization:** Generates entropy comparison plots to visually represent data loss.

**Inputs**

- **Original DataFrame (original_df):** The dataset before binning.
- **Binned DataFrame (binned_df):** The dataset after binning.

**Outputs**

- **Integrity Loss Report (`Integrity_Loss_Report.csv`):** A CSV file detailing entropy loss metrics for each column.
- **Entropy Plot (`entropy_plot.png`):** A visual comparison of original and binned entropies.
- **Overall Loss Metric:** Average percentage loss across all columns.

**Pipeline Role**

Following the binning operation, `DataIntegrityAssessor` assesses the quality and impact of binning by measuring data integrity loss. This evaluation is crucial for understanding the trade-offs between data simplification and information preservation, informing decisions on binning configurations.

---

### 4. density_plotter.py

**Purpose**

`density_plotter.py` features the `DensityPlotter` class, which generates density plots for selected categorical or integer columns in a DataFrame. These plots help visualize the distribution and density of data points across different bins or categories.

**Key Functionalities**

- **Plot Generation:** Creates density plots for specified columns, supporting both original and binned data.
- **Customization:** Allows configuration of figure size, plot style, and save paths.
- **Grid Layout:** Organizes multiple plots in a grid for comprehensive visualization.
- **Saving & Displaying:** Offers options to save plots as images or display them within the Streamlit interface.

**Inputs**

- **DataFrame (dataframe):** The dataset containing the columns to plot.
- **Category Columns (category_columns):** List of columns to generate density plots for.
- **Plotting Parameters:** Including figure size, save path, and plot style.

**Outputs**

- **Density Plots:** Visual representations of data distributions, either displayed in the application or saved as image files.

**Pipeline Role**

Integrates into the visualization phase of the pipeline. After data is binned and integrity assessed, `DensityPlotter` provides insights into the distribution of both original and binned data, facilitating better understanding and interpretation of the binning effects.

---

### 5. Detect_Dtypes.py

**Purpose**

`Detect_Dtypes.py` contains the `DtypeDetector` class, which is pivotal for detecting and converting data types within a DataFrame. Accurate data type detection ensures that subsequent processing steps, like binning, operate correctly.

**Key Functionalities**

- **Data Type Detection:** Identifies whether columns are integers, floats, dates, booleans, factors (categoricals), or strings based on configurable thresholds.
- **Data Conversion:** Converts columns to appropriate types, handling special cases like date formatting and categorical encoding.
- **Category Mapping:** Maintains mappings between categorical codes and their original labels.
- **Reporting:** Generates type conversion reports detailing the transformations applied to each column.
- **Parallel Processing:** Supports concurrent processing of columns to enhance performance.

**Inputs**

- **DataFrame (data):** The dataset requiring data type detection and conversion.
- **Configuration Parameters:** Thresholds for detecting numeric, date, and factor types, along with logging configurations and conversion options.

**Outputs**

- **Processed DataFrame:** The DataFrame with columns converted to their detected types.
- **Type Conversion Report (`Type_Conversion_Report.csv`):** Details the original and new data types of each column.
- **Category Mappings:** CSV files mapping categorical codes to their original labels, saved in the `category_mappings` directory.

**Pipeline Role**

Functions as the initial step in the data processing pipeline. Before any binning or analysis, `DtypeDetector` ensures that all columns are correctly typed, which is essential for accurate binning, integrity assessment, and further analyses.

---

### 6. Process_Data.py

**Purpose**

`Process_Data.py` encapsulates the `DataProcessor` class, which coordinates the data processing workflow. It leverages the `DtypeDetector` to prepare the data for binning and ensures that all necessary outputs and reports are generated and saved appropriately.

**Key Functionalities**

- **Data Reading:** Loads the input dataset from a CSV file.
- **Data Type Processing:** Utilizes `DtypeDetector` to detect and convert column data types.
- **Output Saving:** Saves the processed DataFrame in specified formats (csv, pkl, parquet).
- **Category Mappings:** Saves mappings for categorical columns to facilitate reverse lookup or interpretation.
- **Error Handling:** Implements robust error handling to manage failures in parallel or sequential processing.

**Inputs**

- **Input File Path (input_filepath):** Path to the raw data CSV file.
- **Output File Path (output_filepath):** Destination path for the processed data.
- **Configuration Parameters:** Including thresholds, logging settings, binning options, and save formats.

**Outputs**

- **Processed Data:** DataFrame saved in the chosen format (csv, pkl, or parquet).
- **Type Conversion Report (`Type_Conversion_Report.csv`):** Generated by `DtypeDetector`.
- **Category Mappings:** CSV files mapping categorical codes to original labels.

**Pipeline Role**

Acts as the bridge between raw data ingestion and the binning process. `DataProcessor` ensures that data types are correctly identified and converted, making the dataset ready for effective binning and subsequent analyses. It's invoked by `Application.py` through utility functions to prepare data for user-defined binning operations.

---

### 7. unique_bin_identifier.py

**Purpose**

`unique_bin_identifier.py` houses the `UniqueBinIdentifier` class, which analyzes combinations of binned columns to determine how effectively they uniquely identify records in the dataset. This analysis helps in understanding the discriminative power of various binning configurations.

**Key Functionalities**

- **Combination Analysis:** Evaluates all possible combinations of selected bin columns within specified size ranges.
- **Unique Identification Counting:** Calculates the number of records uniquely identified by each combination.
- **Result Aggregation:** Compiles results into a DataFrame, ranking combinations by their uniqueness.
- **Visualization:** Generates plots highlighting the top combinations that offer the highest unique identifications.
- **Progress Tracking:** Provides real-time feedback on the analysis progress, especially for large datasets.

**Inputs**

- **Original DataFrame (original_df):** The unaltered dataset.
- **Binned DataFrame (binned_df):** The dataset after binning.
- **Analysis Parameters:**
  - `min_comb_size`: Minimum number of columns in a combination.
  - `max_comb_size`: Maximum number of columns in a combination.
  - `columns`: Specific columns to include in the analysis.

**Outputs**

- **Unique Identification Results (`unique_identifications.csv`):** CSV file listing combinations and their corresponding unique identification counts.
- **Plots:** Visual representations of the top combinations with the highest unique identifications.

**Pipeline Role**

Integrates into the post-binning analysis phase, providing insights into how different binning strategies impact the ability to uniquely identify records. This analysis is crucial for applications requiring high data granularity and uniqueness, such as classification tasks or anomaly detection.

---

### 8. utils.py

**Purpose**

`utils.py` comprises a collection of utility functions and configurations that support various operations throughout the application. These functions handle tasks like directory management, data loading and saving, plotting, and user interactions within the Streamlit interface.

**Key Functionalities**

- **Output Directory Management:** Creates necessary directories for storing outputs like processed data, reports, and plots.
- **Streamlit UI Enhancements:** Customizes the Streamlit interface by hiding default menus and footers for a cleaner look.
- **Data Loading:** Facilitates the loading of uploaded files into Pandas DataFrames.
- **Data Alignment:** Ensures that the original and binned DataFrames have consistent columns.
- **Binning Configuration UI:** Generates dynamic sliders for users to configure binning parameters.
- **Plotting Helpers:** Handles the generation and saving of entropy and density plots.
- **Download Handlers:** Manages the downloading of processed data and analysis results.
- **Integrity Assessment Handling:** Orchestrates the process of assessing data integrity post-binning.
- **Unique Identification Analysis Handling:** Manages the workflow for performing and displaying unique identification analyses.

**Inputs and Outputs**

Various: Each utility function has its specific inputs and outputs, such as DataFrames, file paths, user selections, and plot objects.

**Pipeline Role**

Acts as the support backbone of the application, providing essential functions that streamline data processing, user interactions, and result presentations. By encapsulating common tasks, `utils.py` ensures modularity and reusability across different components of the application.

## Additional Components

### A. `outputs` Directory

Houses all the generated outputs from the application, organized into subdirectories for clarity and ease of access.

- **category_mappings:** Contains CSV files mapping categorical codes to their original labels for each factor column.
- **plots:** Stores all generated plots, including density and entropy plots.
- **processed_data:** Contains the processed and binned datasets in the chosen formats (csv, pkl, etc.).
- **reports:** Holds various reports like integrity loss reports and type conversion reports.
- **unique_identifications:** Stores the results of unique identification analyses.

### B. `README.md`

Provides comprehensive documentation about the application, including setup instructions, usage guidelines, and descriptions of functionalities.

### C. `requirements.txt`

Lists all the Python dependencies required to run the application, ensuring that the environment is correctly set up for seamless execution.

## Pipeline Summary

1. **Data Ingestion**
   - **User Action:** Uploads a dataset via `Application.py`.
   - **Processing:** `utils.py`'s `load_data` function reads the uploaded file into a Pandas DataFrame.

2. **Data Processing & Type Conversion**
   - **Processing:** `Process_Data.py` utilizes `DtypeDetector` from `Detect_Dtypes.py` to detect and convert data types.
   - **Output:** Saves the processed DataFrame and generates type conversion reports.

3. **Binning Configuration & Execution**
   - **User Action:** Selects binning methods and columns within the Streamlit interface.
   - **Processing:** `data_binner.py`'s `DataBinner` bins the selected columns based on user configurations.
   - **Output:** Binned DataFrame and categorization of binned columns.

4. **Data Integrity Assessment**
   - **Processing:** `data_integrity_assessor.py` evaluates the entropy loss due to binning.
   - **Output:** Integrity loss reports and entropy plots.

5. **Visualization**
   - **Processing:** `density_plotter.py` generates density plots for both original and binned data.
   - **Output:** Density plots displayed within the application and saved for reference.

6. **Unique Identification Analysis**
   - **User Action:** Initiates analysis of unique identifications through the interface.
   - **Processing:** `unique_bin_identifier.py` analyzes combinations of binned columns to determine uniqueness.
   - **Output:** Analysis results displayed and saved as CSV and visual plots.

7. **Reporting & Downloading**
   - **User Action:** Downloads processed data, reports, and analysis results as needed.
   - **Processing:** `utils.py` manages download buttons and file preparations.

## Requirements

All dependencies are listed in the `requirements.txt` file. Ensure you have `Python 3.10.11`.

## License

This project is licensed under the [MIT License](LICENSE).

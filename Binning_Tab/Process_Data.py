# Process_Data.py

import pandas as pd
import os
from Binning_Tab.Detect_Dtypes import DtypeDetector  # Ensure dtype_detector.py is in the same directory or PYTHONPATH
import logging
from typing import Optional, Dict, Any, Tuple

class DataProcessor:
    def __init__(
        self,
        input_filepath: str = 'Data.csv',
        output_filepath: str = 'Processed_Data.csv',
        report_path: str = 'Type_Conversion_Report.csv',
        return_category_mappings: bool = True,
        mapping_directory: str = 'Category_Mappings',
        parallel_processing: bool = False,
        date_threshold: float = 0.6,
        numeric_threshold: float = 0.9,
        factor_threshold_ratio: float = 0.2,
        factor_threshold_unique: int = 500,
        dayfirst: bool = True,
        log_level: str = 'INFO',
        log_file: Optional[str] = None,
        convert_factors_to_int: bool = True,
        date_format: Optional[str] = '%d-%m-%Y',  # **New Parameter**
        save_type: str = 'csv' # **New Parameter**
    ):
        """
        Initialize the DataProcessor with file paths and configuration parameters.

        Parameters:
            ... [Existing Parameters] ...
            date_format (Optional[str]): Desired date format for output (e.g., '%d-%m-%Y'). 
                If specified, date columns will be formatted as strings in this format.
        """
        # Assigning file paths and configurations
        self.input_filepath = input_filepath
        self.output_filepath = output_filepath
        self.report_path = report_path
        self.return_category_mappings = return_category_mappings
        self.mapping_directory = mapping_directory
        self.parallel_processing = parallel_processing
        self.save_type = save_type

        # Initialize the DtypeDetector with provided thresholds and configurations
        self.detector = DtypeDetector(
            date_threshold=date_threshold,
            numeric_threshold=numeric_threshold,
            factor_threshold_ratio=factor_threshold_ratio,
            factor_threshold_unique=factor_threshold_unique,
            dayfirst=dayfirst,
            log_level=log_level,
            log_file=log_file,
            convert_factors_to_int=convert_factors_to_int,
            date_format=date_format  # **Pass the New Parameter**
        )

    def process(self):
        """
        Execute the data processing workflow:
            1. Read and clean the input CSV file.
            2. Determine and convert column data types.
            3. Generate and save a type conversion report.
            4. Save the processed data to the output CSV file.
            5. Optionally, save category mappings for 'factor' columns.
        """
        try:
            # Attempt to process the dataframe with the specified parallel processing option
            processed_data = self.detector.process_dataframe(
                filepath=self.input_filepath,
                use_parallel=self.parallel_processing,
                report_path=self.report_path
            )
            print(processed_data.dtypes)
        except Exception as e:
            # If parallel processing fails, attempt sequential processing
            print("Unable to use parallel processing, trying without parallel processing.")
            self.detector.logger.warning(f"Parallel processing failed: {e}")
            try:
                processed_data = self.detector.process_dataframe(
                    filepath=self.input_filepath,
                    use_parallel=False,
                    report_path=self.report_path
                )
            except Exception as e:
                # If sequential processing also fails, log the error and exit
                print(f"Error processing data: {e}")
                self.detector.logger.error(f"Error processing data: {e}")
                return


        if self.save_type == 'csv':
            #---------------------------------------------------------------------------
            # Save the processed data to the specified output CSV file
            try:
                processed_data.to_csv(self.output_filepath, index=False)
                self.detector.logger.info(f"Processed data saved to {self.output_filepath}")
            except Exception as e:
                self.detector.logger.error(f"Failed to save processed data: {e}")
                print(f"Failed to save processed data: {e}")
            #---------------------------------------------------------------------------

        elif self.save_type == 'pickle':
            #---------------------------------------------------------------------------
            # Save the processed data to the specified output Pickle file
            try:
                processed_data.to_pickle(self.output_filepath)
                self.detector.logger.info(f"Processed data saved to {self.output_filepath}")
            except Exception as e:
                self.detector.logger.error(f"Failed to save processed data: {e}")
                print(f"Failed to save processed data: {e}")
            #---------------------------------------------------------------------------

        elif self.save_type == 'parquet':
            #---------------------------------------------------------------------------
            # Save the processed data to the specified output Pickle file
            try:
                processed_data.to_parquet(self.output_filepath, index=False)
                self.detector.logger.info(f"Processed data saved to {self.output_filepath}")
            except Exception as e:
                self.detector.logger.error(f"Failed to save processed data: {e}")
                print(f"Failed to save processed data: {e}")
            #---------------------------------------------------------------------------

        # If configured to return category mappings, save them
        if self.return_category_mappings:
            self.save_category_mappings()
        

    def save_category_mappings(self):
        """
        Save category mappings for 'factor' columns to the specified directory.
        Each mapping is saved as a separate CSV file named '<Column_Name>_mapping.csv'.
        """
        try:
            # Create the mapping directory if it doesn't exist
            os.makedirs(self.mapping_directory, exist_ok=True)
            self.detector.logger.debug(f"Mapping directory '{self.mapping_directory}' is ready.")
        except Exception as e:
            self.detector.logger.error(f"Failed to create mapping directory '{self.mapping_directory}': {e}")
            print(f"Failed to create mapping directory '{self.mapping_directory}': {e}")
            return

        # Remove existing mapping files to avoid duplication or outdated mappings
        try:
            for file in os.listdir(self.mapping_directory):
                file_path = os.path.join(self.mapping_directory, file)
                if os.path.isfile(file_path):
                    os.remove(file_path)
                    self.detector.logger.debug(f"Removed existing mapping file '{file_path}'.")
        except Exception as e:
            self.detector.logger.error(f"Failed to clean mapping directory '{self.mapping_directory}': {e}")
            print(f"Failed to clean mapping directory '{self.mapping_directory}': {e}")
            return

        # Retrieve category mappings from the DtypeDetector
        category_mappings = self.detector.get_category_mapping()

        # Save each category mapping to a separate CSV file
        if category_mappings:
            for col, mapping in category_mappings.items():
                try:
                    # Convert the mapping dictionary to a DataFrame for easier CSV export
                    mapping_df = pd.DataFrame(list(mapping.items()), columns=['Code', 'Category'])
                    # Define the mapping file path
                    mapping_filepath = os.path.join(self.mapping_directory, f"{col}_mapping.csv")

                    # Save the mapping DataFrame to CSV
                    mapping_df.to_csv(mapping_filepath, index=False)
                    self.detector.logger.info(f"Category mapping for '{col}' saved to '{mapping_filepath}'")
                except Exception as e:
                    self.detector.logger.error(f"Failed to save category mapping for '{col}': {e}")
                    print(f"Failed to save category mapping for '{col}': {e}")
        else:
            self.detector.logger.info("No category mappings to save.")

# Detect_Dtypes.py

import pandas as pd
import numpy as np
import logging
from typing import Dict, Tuple, Optional
import sys
import concurrent.futures
import re


class DtypeDetector:
    def __init__(
        self,
        date_threshold: float = 0.5,
        numeric_threshold: float = 0.9,
        factor_threshold_ratio: float = 0.5,
        factor_threshold_unique: int = 50,
        dayfirst: bool = False,
        log_level: str = 'INFO',
        log_file: Optional[str] = None,
        convert_factors_to_int: bool = True,
        date_format: Optional[str] = None  # **New Parameter**
    ):
        """
        Initialize the DtypeDetector with configurable thresholds and logging.

        Parameters:
            date_threshold (float): Threshold for date detection.
            numeric_threshold (float): Threshold for numeric detection.
            factor_threshold_ratio (float): Threshold ratio for factor detection.
            factor_threshold_unique (int): Threshold for unique values in factor detection.
            dayfirst (bool): Whether to interpret the first value in dates as the day.
            log_level (str): Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
            log_file (Optional[str]): Path to save the log file. If None, logs are printed to stdout.
            convert_factors_to_int (bool): Whether to convert factors to integer codes. 
                If False, factors remain as categorical types with original string labels.
            date_format (Optional[str]): Desired date format (e.g., '%d-%m-%Y'). 
                If specified, date columns will be formatted as strings in this format.
        """
        self.convert_factors_to_int = convert_factors_to_int
        self.date_format = date_format  # **Store the New Parameter**
        self.thresholds = {
            'date_threshold': date_threshold,
            'numeric_threshold': numeric_threshold,
            'factor_threshold_ratio': factor_threshold_ratio,
            'factor_threshold_unique': factor_threshold_unique
        }
        self.dayfirst = dayfirst
        self.data_types: Dict[str, str] = {}
        self.series_mapping: Dict[str, Dict[int, str]] = {}

        # Configure logging
        log_handlers = [logging.StreamHandler(sys.stdout)]
        if log_file:
            log_handlers.append(logging.FileHandler(log_file))
        
        logging.basicConfig(
            level=getattr(logging, log_level.upper(), logging.INFO),
            format='%(asctime)s [%(levelname)s] %(message)s',
            handlers=log_handlers
        )
        self.logger = logging.getLogger(__name__)
    
    def clean_column_names(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Clean column names by removing any '/' or '\' characters.

        Parameters:
            data (pd.DataFrame): The DataFrame with original column names.

        Returns:
            pd.DataFrame: The DataFrame with cleaned column names.
        """
        original_columns = data.columns.tolist()
        cleaned_columns = []
        for col in original_columns:
            # Remove '/' and '\' from column names
            cleaned_col = re.sub(r'[\\/]', '', col)
            cleaned_columns.append(cleaned_col)
            if cleaned_col != col:
                self.logger.debug(f"Renamed column '{col}' to '{cleaned_col}'")
        data.columns = cleaned_columns
        return data

    def determine_column_type(
        self,
        series: pd.Series
    ) -> str:
        """
        Determine the type of a pandas Series.
        Returns 'int', 'float', 'date', 'factor', 'bool', or 'string'.
        """
        total = len(series)
        if total == 0:
            self.logger.debug(f"Column '{series.name}' is empty. Defaulting to 'string'.")
            return 'string'  # Default to string for empty columns

        num_unique = series.nunique(dropna=True)
        self.logger.debug(f"Column '{series.name}': Total={total}, Unique={num_unique}")

        # Attempt to convert to numeric first
        try:
            s_numeric = pd.to_numeric(series, errors='coerce')
            num_not_missing_numeric = s_numeric.notnull().sum()
            percent_numeric = num_not_missing_numeric / total
            self.logger.debug(f"Column '{series.name}': Numeric parse success rate: {percent_numeric:.2f}")
            if percent_numeric > self.thresholds['numeric_threshold']:
                # Check if all non-NaN values are integers within a tolerance
                if np.allclose(s_numeric.dropna(), s_numeric.dropna().astype(int), atol=1e-8):
                    return 'int'
                else:
                    return 'float'
        except Exception as e:
            self.logger.debug(f"Column '{series.name}': Numeric parsing failed: {e}")

        # Attempt to parse dates
        try:
            # Include time in date parsing by specifying formats that include time
            # Example formats: '%d/%m/%Y %H:%M', '%Y-%m-%d %H:%M', etc.
            # You can expand this list based on your data
            date_formats = [
                '%d/%m/%Y %H:%M',
                '%d/%m/%Y',
                '%Y-%m-%d %H:%M',
                '%Y-%m-%d',
                '%m/%d/%Y %H:%M',
                '%m/%d/%Y',
                '%d-%m-%Y %H:%M',
                '%d-%m-%Y',
                '%Y/%m/%d %H:%M',
                '%Y/%m/%d',
                '%Y-%m-%d %H:%M:%S%z',
                '%a %b %d %H:%M:%S %z %Y',
                '%a %b %d %H:%M:%S +0000 %Y'
            ]
            for fmt in date_formats:
                s_date = pd.to_datetime(series, errors='coerce', format=fmt, dayfirst=self.dayfirst)
                num_not_missing_date = s_date.notnull().sum()
                percent_date = num_not_missing_date / total
                self.logger.debug(f"Column '{series.name}': Date parse success rate with format '{fmt}': {percent_date:.2f}")
                if percent_date > self.thresholds['date_threshold']:
                    return 'date'
        except Exception as e:
            self.logger.debug(f"Column '{series.name}': Date parsing failed: {e}")

        # Check for boolean
        unique_values = set(series.dropna().unique())
        if unique_values <= {0, 1, '0', '1', 'True', 'False', 'true', 'false'}:
            return 'bool'

        # Check for categorical (factor) with AND condition
        try:
            if (num_unique / total) < self.thresholds['factor_threshold_ratio'] and num_unique < self.thresholds['factor_threshold_unique']:
                return 'factor'
        except Exception as e:
            self.logger.debug(f"Column '{series.name}': Factor determination failed: {e}")

        return 'string'

    def convert_series(self, series: pd.Series, dtype: str) -> pd.Series:
        """
        Convert a pandas Series to the specified dtype.
        """
        if dtype == 'date':
            dt_series = pd.to_datetime(series, errors='coerce', dayfirst=self.dayfirst)
            if self.date_format:
                # Format datetime as string in the specified format
                formatted_series = dt_series.dt.strftime(self.date_format)
                return formatted_series
            else:
                return dt_series
        elif dtype == 'factor':
            category = series.astype('category')
            # Store the mapping of codes to categories
            self.series_mapping[series.name] = dict(enumerate(category.cat.categories))
            if self.convert_factors_to_int:
                return category.cat.codes  # **Convert to integer codes**
            else:
                return category  # **Retain as categorical with string labels**
        elif dtype == 'int':
            return pd.to_numeric(series, errors='coerce').astype('Int64')
        elif dtype == 'float':
            return pd.to_numeric(series, errors='coerce')
        elif dtype == 'bool':
            # Map various representations of booleans to actual booleans
            return series.map({
                'True': True, 'False': False, 'true': True, 'false': False,
                1: True, 0: False
            }).astype('bool')
        else:
            return series.astype(str)

    def process_column(self, col: str, data: pd.DataFrame) -> Tuple[str, str, pd.Series]:
        """
        Process a single column: determine its type and convert it.
        Returns the column name, detected type, and converted series.
        """
        try:
            dtype = self.determine_column_type(data[col])
            converted_series = self.convert_series(data[col], dtype)
            self.logger.info(f"Column: {col}, Type Assessed: {dtype}, New Type: {converted_series.dtype}")
            return (col, dtype, converted_series)
        except Exception as e:
            self.logger.warning(f"Failed to process column '{col}': {e}")
            # Default to string if conversion fails
            converted_series = data[col].astype(str)
            self.logger.info(f"      New Type: {converted_series.dtype} (defaulted to string)")
            return (col, 'string', converted_series)

    def process_dataframe(
        self,
        filepath: str,
        use_parallel: bool = True,
        report_path: str = 'Type_Conversion_Report.csv'
    ) -> pd.DataFrame:
        """
        Read a CSV file, determine column types, convert columns accordingly, and generate a report.

        Parameters:
            filepath (str): Path to the input CSV file.
            use_parallel (bool): Whether to use parallel processing for columns.
            report_path (str): Path to save the type conversion report.

        Returns:
            pd.DataFrame: The processed DataFrame.
        """
        try:
            data = pd.read_csv(filepath, sep=',')  # Assuming comma deliminated values
            data = self.clean_column_names(data)
            self.logger.info(f"Successfully read file: {filepath}")
        except FileNotFoundError:
            self.logger.error(f"File not found: {filepath}")
            raise
        except pd.errors.EmptyDataError:
            self.logger.error("No data: The file is empty.")
            raise
        except Exception as e:
            self.logger.error(f"Error reading the file: {e}")
            raise

        data_types: Dict[str, str] = {}

        if use_parallel:
            with concurrent.futures.ThreadPoolExecutor() as executor:
                futures = {executor.submit(self.process_column, col, data): col for col in data.columns}
                for future in concurrent.futures.as_completed(futures):
                    col, dtype, converted_series = future.result()
                    data_types[col] = dtype
                    data[col] = converted_series
        else:
            for col in data.columns:
                col_name, dtype, converted_series = self.process_column(col, data)
                data_types[col_name] = dtype
                data[col_name] = converted_series

        # Generate type conversion report
        report = pd.DataFrame(list(data_types.items()), columns=['Column', 'Type'])
        report.to_csv(report_path, index=False)
        self.logger.info(f"Type conversion report saved to {report_path}")

        return data

    def get_category_mapping(self) -> Dict[str, Dict[int, str]]:
        """
        Get the mapping of categorical codes to original categories.

        Returns:
            Dict[str, Dict[int, str]]: Mapping for each categorical column.
        """
        return self.series_mapping

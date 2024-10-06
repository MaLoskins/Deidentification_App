# data_binner.py

import pandas as pd
from typing import Tuple, Dict, List
import math

class DataBinner:
    """
    A class to bin specified columns in a Pandas DataFrame based on provided bin counts and binning methods.
    """
    
    def __init__(self, Data: pd.DataFrame, method: str = 'equal width'):
        """
            Initializes the DataBinner with the original DataFrame and binning method.
        """
        self.original_df = Data.copy()
        self.binned_df = pd.DataFrame()
        self.binned_columns = {
            'datetime': [],
            'integer': [],
            'float': [],
            'category_grouped': [],
            'unsupported': []
        }
        self.method = method.lower()
        self._validate_method()
    
    def _validate_method(self):
        """
            Validates the binning method. Raises a ValueError if the method is unsupported.
        """
        supported_methods = ['equal width', 'quantile']
        if self.method not in supported_methods:
            raise ValueError(f"Unsupported binning method '{self.method}'. Supported methods are: {supported_methods}")
    
    def bin_columns(
        self,
        bin_dict: Dict[str, int]
    ) -> Tuple[pd.DataFrame, Dict[str, List[str]]]:
        """
            Bins specified columns in the DataFrame based on the provided bin counts and binning method.
        """
        # Initialize dictionary to categorize binned columns
        self.binned_columns = {
            'datetime': [],
            'integer': [],
            'float': [],
            'category_grouped': [],
            'unsupported': []
        }
        # Create a copy of the DataFrame to avoid modifying the original data
        Bin_Data = self.original_df.copy()

        for col, bins in bin_dict.items():
            if col not in Bin_Data.columns:
                print(f"⚠️ Column '{col}' does not exist in the DataFrame. Skipping.")
                continue

            try:
                if pd.api.types.is_datetime64_any_dtype(Bin_Data[col]):
                    # Binning datetime columns using pd.cut or pd.qcut based on method
                    Bin_Data[col], bin_labels = self._bin_column(Bin_Data[col], bins, self.method, is_datetime=True)
                    self.binned_columns['datetime'].append(col)
                    self.binned_columns[f'{col}_bins'] = bin_labels  # Store bin labels

                elif pd.api.types.is_integer_dtype(Bin_Data[col]):
                    Bin_Data[col], bin_labels = self._bin_column(Bin_Data[col], bins, self.method, is_datetime=False)
                    self.binned_columns['integer'].append(col)
                    self.binned_columns[f'{col}_bins'] = bin_labels

                elif pd.api.types.is_float_dtype(Bin_Data[col]):
                    Bin_Data[col], bin_labels = self._bin_column(Bin_Data[col], bins, self.method, is_datetime=False)
                    self.binned_columns['float'].append(col)
                    self.binned_columns[f'{col}_bins'] = bin_labels
                
                elif pd.api.types.is_categorical_dtype(Bin_Data[col]) or pd.api.types.is_object_dtype(Bin_Data[col]):
                    # Group categorical columns into specified number of bins
                    Bin_Data[col], category_groups = self._bin_categorical_column(Bin_Data[col], bins)
                    self.binned_columns['category_grouped'].append(col)
                    self.binned_columns[f'{col}_groups'] = category_groups
                else:
                    print(f"Column '{col}' has unsupported dtype '{Bin_Data[col].dtype}'. Skipping.")
                    self.binned_columns['unsupported'].append(col)

            except Exception as e:
                # Detailed error messages based on column type
                if pd.api.types.is_datetime64_any_dtype(Bin_Data[col]):
                    print(f"Failed to bin datetime column '{col}': {e}")
                elif pd.api.types.is_integer_dtype(Bin_Data[col]):
                    print(f"Failed to bin integer column '{col}': {e}")
                elif pd.api.types.is_float_dtype(Bin_Data[col]):
                    print(f"Failed to bin float column '{col}': {e}")
                elif pd.api.types.is_categorical_dtype(Bin_Data[col]) or pd.api.types.is_object_dtype(Bin_Data[col]):
                    print(f"Failed to bin category column '{col}': {e}")
                else:
                    print(f"Failed to bin column '{col}': {e}")
                self.binned_columns['unsupported'].append(col)

        # Retain only the successfully binned columns
        successfully_binned = (
            self.binned_columns['datetime'] +
            self.binned_columns['integer'] +
            self.binned_columns['float'] +
            self.binned_columns['category_grouped']
        )
        self.binned_df = Bin_Data[successfully_binned]

        return self.binned_df, self.binned_columns

    def _bin_column(self, series: pd.Series, bins: int, method: str, is_datetime: bool = False) -> Tuple[pd.Series, List[str]]:
        """
            Bins a single column using the specified method and returns labeled bins.

            Parameters:
                series (pd.Series): The column to bin.
                bins (int): The number of bins.
                method (str): The binning method ('equal width' or 'quantile').
                is_datetime (bool): Flag indicating if the column is datetime.

            Returns:
                Tuple[pd.Series, List[str]]: The binned column as a categorical Series and list of bin labels.
        """
        unique_values = series.nunique(dropna=True)
        print(f"Column '{series.name}': {unique_values} unique values requested {bins} bins.")

        if bins > unique_values:
            print(f"⚠️ Requested bin count {bins} exceeds unique values {unique_values} for column '{series.name}'. Adjusting bin count to {unique_values}.")
            bins = unique_values

        if bins == unique_values:
            # Assign each unique value to its own bin
            print(f"Assigning each unique value in column '{series.name}' to its own bin.")
            sorted_unique = series.dropna().unique()
            sorted_unique.sort()
            bin_labels = [f"[{x}]" if not is_datetime else f"[{x.strftime('%Y-%m-%d')}]"
                          for x in sorted_unique]
            value_to_bin = {value: label for value, label in zip(sorted_unique, bin_labels)}
            binned_series = series.map(value_to_bin).astype('category')
            return binned_series, bin_labels

        else:
            if method == 'equal width':
                binned, bins_edges = pd.cut(
                    series,
                    bins=bins,
                    retbins=True,
                    duplicates='drop'
                )
            elif method == 'quantile':
                if unique_values < bins:
                    print(f"⚠️ Not enough unique values for quantile binning on column '{series.name}'. Using {unique_values} bins instead of {bins}.")
                    bins = unique_values
                binned, bins_edges = pd.qcut(
                    series,
                    q=bins,
                    retbins=True,
                    duplicates='drop'
                )
            else:
                raise ValueError(f"Unsupported binning method '{method}'.")

            # Create descriptive bin labels
            bin_labels = []
            for i in range(len(bins_edges)-1):
                lower = bins_edges[i]
                upper = bins_edges[i+1]
                if is_datetime:
                    lower_str = pd.to_datetime(lower).strftime('%Y-%m-%d')
                    upper_str = pd.to_datetime(upper).strftime('%Y-%m-%d')
                else:
                    lower_str = f"{lower:.2f}" if not math.isclose(lower, int(lower)) else f"{int(lower)}"
                    upper_str = f"{upper:.2f}" if not math.isclose(upper, int(upper)) else f"{int(upper)}"
                label = f"[{lower_str} -> {upper_str})"
                bin_labels.append(label)
            
            # Assign labels to bins
            binned = pd.cut(
                series,
                bins=bins_edges,
                labels=bin_labels,
                include_lowest=True,
                duplicates='drop'
            )
            
            actual_bins = binned.nunique(dropna=True)
            print(f"Column '{series.name}': Requested {bins} bins, but got {actual_bins} unique binned values.")
            return binned.astype('category'), bin_labels

    def _bin_categorical_column(self, series: pd.Series, bins: int) -> Tuple[pd.Series, List[str]]:
        """
            Groups categorical column into specified number of bins based on frequency.

            Parameters:
                series (pd.Series): The categorical column to bin.
                bins (int): The desired number of bins (groups).

            Returns:
                Tuple[pd.Series, List[str]]: The binned categorical column and list of combined category names.
        """
        if not pd.api.types.is_categorical_dtype(series):
            series = series.astype('category')

        category_counts = series.value_counts().sort_values(ascending=False)
        unique_categories = len(category_counts)

        if bins >= unique_categories:
            print(f"⚠️ Requested bin count {bins} is greater than or equal to unique categories {unique_categories} for column '{series.name}'. No binning applied.")
            return series, [cat for cat in category_counts.index]

        # Calculate the number of categories per bin
        categories_per_bin = math.ceil(unique_categories / bins)
        grouped_categories = []
        current_bin = 1
        bin_to_categories = {}
        for i, category in enumerate(category_counts.index):
            if (i > 0) and (i % categories_per_bin == 0) and (current_bin < bins):
                current_bin += 1
            grouped_categories.append((category, current_bin))
            if current_bin not in bin_to_categories:
                bin_to_categories[current_bin] = []
            bin_to_categories[current_bin].append(category)

        # Create combined category names
        combined_names = {}
        for bin_num, cats in bin_to_categories.items():
            combined_name = '-'.join(cats)
            for cat in cats:
                combined_names[cat] = combined_name

        # Map original categories to combined names
        binned_series = series.map(combined_names).astype('category')

        # List of combined category names
        combined_category_names = list(bin_to_categories.keys())
        combined_category_names = [ '-'.join(bin_to_categories[bin_num]) for bin_num in sorted(bin_to_categories.keys()) ]

        return binned_series, combined_category_names

    def get_binned_data(self) -> pd.DataFrame:
        """
            Retrieves the binned DataFrame.
        """
        return self.binned_df.copy()

    def get_binned_columns(self) -> Dict[str, List[str]]:
        """
            Retrieves the categorization of binned columns by data type.
        """
        return self.binned_columns.copy()

# src/binning_optimizer.py

import pandas as pd
import numpy as np
import random
import math
import warnings
import matplotlib.pyplot as plt
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
import os
from itertools import combinations
from typing import Tuple, Dict, List, Optional, Callable
from multiprocessing import Manager

from scipy.stats import wasserstein_distance  # Required for t-closeness

warnings.filterwarnings("ignore")

# Global variables for worker processes
global_original_data = None
global_k = None
global_privacy_model = None
global_sensitive_attributes = None
global_l = None
global_t = None

def initialize_worker(original_data, k, privacy_model, sensitive_attributes, l, t):
    """
    Initializes global variables for worker processes.

    Parameters:
        original_data (pd.DataFrame): The original DataFrame.
        k (int): Desired k-anonymity level.
        privacy_model (str): The privacy model ('k_anonymity', 'l_diversity', 't_closeness').
        sensitive_attributes (list): List of sensitive attribute column names.
        l (int): Desired l-diversity level.
        t (float): Desired t-closeness threshold.
    """
    global global_original_data
    global global_k
    global global_privacy_model
    global global_sensitive_attributes
    global global_l
    global global_t
    global_original_data = original_data
    global_k = k
    global_privacy_model = privacy_model
    global_sensitive_attributes = sensitive_attributes
    global_l = l
    global_t = t

def calculate_l_diversity(binned_df: pd.DataFrame, sensitive_attrs: List[str], l: int) -> int:
    """
    Calculates the l-diversity penalty for the binned DataFrame.

    Parameters:
        binned_df (pd.DataFrame): The binned DataFrame.
        sensitive_attrs (list): List of sensitive attribute column names.
        l (int): Desired l-diversity level.

    Returns:
        int: Total l-diversity penalty.
    """
    penalties = 0
    for bin_group, group in binned_df.groupby(list(binned_df.columns)):
        for attr in sensitive_attrs:
            unique_values = group[attr].nunique()
            if unique_values < l:
                penalties += (l - unique_values)
    return penalties

def calculate_t_closeness(binned_df: pd.DataFrame, sensitive_attrs: List[str], t: float, original_data: pd.DataFrame) -> float:
    """
    Calculates the t-closeness penalty for the binned DataFrame.

    Parameters:
        binned_df (pd.DataFrame): The binned DataFrame.
        sensitive_attrs (list): List of sensitive attribute column names.
        t (float): Desired t-closeness threshold.
        original_data (pd.DataFrame): The original DataFrame.

    Returns:
        float: Total t-closeness penalty.
    """
    penalties = 0.0
    overall_distributions = {}
    for attr in sensitive_attrs:
        overall_distributions[attr] = original_data[attr].value_counts(normalize=True)

    for bin_group, group in binned_df.groupby(list(binned_df.columns)):
        for attr in sensitive_attrs:
            bin_distribution = group[attr].value_counts(normalize=True)
            # Align the distributions
            combined = pd.concat([overall_distributions[attr], bin_distribution], axis=1).fillna(0)
            distance = wasserstein_distance(combined.iloc[:,0], combined.iloc[:,1])
            
            if distance > t:
                penalties += (distance - t)
    return penalties

def evaluate_fitness(bin_dict):
    """
    Evaluates the fitness of a binning configuration based on the selected privacy model.

    Parameters:
        bin_dict (dict): A dictionary mapping columns to bin counts.

    Returns:
        float: The fitness score (lower is better).
    """
    try:
        # Bin the data using the global original data
        binned_df, _ = bin_columns(bin_dict, global_original_data)

        # Calculate base fitness based on k-anonymity
        small_groups = find_small_groups(binned_df, global_k)

        total_fitness = small_groups

        # Add penalties based on the privacy model
        if global_privacy_model == "l_diversity":
            l_diversity_penalty = calculate_l_diversity(binned_df, global_sensitive_attributes, global_l)
            total_fitness += l_diversity_penalty
        elif global_privacy_model == "t_closeness":
            t_closeness_penalty = calculate_t_closeness(binned_df, global_sensitive_attributes, global_t, global_original_data)
            total_fitness += t_closeness_penalty

        return total_fitness
    except Exception as e:
        print(f"⚠️ Error during fitness evaluation for bin_dict {bin_dict}: {e}")
        return np.inf

def _bin_categorical_column(series: pd.Series, bins: int) -> Tuple[pd.Series, List[str]]:
    """
    Bins a categorical column by grouping infrequent categories.

    Parameters:
        series (pd.Series): The categorical column to bin.
        bins (int): The number of bins (groups) to create.

    Returns:
        Tuple[pd.Series, List[str]]: The binned categorical series and list of group labels.
    """
    # Calculate category frequencies
    freq = series.value_counts().sort_values(ascending=False)
    # Initialize groups
    groups = {}
    current_group = []
    current_bins = 1

    for category, count in freq.items():
        current_group.append(category)
        if len(current_group) >= math.ceil(len(freq) / bins):
            groups[f'Group_{current_bins}'] = current_group
            current_group = []
            current_bins += 1
            if current_bins > bins:
                break

    # Assign remaining categories to the last group
    if current_group:
        if f'Group_{current_bins}' in groups:
            groups[f'Group_{current_bins}'].extend(current_group)
        else:
            groups[f'Group_{current_bins}'] = current_group

    # Create a mapping from category to group label
    category_to_group = {}
    for group_label, categories in groups.items():
        for category in categories:
            category_to_group[category] = group_label

    # Map the series to group labels
    binned_series = series.map(category_to_group).astype('category')
    bin_labels = list(groups.keys())

    return binned_series, bin_labels

def bin_columns(bin_dict: Dict[str, int], original_data: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, List[str]]]:
    """
    Bins specified columns in the DataFrame based on the provided bin counts and binning method.

    Parameters:
        bin_dict (dict): A dictionary mapping columns to bin counts.
        original_data (pd.DataFrame): The original DataFrame to bin.

    Returns:
        Tuple[pd.DataFrame, Dict[str, List[str]]]: The binned DataFrame and a dictionary of binned columns.
    """
    # Initialize dictionary to categorize binned columns
    binned_columns = {
        'datetime': [],
        'integer': [],
        'float': [],
        'category_grouped': [],
        'unsupported': []
    }
    # Create a copy of the DataFrame to avoid modifying the original data
    Bin_Data = original_data.copy()

    for col, bins in bin_dict.items():
        if col not in Bin_Data.columns:
            print(f"⚠️ Column '{col}' does not exist in the DataFrame. Skipping.")
            continue

        try:
            if pd.api.types.is_datetime64_any_dtype(Bin_Data[col]):
                # Binning datetime columns using pd.cut or pd.qcut based on method
                binned_series, bin_labels = _bin_column(Bin_Data[col], bins, "quantile", is_datetime=True)
                Bin_Data[col] = binned_series
                binned_columns['datetime'].append(col)
                binned_columns[f'{col}_bins'] = bin_labels  # Store bin labels

            elif pd.api.types.is_integer_dtype(Bin_Data[col]):
                binned_series, bin_labels = _bin_column(Bin_Data[col], bins, "quantile", is_datetime=False)
                Bin_Data[col] = binned_series
                binned_columns['integer'].append(col)
                binned_columns[f'{col}_bins'] = bin_labels

            elif pd.api.types.is_float_dtype(Bin_Data[col]):
                binned_series, bin_labels = _bin_column(Bin_Data[col], bins, "quantile", is_datetime=False)
                Bin_Data[col] = binned_series
                binned_columns['float'].append(col)
                binned_columns[f'{col}_bins'] = bin_labels

            elif pd.api.types.is_categorical_dtype(Bin_Data[col]) or pd.api.types.is_object_dtype(Bin_Data[col]):
                # Group categorical columns into specified number of bins
                binned_series, category_groups = _bin_categorical_column(Bin_Data[col], bins)
                Bin_Data[col] = binned_series
                binned_columns['category_grouped'].append(col)
                binned_columns[f'{col}_groups'] = category_groups
            else:
                print(f"⚠️ Column '{col}' has unsupported dtype '{Bin_Data[col].dtype}'. Skipping.")
                binned_columns['unsupported'].append(col)

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
            binned_columns['unsupported'].append(col)

    # Retain only the successfully binned columns
    successfully_binned = (
        binned_columns['datetime'] +
        binned_columns['integer'] +
        binned_columns['float'] +
        binned_columns['category_grouped']
    )
    binned_df = Bin_Data[successfully_binned]

    return binned_df, binned_columns

def _bin_column(series: pd.Series, bins: int, method: str, is_datetime: bool = False) -> Tuple[pd.Series, List[str]]:
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

    if bins > unique_values:
        print(f"⚠️ Requested bin count {bins} exceeds unique values {unique_values} for column '{series.name}'. Adjusting bin count to {unique_values}.")
        bins = unique_values

    if bins == unique_values:
        # Assign each unique value to its own bin
        sorted_unique = series.dropna().unique()
        sorted_unique.sort()
        bin_labels = [f"[{x}]" if not is_datetime else f"[{x.strftime('%Y-%m-%d')}]"
                      for x in sorted_unique]
        value_to_bin = {value: label for value, label in zip(sorted_unique, bin_labels)}
        binned_series = series.map(value_to_bin).astype('category')
        return binned_series, bin_labels

    else:
        if method == 'equal width':
            try:
                binned, bins_edges = pd.cut(
                    series,
                    bins=bins,
                    retbins=True,
                    duplicates='drop'
                )
            except Exception as e:
                print(f"⚠️ Error during equal width binning for column '{series.name}': {e}")
                binned, bins_edges = pd.cut(
                    series,
                    bins=unique_values,
                    retbins=True,
                    duplicates='drop'
                )
        elif method == 'quantile':
            if unique_values < bins:
                print(f"⚠️ Not enough unique values for quantile binning on column '{series.name}'. Using {unique_values} bins instead of {bins}.")
                bins = unique_values
            try:
                binned, bins_edges = pd.qcut(
                    series,
                    q=bins,
                    retbins=True,
                    duplicates='drop'
                )
            except Exception as e:
                print(f"⚠️ Error during quantile binning for column '{series.name}': {e}")
                binned, bins_edges = pd.qcut(
                    series,
                    q=unique_values,
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
        try:
            binned = pd.cut(
                series,
                bins=bins_edges,
                labels=bin_labels,
                include_lowest=True,
                duplicates='drop'
            )
        except Exception as e:
            print(f"⚠️ Error assigning labels for column '{series.name}': {e}")
            binned = series.astype('category')

        actual_bins = binned.nunique(dropna=True)
        return binned.astype('category'), bin_labels

def find_small_groups(binned_df: pd.DataFrame, k: int) -> int:
    """
    Identifies the number of small groups in the binned DataFrame based on k-anonymity.

    Parameters:
        binned_df (pd.DataFrame): The binned DataFrame.
        k (int): Desired k-anonymity level.

    Returns:
        int: Number of small groups.
    """
    # Assuming that the combination to check is all columns
    group_counts = binned_df.groupby(list(binned_df.columns)).size()
    small_groups = (group_counts < k).sum()
    return small_groups

class BinningOptimizer:
    """
    A class to perform k-anonymity, l-diversity, or t-closeness binning using either Genetic Algorithm or Simulated Annealing.

    Attributes:
        original_data (pd.DataFrame): The original dataset.
        k (int): Desired k-anonymity level.
        min_comb_size (int): Minimum combination size of columns to consider.
        max_comb_size (int): Maximum combination size of columns to consider.
        columns (list): List of columns to bin.
        min_bins_per_column (dict): Minimum number of bins per column.
        max_bins_per_column (dict): Maximum number of bins per column.
        max_iterations (int): Maximum number of iterations for random sampling.
        optimizer (str): Optimization method to use ('genetic' or 'simulated_annealing').
        method (str): Binning method ('quantile' or 'equal width').
        privacy_model (str): Privacy model to enforce ('k_anonymity', 'l_diversity', 't_closeness').
        sensitive_attributes (list): List of sensitive attribute column names.
        l (int): Desired l-diversity level.
        t (float): Desired t-closeness threshold.
    """

    def __init__(
        self,
        original_data: pd.DataFrame,
        k: int,
        privacy_model: str = "k_anonymity",
        sensitive_attributes: Optional[List[str]] = None,
        l: Optional[int] = None,
        t: Optional[float] = None,
        min_comb_size: int = 1,
        max_comb_size: Optional[int] = None,
        columns: Optional[List[str]] = None,
        min_bins_per_column: Optional[Dict[str, int]] = None,
        max_bins_per_column: Optional[Dict[str, int]] = None,
        max_iterations: int = 1000,
        optimizer: str = "genetic",
        method: str = "quantile",
    ):
        """
        Initializes the BinningOptimizer with the dataset and parameters.

        Parameters:
            original_data (pd.DataFrame): The original full DataFrame.
            k (int): Desired k-anonymity level.
            privacy_model (str, optional): Privacy model to enforce ('k_anonymity', 'l_diversity', 't_closeness').
            sensitive_attributes (list, optional): List of sensitive attribute column names.
            l (int, optional): Desired l-diversity level.
            t (float, optional): Desired t-closeness threshold.
            min_comb_size (int, optional): Minimum combination size of columns to consider.
            max_comb_size (int, optional): Maximum combination size of columns to consider.
            columns (list, optional): List of columns to consider. Defaults to all columns.
            min_bins_per_column (dict, optional): Minimum number of bins per column.
            max_bins_per_column (dict, optional): Maximum number of bins per column.
            max_iterations (int, optional): Maximum number of iterations to try in random sampling.
            optimizer (str, optional): Optimization method to use ('genetic' or 'simulated_annealing').
            method (str, optional): Binning method to use ('quantile' or 'equal width').
        """
        self.fitness_cache = {}  # Initialize fitness cache (non-managed for efficiency)
        self.original_data = original_data.copy()
        self.k = k
        self.privacy_model = privacy_model.lower()
        self.sensitive_attributes = sensitive_attributes if sensitive_attributes else []
        self.l = l
        self.t = t
        self.min_comb_size = min_comb_size
        self.columns = columns if columns is not None else original_data.columns.tolist()
        self.max_comb_size = max_comb_size if max_comb_size is not None else len(self.columns)
        self.min_bins_per_column = min_bins_per_column if min_bins_per_column else {}
        self.max_bins_per_column = max_bins_per_column if max_bins_per_column else {}
        self.max_iterations = max_iterations
        self.optimizer = optimizer.lower()
        self.method = method.lower()
        self.times = []  # To track time taken per iteration
        self.fitness_history = []  # To track fitness per iteration
        self.best_bin_dict = None
        self.best_binned_df = None
        self.best_fitness = np.inf

        # Initialize a Manager for shared state
        self.manager = Manager()
        self.progress = self.manager.Value('i', 0)  # Shared integer for progress percentage

        # Validate inputs
        self._validate_inputs()
        # Define possible bins per column
        self.possible_bins_per_column = self._define_possible_bins()

    def _validate_inputs(self):
        """
        Validates input parameters for correctness.
        """
        if not isinstance(self.k, int) or self.k < 1:
            raise ValueError("Parameter 'k' must be an integer greater than or equal to 1.")

        if self.privacy_model not in ['k_anonymity', 'l_diversity', 't_closeness']:
            raise ValueError("Unsupported privacy model. Choose 'k_anonymity', 'l_diversity', or 't_closeness'.")

        if self.privacy_model == 'l_diversity':
            if not isinstance(self.l, int) or self.l < 1:
                raise ValueError("Parameter 'l' must be an integer greater than or equal to 1 for l-diversity.")
            if not self.sensitive_attributes:
                raise ValueError("Parameter 'sensitive_attributes' must be provided for l-diversity.")

        if self.privacy_model == 't_closeness':
            if not isinstance(self.t, (int, float)) or self.t <= 0:
                raise ValueError("Parameter 't' must be a positive number for t-closeness.")
            if not self.sensitive_attributes:
                raise ValueError("Parameter 'sensitive_attributes' must be provided for t-closeness.")

        if not isinstance(self.min_comb_size, int) or self.min_comb_size < 1:
            raise ValueError("Parameter 'min_comb_size' must be an integer greater than or equal to 1.")

        if self.max_comb_size is not None:
            if not isinstance(self.max_comb_size, int) or self.max_comb_size < self.min_comb_size:
                raise ValueError("Parameter 'max_comb_size' must be an integer greater than or equal to 'min_comb_size'.")

        if not set(self.columns).issubset(set(self.original_data.columns)):
            missing_cols = set(self.columns) - set(self.original_data.columns)
            raise ValueError(f"The following columns are not in the original DataFrame: {missing_cols}")

        if self.optimizer not in ['genetic', 'simulated_annealing']:
            raise ValueError("Unsupported optimizer. Choose 'genetic' or 'simulated_annealing'.")

        if not isinstance(self.max_iterations, int) or self.max_iterations < 1:
            raise ValueError("Parameter 'max_iterations' must be an integer greater than or equal to 1.")

        if self.method not in ['quantile', 'equal width']:
            raise ValueError("Unsupported binning method. Choose 'quantile' or 'equal width'.")

    def _define_possible_bins(self):
        """
        Defines possible bin counts for each column based on min and max bins.

        Returns:
            dict: A dictionary with column names as keys and lists of possible bin counts as values.
        """
        possible_bins_per_column = {}
        for col in self.columns:
            unique_values = self.original_data[col].nunique(dropna=True)
            print(f"Column '{col}' has {unique_values} unique values.")

            if unique_values == 0:
                # No unique values, set bins to 1
                print(f"⚠️ Column '{col}' has no unique values. Skipping.")
                possible_bins_per_column[col] = [1]
                continue

            # Set default min and max bins per column if not provided
            min_bins = self.min_bins_per_column.get(col, max(1, min(2, unique_values // 10)))
            max_bins = self.max_bins_per_column.get(col, unique_values)

            # Ensure min_bins and max_bins do not exceed the number of unique values
            max_possible_bins = unique_values
            min_bins = max(1, min(min_bins, max_possible_bins))
            max_bins = max(1, min(max_bins, max_possible_bins))

            # Ensure min_bins <= max_bins
            if min_bins > max_bins:
                min_bins, max_bins = max_bins, min_bins

            # Create list of possible bin counts for this column
            possible_bins = list(range(min_bins, max_bins + 1))
            possible_bins_per_column[col] = possible_bins
            print(f"Possible bins for column '{col}': {possible_bins}")

        return possible_bins_per_column

    def fitness_function(self, bin_dict: Dict[str, int]) -> float:
        """
        Evaluates the fitness of a binning configuration with caching.

        Parameters:
            bin_dict (dict): A dictionary mapping columns to bin counts.

        Returns:
            float: The fitness score (lower is better).
        """
        # Convert bin_dict to a hashable tuple
        bin_dict_tuple = tuple(sorted(bin_dict.items()))
        if bin_dict_tuple in self.fitness_cache:
            return self.fitness_cache[bin_dict_tuple]

        try:
            # Bin the data
            binned_df, _ = bin_columns(bin_dict, self.original_data)

            # Find small groups based on k-anonymity
            small_groups = find_small_groups(binned_df, self.k)

            total_fitness = small_groups

            # Add penalties based on the privacy model
            if self.privacy_model == "l_diversity":
                l_diversity_penalty = calculate_l_diversity(binned_df, self.sensitive_attributes, self.l)
                total_fitness += l_diversity_penalty
            elif self.privacy_model == "t_closeness":
                t_closeness_penalty = calculate_t_closeness(binned_df, self.sensitive_attributes, self.t, self.original_data)
                total_fitness += t_closeness_penalty

            # Update the best fitness and bin_dict if improved
            if total_fitness < self.best_fitness:
                self.best_fitness = total_fitness
                self.best_bin_dict = bin_dict.copy()

            # Cache the result
            self.fitness_cache[bin_dict_tuple] = total_fitness

            return total_fitness

        except Exception as e:
            print(f"⚠️ Error during fitness evaluation for bin_dict {bin_dict}: {e}")
            # Assign a high fitness value if an error occurs
            self.fitness_cache[bin_dict_tuple] = np.inf
            return np.inf

    def genetic_algorithm(
        self, population_size: int = 50, generations: int = 100, mutation_rate: float = 0.1, progress_callback: Optional[Callable[[int], None]] = None
    ) -> Dict[str, int]:
        """
        Performs the Genetic Algorithm to find the best binning configuration.

        Parameters:
            population_size (int, optional): Number of individuals in the population.
            generations (int, optional): Number of generations to evolve.
            mutation_rate (float, optional): Probability of mutation.
            progress_callback (Callable, optional): Function to call with progress percentage.

        Returns:
            dict: The best binning configuration found.
        """
        def crossover(parent1, parent2):
            # Single-point crossover
            crossover_point = random.randint(1, len(parent1) - 1)
            child = {}
            for i, col in enumerate(parent1.keys()):
                if i < crossover_point:
                    child[col] = parent1[col]
                else:
                    child[col] = parent2[col]
            return child

        def mutate(individual):
            # Randomly change the bin count of one column
            col = random.choice(list(individual.keys()))
            individual[col] = random.choice(self.possible_bins_per_column[col])

        # Initialize population
        population = []
        for _ in range(population_size):
            bin_dict = {
                col: random.choice(self.possible_bins_per_column[col]) for col in self.columns
            }
            population.append(bin_dict)

        # Determine number of workers
        max_workers = min(8, os.cpu_count() or 1)  # Limit to 8 or available CPUs

        with ProcessPoolExecutor(max_workers=max_workers, initializer=initialize_worker, 
                                 initargs=(self.original_data, self.k, self.privacy_model, self.sensitive_attributes, self.l, self.t)) as executor:
            for generation in range(generations):
                start_time = time.time()
                # Evaluate fitness in parallel
                fitness_scores = list(executor.map(evaluate_fitness, population))

                # Pair fitness scores with individuals
                fitness_individuals = list(zip(fitness_scores, population))

                # Sort by fitness (lower is better)
                fitness_individuals.sort(key=lambda x: x[0])
                best_fitness, best_individual = fitness_individuals[0]
                self.fitness_history.append(best_fitness)
                print(f"Generation {generation + 1}/{generations}: Best Fitness = {best_fitness}")

                # Update best_fitness and best_bin_dict
                if best_fitness < self.best_fitness:
                    self.best_fitness = best_fitness
                    self.best_bin_dict = best_individual.copy()

                # Report progress
                if progress_callback:
                    progress_percentage = int(((generation + 1) / generations) * 100)
                    progress_callback(progress_percentage)

                # Check for optimal solution
                if best_fitness == 0:
                    print("✅ Optimal solution found via Genetic Algorithm.")
                    end_time = time.time()
                    self.times.append(end_time - start_time)
                    return best_individual

                # Selection (e.g., top 50%)
                selected = [ind for _, ind in fitness_individuals[: population_size // 2]]

                # Crossover to produce offspring
                offspring = []
                while len(offspring) < population_size - len(selected):
                    parent1, parent2 = random.sample(selected, 2)
                    child = crossover(parent1, parent2)
                    offspring.append(child)

                # Mutation
                for individual in offspring:
                    if random.random() < mutation_rate:
                        mutate(individual)

                # New population
                population = selected + offspring
                end_time = time.time()
                self.times.append(end_time - start_time)

        print("❌ Optimal solution not found via Genetic Algorithm within the given generations.")
        print(f"Best Fitness Achieved: {self.best_fitness}")
        return self.best_bin_dict

    def simulated_annealing(
        self,
        initial_temperature: float = 1000,
        cooling_rate: float = 0.95,
        iterations: int = 1000,
        neighbors_per_iteration: int = 5,  # Number of neighbors to evaluate in parallel
        progress_callback: Optional[Callable[[int], None]] = None
    ) -> Dict[str, int]:
        """
        Performs Simulated Annealing to find the best binning configuration with parallel neighbor evaluations.

        Parameters:
            initial_temperature (float, optional): Starting temperature.
            cooling_rate (float, optional): Rate at which the temperature decreases.
            iterations (int, optional): Number of iterations to perform.
            neighbors_per_iteration (int, optional): Number of neighbors to evaluate in parallel each iteration.
            progress_callback (Callable, optional): Function to call with progress percentage.

        Returns:
            dict: The best binning configuration found.
        """
        # Start with a random solution
        current_solution = {
            col: random.choice(self.possible_bins_per_column[col]) for col in self.columns
        }
        current_fitness = self.fitness_function(current_solution)
        best_solution = current_solution.copy()
        best_fitness = current_fitness
        temperature = initial_temperature

        max_workers = min(8, os.cpu_count() or 1)

        with ProcessPoolExecutor(max_workers=max_workers, initializer=initialize_worker, 
                                 initargs=(self.original_data, self.k, self.privacy_model, self.sensitive_attributes, self.l, self.t)) as executor:
            for iteration in range(iterations):
                start_time = time.time()
                neighbors = []
                for _ in range(neighbors_per_iteration):
                    neighbor = current_solution.copy()
                    col = random.choice(list(neighbor.keys()))
                    neighbor[col] = random.choice(self.possible_bins_per_column[col])
                    neighbors.append(neighbor)

                # Evaluate fitness of all neighbors in parallel
                fitness_results = list(executor.map(evaluate_fitness, neighbors))

                # Find the best neighbor
                min_fitness = min(fitness_results)
                min_index = fitness_results.index(min_fitness)
                best_neighbor = neighbors[min_index]

                # Decide whether to accept the best neighbor
                if (
                    min_fitness < current_fitness
                    or random.uniform(0, 1) < math.exp((current_fitness - min_fitness) / max(temperature, 1e-10))
                ):
                    current_solution = best_neighbor
                    current_fitness = min_fitness

                    # Update the best solution found
                    if current_fitness < best_fitness:
                        best_solution = current_solution.copy()
                        best_fitness = current_fitness

                # Cool down the temperature
                temperature *= cooling_rate
                self.fitness_history.append(best_fitness)
                print(f"Iteration {iteration + 1}/{iterations}: Best Fitness = {best_fitness}")

                # Report progress
                if progress_callback:
                    progress_percentage = int(((iteration + 1) / iterations) * 100)
                    progress_callback(progress_percentage)

                # Terminate early if optimal solution is found
                if best_fitness == 0:
                    print("✅ Optimal solution found via Simulated Annealing.")
                    end_time = time.time()
                    self.times.append(end_time - start_time)
                    break
                end_time = time.time()
                self.times.append(end_time - start_time)

        if best_fitness != 0:
            print("❌ Optimal solution not found via Simulated Annealing within the given iterations.")
            print(f"Best Fitness Achieved: {best_fitness}")

        self.best_fitness = best_fitness
        self.best_bin_dict = best_solution.copy()
        return best_solution

    def find_best_binned_data(self) -> Tuple[Optional[Dict[str, int]], Optional[pd.DataFrame]]:
        """
        Finds the best binning configuration to achieve the selected privacy model using the chosen optimizer.

        Returns:
            Tuple[Dict[str, int], pd.DataFrame] or (None, None): The bin dictionary and the binned DataFrame if found; otherwise, (None, None).
        """
        # Random Sampling Phase
        print("🔍 Starting Random Sampling Phase...")
        bin_dict, binned_df = self._random_sampling_phase(progress_callback=self.update_progress)
        if bin_dict:
            self.best_bin_dict = bin_dict.copy()
            self.best_binned_df = binned_df.copy()
            return bin_dict, binned_df

        # Optimization Phase
        print(f"🚀 Starting Optimization Phase using {self.optimizer.capitalize()}...")
        if self.optimizer == "genetic":
            best_bin_dict = self.genetic_algorithm(progress_callback=self.update_progress)
        elif self.optimizer == "simulated_annealing":
            best_bin_dict = self.simulated_annealing(progress_callback=self.update_progress)
        else:
            raise ValueError("Unsupported optimizer. Choose 'genetic' or 'simulated_annealing'.")

        # Bin the data using the best bin dict found
        binned_df, _ = bin_columns(best_bin_dict, self.original_data)
        self.best_binned_df = binned_df.copy()

        return best_bin_dict, binned_df

    def _random_sampling_phase(self, progress_callback: Optional[Callable[[int], None]] = None) -> Tuple[Optional[Dict[str, int]], Optional[pd.DataFrame]]:
        """
        Performs random sampling to find a binning configuration that satisfies the selected privacy model.

        Parameters:
            progress_callback (Callable, optional): Function to call with progress percentage.

        Returns:
            Tuple[Dict[str, int], pd.DataFrame] or (None, None): The bin dictionary and the binned DataFrame if found; otherwise, (None, None).
        """
        total_combinations = np.prod(
            [len(self.possible_bins_per_column[col]) for col in self.columns]
        )
        print(f"Total possible combinations: {total_combinations}")

        max_iterations = min(self.max_iterations, total_combinations)
        print(f"Sampling up to {max_iterations} random combinations.")

        tried_bin_dicts = set()
        progress_interval = 100  # Define how often to print progress

        # Define the number of workers
        max_workers = min(8, os.cpu_count() or 1)  # Adjust as needed

        with ProcessPoolExecutor(max_workers=max_workers, initializer=initialize_worker, 
                                 initargs=(self.original_data, self.k, self.privacy_model, self.sensitive_attributes, self.l, self.t)) as executor:
            # Prepare a generator for random bin_dicts
            def generate_bin_dicts():
                while len(tried_bin_dicts) < max_iterations:
                    bin_dict = {
                        col: random.choice(self.possible_bins_per_column[col]) for col in self.columns
                    }
                    bin_dict_tuple = tuple(sorted(bin_dict.items()))
                    if bin_dict_tuple in tried_bin_dicts:
                        continue
                    tried_bin_dicts.add(bin_dict_tuple)
                    yield bin_dict

            # Process in chunks for efficiency
            chunk_size = 100  # Adjust based on memory and performance
            bin_dict_generator = generate_bin_dicts()
            bin_dicts_processed = 0

            while True:
                bin_dicts = []
                try:
                    for _ in range(chunk_size):
                        bin_dicts.append(next(bin_dict_generator))
                except StopIteration:
                    pass

                if not bin_dicts:
                    break

                # Evaluate fitness in parallel
                fitness_results = list(executor.map(evaluate_fitness, bin_dicts))

                for bin_dict, fitness in zip(bin_dicts, fitness_results):
                    bin_dicts_processed += 1

                    if fitness == 0:
                        # Bin the data
                        binned_df, _ = bin_columns(bin_dict, self.original_data)
                        print(f"✅ Desired privacy model achieved with bin_dict: {bin_dict}")
                        self.best_fitness = fitness
                        return bin_dict, binned_df
                    else:
                        if fitness < self.best_fitness:
                            self.best_fitness = fitness
                            self.best_bin_dict = bin_dict.copy()

                    # Periodic updates
                    if bin_dicts_processed % progress_interval == 0:
                        print(f"🔄 Random Sampling Progress: {bin_dicts_processed}/{max_iterations} combinations tried.")
                        if progress_callback:
                            progress_percentage = int((bin_dicts_processed / max_iterations) * 100)
                            progress_callback(progress_percentage)

                    if bin_dicts_processed >= max_iterations:
                        break

        print(
            f"❌ Desired privacy model not achieved after {max_iterations} random sampling iterations."
        )
        print(f"Best Fitness Achieved: {self.best_fitness}")
        return None, None

    def update_progress(self, progress_percentage: int):
        """
        Updates the progress percentage.

        Parameters:
            progress_percentage (int): The current progress percentage.
        """
        self.progress.value = progress_percentage

    def plot_time_taken(self, title: str, filename: str):
        """
        Plots and saves the time taken for each iteration.

        Parameters:
            title (str): Title of the plot.
            filename (str): File path to save the plot.
        """
        if not self.times:
            print("⚠️ No time data to plot.")
            return
        plt.figure(figsize=(10, 6))
        plt.plot(self.times, marker='o', linestyle='-')
        plt.xlabel("Iteration")
        plt.ylabel("Time Taken (seconds)")
        plt.title(title)
        plt.grid(True)
        plt.savefig(filename)
        plt.close()
        print(f"📊 Time taken plot saved as {filename}")

    def plot_fitness_history(self, fitness_history: List[float], title: str) -> plt.Figure:
        """
        Plots the fitness over iterations.

        Parameters:
            fitness_history (List[float]): List of fitness scores over iterations.
            title (str): Title of the plot.

        Returns:
            plt.Figure: The matplotlib figure object.
        """
        if not fitness_history:
            print("⚠️ No fitness data to plot.")
            return plt.Figure()

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(fitness_history, marker='o', linestyle='-')
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Fitness (Lower is Better)")
        ax.set_title(title)
        ax.grid(True)
        return fig

    def plot_time_taken(self, times: List[float], title: str) -> plt.Figure:
        """
        Plots the time taken per iteration.

        Parameters:
            times (List[float]): List of time taken for each iteration.
            title (str): Title of the plot.

        Returns:
            plt.Figure: The matplotlib figure object.
        """
        if not times:
            print("⚠️ No time data to plot.")
            return plt.Figure()

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(times, marker='o', color='orange', linestyle='-')
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Time Taken (seconds)")
        ax.set_title(title)
        ax.grid(True)
        return fig

    def get_optimization_summary(self) -> Dict[str, any]:
        """
        Returns a summary of the optimization process, including privacy model specific metrics.

        Returns:
            dict: A dictionary containing summary statistics.
        """
        summary = {
            'Best Fitness': self.best_fitness,
            'Best Bin Dict': self.best_bin_dict,
            'Total Iterations': len(self.fitness_history),
            'Total Time (s)': sum(self.times),
            'Average Time per Iteration (s)': np.mean(self.times) if self.times else None,
        }

        # Add privacy model specific details
        if self.privacy_model == 'k_anonymity':
            # Calculate number of small groups
            if self.best_bin_dict:
                binned_df, _ = bin_columns(self.best_bin_dict, self.original_data)
                small_groups = find_small_groups(binned_df, self.k)
                summary['Small Groups'] = small_groups
        elif self.privacy_model == 'l_diversity':
            if self.best_bin_dict:
                binned_df, _ = bin_columns(self.best_bin_dict, self.original_data)
                l_diversity_penalty = calculate_l_diversity(binned_df, self.sensitive_attributes, self.l)
                summary['L-Diversity Penalty'] = l_diversity_penalty
        elif self.privacy_model == 't_closeness':
            if self.best_bin_dict:
                binned_df, _ = bin_columns(self.best_bin_dict, self.original_data)
                t_closeness_penalty = calculate_t_closeness(binned_df, self.sensitive_attributes, self.t, self.original_data)
                summary['T-Closeness Penalty'] = t_closeness_penalty

        return summary

    def save_best_binned_data(self, filepath: str):
        """
        Saves the best binned DataFrame to a CSV file.

        Parameters:
            filepath (str): The path where the DataFrame will be saved.
        """
        if self.best_binned_df is not None:
            self.best_binned_df.to_csv(filepath, index=False)
            print(f"✅ Best binned data saved to {filepath}")
        else:
            print("⚠️ No binned data to save.")



def runk():
    # example_usage_k_anonymity.py

    import pandas as pd
    import os

    # Load your data
    original_data = pd.read_csv("data/Data.csv")
    # Select relevant columns (modify indices as needed)
    original_data = original_data.iloc[:, [1, 3, 4]]

    # Define your parameters
    k = 2  # Desired k-anonymity level
    max_comb_size = 3  # Maximum combination size to consider
    columns_to_bin = original_data.columns.tolist()

    # Define minimum and maximum bins per column
    min_bins_per_column = {col: 3 for col in columns_to_bin}
    max_bins_per_column = {col: 20 for col in columns_to_bin}

    # Initialize the optimizer for k-anonymity
    binning_optimizer = BinningOptimizer(
        original_data=original_data,
        k=k,
        privacy_model="k_anonymity",  # Privacy model
        sensitive_attributes=None,    # Not needed for k-anonymity
        l=None,                       # Not needed for l-diversity
        t=None,                       # Not needed for t-closeness
        min_comb_size=1,
        max_comb_size=max_comb_size,
        columns=columns_to_bin,
        min_bins_per_column=min_bins_per_column,
        max_bins_per_column=max_bins_per_column,
        max_iterations=1000,
        optimizer="genetic",           # Choose 'genetic' or 'simulated_annealing'
        method="quantile",             # Binning method
    )

    # Perform the binning
    best_bin_dict, best_binned_df = binning_optimizer.find_best_binned_data()

    # Output the results
    if best_bin_dict:
        print("\n🎯 Final Binning Configuration (k-anonymity):")
        for col, bins in best_bin_dict.items():
            print(f" - {col}: {bins} bins")

    if best_binned_df is not None and not best_binned_df.empty:
        print("\n📊 Binned Data Sample:")
        print(best_binned_df.head())

    # Save the best binned data
    binning_optimizer.save_best_binned_data("data/binned_data_k_anonymity.csv")

    # Plot time taken per iteration
    binning_optimizer.plot_time_taken(
        title="Time Taken per Iteration (k-anonymity)",
        filename="data/time_taken_k_anonymity.png"
    )

    # Plot fitness history
    binning_optimizer.plot_fitness_history(
        title="Fitness Over Iterations (k-anonymity)",
        filename="data/fitness_history_k_anonymity.png"
    )

    # Print optimization summary
    summary = binning_optimizer.get_optimization_summary()
    print("\n📈 Optimization Summary (k-anonymity):")
    for key, value in summary.items():
        print(f" - {key}: {value}")

def runl():
    # example_usage_l_diversity.py

    import pandas as pd
    import os

    # Load your data
    original_data = pd.read_csv("data/Data.csv")
    # Select relevant columns (modify indices as needed)
    original_data = original_data.iloc[:, [1, 3, 4, 5]]

    # Define your parameters
    k = 2  # Desired k-anonymity level
    l = 3  # Desired l-diversity level
    max_comb_size = 4  # Maximum combination size to consider
    columns_to_bin = original_data.columns.tolist()

    # Define minimum and maximum bins per column
    min_bins_per_column = {col: 3 for col in columns_to_bin}
    max_bins_per_column = {col: 20 for col in columns_to_bin}

    # Define sensitive attributes
    sensitive_attributes = ['Stay ID']  # Replace with your sensitive columns

    # Initialize the optimizer for l-diversity
    binning_optimizer = BinningOptimizer(
        original_data=original_data,
        k=k,
        privacy_model="l_diversity",     # Privacy model
        sensitive_attributes=sensitive_attributes,
        l=l,                             # Desired l-diversity
        t=None,                          # Not needed for t-closeness
        min_comb_size=1,
        max_comb_size=max_comb_size,
        columns=columns_to_bin,
        min_bins_per_column=min_bins_per_column,
        max_bins_per_column=max_bins_per_column,
        max_iterations=1000,
        optimizer="genetic",  # Choose 'genetic' or 'simulated_annealing'
        method="quantile",                # Binning method
    )

    # Perform the binning
    best_bin_dict, best_binned_df = binning_optimizer.find_best_binned_data()

    # Output the results
    if best_bin_dict:
        print("\n🎯 Final Binning Configuration (l-diversity):")
        for col, bins in best_bin_dict.items():
            print(f" - {col}: {bins} bins")

    if best_binned_df is not None and not best_binned_df.empty:
        print("\n📊 Binned Data Sample:")
        print(best_binned_df.head())

    # Save the best binned data
    binning_optimizer.save_best_binned_data("data/binned_data_l_diversity.csv")

    # Plot time taken per iteration
    binning_optimizer.plot_time_taken(
        title="Time Taken per Iteration (l-diversity)",
        filename="data/time_taken_l_diversity.png"
    )

    # Plot fitness history
    binning_optimizer.plot_fitness_history(
        title="Fitness Over Iterations (l-diversity)",
        filename="data/fitness_history_l_diversity.png"
    )

    # Print optimization summary
    summary = binning_optimizer.get_optimization_summary()
    print("\n📈 Optimization Summary (l-diversity):")
    for key, value in summary.items():
        print(f" - {key}: {value}")


def runt():
    # example_usage_t_closeness.py

    import pandas as pd
    import os

    # Load your data
    original_data = pd.read_csv("data/Data.csv")
    # Select relevant columns (modify indices as needed)
    original_data = original_data.iloc[:, [1, 3, 4, 5, 6]]

    # Define your parameters
    k = 2          # Desired k-anonymity level
    t = 0.05       # Desired t-closeness threshold
    max_comb_size = 3 # Maximum combination size to consider
    columns_to_bin = original_data.columns.tolist()

    # Define minimum and maximum bins per column
    min_bins_per_column = {col: 3 for col in columns_to_bin}
    max_bins_per_column = {col: 20 for col in columns_to_bin}

    # Define sensitive attributes
    sensitive_attributes = ['Age at Colln']  # Replace with your sensitive columns

    # Initialize the optimizer for t-closeness
    binning_optimizer = BinningOptimizer(
        original_data=original_data,
        k=k,
        privacy_model="t_closeness",     # Privacy model
        sensitive_attributes=sensitive_attributes,
        l=None,                           # Not needed for l-diversity
        t=t,                              # Desired t-closeness
        min_comb_size=1,
        max_comb_size=max_comb_size,
        columns=columns_to_bin,
        min_bins_per_column=min_bins_per_column,
        max_bins_per_column=max_bins_per_column,
        max_iterations=1000,
        optimizer="genetic",              # Choose 'genetic' or 'simulated_annealing'
        method="quantile",                # Binning method
    )

    # Perform the binning
    best_bin_dict, best_binned_df = binning_optimizer.find_best_binned_data()

    # Output the results
    if best_bin_dict:
        print("\n🎯 Final Binning Configuration (t-closeness):")
        for col, bins in best_bin_dict.items():
            print(f" - {col}: {bins} bins")

    if best_binned_df is not None and not best_binned_df.empty:
        print("\n📊 Binned Data Sample:")
        print(best_binned_df.head())

    # Save the best binned data
    binning_optimizer.save_best_binned_data("data/binned_data_t_closeness.csv")

    # Plot time taken per iteration
    binning_optimizer.plot_time_taken(
        title="Time Taken per Iteration (t-closeness)",
        filename="data/time_taken_t_closeness.png"
    )

    # Plot fitness history
    binning_optimizer.plot_fitness_history(
        title="Fitness Over Iterations (t-closeness)",
        filename="data/fitness_history_t_closeness.png"
    )

    # Print optimization summary
    summary = binning_optimizer.get_optimization_summary()
    print("\n📈 Optimization Summary (t-closeness):")
    for key, value in summary.items():
        print(f" - {key}: {value}")


if __name__ == "__main__":
    runt()

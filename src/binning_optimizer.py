# src/binning_optimizer.py

import pandas as pd
import numpy as np
import random
import math
import warnings
import matplotlib.pyplot as plt
import time
from concurrent.futures import ThreadPoolExecutor
import os
from typing import Tuple, Dict, List, Optional, Callable
import logging
from scipy.stats import wasserstein_distance  # Required for t-closeness

warnings.filterwarnings("ignore")

def calculate_l_diversity(binned_df: pd.DataFrame, sensitive_attrs: List[str], l: int) -> int:
    """
    Calculates the l-diversity penalty for the binned DataFrame.
    """
    penalties = 0
    for _, group in binned_df.groupby(list(binned_df.columns)):
        for attr in sensitive_attrs:
            unique_values = group[attr].nunique()
            if unique_values < l:
                penalties += (l - unique_values)
    return penalties

def calculate_t_closeness(binned_df: pd.DataFrame, sensitive_attrs: List[str], t: float, original_data: pd.DataFrame) -> float:
    """
    Calculates the t-closeness penalty for the binned DataFrame.
    """
    penalties = 0.0
    overall_distributions = {}
    for attr in sensitive_attrs:
        overall_distributions[attr] = original_data[attr].value_counts(normalize=True)

    for _, group in binned_df.groupby(list(binned_df.columns)):
        for attr in sensitive_attrs:
            bin_distribution = group[attr].value_counts(normalize=True)
            combined = pd.concat([overall_distributions[attr], bin_distribution], axis=1).fillna(0)
            distance = wasserstein_distance(combined.iloc[:, 0], combined.iloc[:, 1])
            if distance > t:
                penalties += (distance - t)
    return penalties

def _bin_categorical_column(series: pd.Series, bins: int) -> Tuple[pd.Series, List[str]]:
    """
    Bins a categorical column by grouping infrequent categories.
    """
    # Calculate category frequencies
    freq = series.value_counts().sort_values(ascending=False)
    # Initialize groups
    groups = {}
    current_group = []
    current_bins = 1

    for category in freq.index:
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

def _bin_column(series: pd.Series, bins: int, method: str, is_datetime: bool = False) -> Tuple[pd.Series, List[str]]:
    """
    Bins a single column using the specified method and returns labeled bins.
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

        return binned.astype('category'), bin_labels

def bin_columns(bin_dict: Dict[str, int], original_data: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, List[str]]]:
    """
    Bins specified columns in the DataFrame based on the provided bin counts and binning method.
    """
    binned_columns = {
        'datetime': [],
        'integer': [],
        'float': [],
        'category_grouped': [],
        'unsupported': []
    }
    Bin_Data = original_data.copy()

    for col, bins in bin_dict.items():
        if col not in Bin_Data.columns:
            print(f"⚠️ Column '{col}' does not exist in the DataFrame. Skipping.")
            continue

        try:
            if pd.api.types.is_datetime64_any_dtype(Bin_Data[col]):
                binned_series, bin_labels = _bin_column(Bin_Data[col], bins, "quantile", is_datetime=True)
                Bin_Data[col] = binned_series
                binned_columns['datetime'].append(col)
                binned_columns[f'{col}_bins'] = bin_labels

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
                binned_series, category_groups = _bin_categorical_column(Bin_Data[col], bins)
                Bin_Data[col] = binned_series
                binned_columns['category_grouped'].append(col)
                binned_columns[f'{col}_groups'] = category_groups
            else:
                print(f"⚠️ Column '{col}' has unsupported dtype '{Bin_Data[col].dtype}'. Skipping.")
                binned_columns['unsupported'].append(col)

        except Exception as e:
            print(f"Failed to bin column '{col}': {e}")
            binned_columns['unsupported'].append(col)

    successfully_binned = (
        binned_columns['datetime'] +
        binned_columns['integer'] +
        binned_columns['float'] +
        binned_columns['category_grouped']
    )
    binned_df = Bin_Data[successfully_binned]

    return binned_df, binned_columns

def find_small_groups(binned_df: pd.DataFrame, k: int) -> int:
    """
    Identifies the number of small groups in the binned DataFrame based on k-anonymity.
    """
    group_counts = binned_df.groupby(list(binned_df.columns)).size()
    small_groups = (group_counts < k).sum()
    return small_groups

class BinningOptimizer:
    """
    A class to perform k-anonymity, l-diversity, or t-closeness binning using either Genetic Algorithm or Simulated Annealing.
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
        logger: Optional[logging.Logger] = None,
        generations: Optional[int] = None,
        population_size: Optional[int] = None,
        mutation_rate: Optional[float] = None,
        initial_temperature: Optional[float] = None,
        cooling_rate: Optional[float] = None,
        iterations: Optional[int] = None,
        neighbors_per_iteration: Optional[int] = None,
        max_workers: Optional[int] = None
    ):
        """
        Initializes the BinningOptimizer with the dataset and parameters.
        """
        self.fitness_cache = {}  # Initialize fitness cache
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
        self.logs = []  # To store logs for the UI
        self.logger = logger or logging.getLogger(__name__)

        # Optimizer-specific hyperparameters
        self.generations = generations or 100
        self.population_size = population_size or 50
        self.mutation_rate = mutation_rate or 0.1
        self.initial_temperature = initial_temperature or 1000.0
        self.cooling_rate = cooling_rate or 0.95
        self.iterations = iterations or 1000
        self.neighbors_per_iteration = neighbors_per_iteration or 5
        self.max_workers = max_workers or min(8, os.cpu_count() or 1)

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
            if not isinstance(self.t, (int, float)) or self.t <= 0 or self.t > 1:
                raise ValueError("Parameter 't' must be a number between 0 and 1 for t-closeness.")
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
        """
        possible_bins_per_column = {}
        for col in self.columns:
            unique_values = self.original_data[col].nunique(dropna=True)
            self.logger.info(f"Column '{col}' has {unique_values} unique values.")

            if unique_values == 0:
                self.logger.warning(f"⚠️ Column '{col}' has no unique values. Skipping.")
                possible_bins_per_column[col] = [1]
                continue

            min_bins = self.min_bins_per_column.get(col, max(1, min(2, unique_values // 10)))
            max_bins = self.max_bins_per_column.get(col, unique_values)

            max_possible_bins = unique_values
            min_bins = max(1, min(min_bins, max_possible_bins))
            max_bins = max(1, min(max_bins, max_possible_bins))

            if min_bins > max_bins:
                min_bins, max_bins = max_bins, min_bins

            possible_bins = list(range(min_bins, max_bins + 1))
            possible_bins_per_column[col] = possible_bins
            self.logger.info(f"Possible bins for column '{col}': {possible_bins}")

        return possible_bins_per_column

    def fitness_function(self, bin_dict: Dict[str, int]) -> float:
        """
        Evaluates the fitness of a binning configuration with caching.
        """
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

            # Log the fitness evaluation
            self.logger.info(f"Evaluated bin_dict {bin_dict}: Fitness = {total_fitness}")

            return total_fitness

        except Exception as e:
            self.logger.error(f"⚠️ Error during fitness evaluation for bin_dict {bin_dict}: {e}", exc_info=True)
            self.fitness_cache[bin_dict_tuple] = np.inf
            return np.inf

    def genetic_algorithm(self, progress_callback: Optional[Callable[[int, str], None]] = None) -> Dict[str, int]:
        """
        Performs the Genetic Algorithm to find the best binning configuration.
        """
        def crossover(parent1, parent2):
            try:
                # Single-point crossover
                crossover_point = random.randint(1, len(parent1) - 1)
                child = {}
                for i, col in enumerate(parent1.keys()):
                    if i < crossover_point:
                        child[col] = parent1[col]
                    else:
                        child[col] = parent2[col]
                return child
            except Exception as e:
                self.logger.error(f"Error during crossover: {e}", exc_info=True)
                return parent1.copy()

        def mutate(individual):
            try:
                # Randomly change the bin count of one column
                col = random.choice(list(individual.keys()))
                old_bins = individual[col]
                individual[col] = random.choice(self.possible_bins_per_column[col])
                self.logger.info(f"Mutated column '{col}' from {old_bins} bins to {individual[col]} bins.")
            except Exception as e:
                self.logger.error(f"Error during mutation: {e}", exc_info=True)

        population_size = self.population_size
        generations = self.generations
        mutation_rate = self.mutation_rate

        population = []
        for _ in range(population_size):
            bin_dict = {
                col: random.choice(self.possible_bins_per_column[col]) for col in self.columns
            }
            population.append(bin_dict)

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            for generation in range(generations):
                start_time = time.time()
                fitness_scores = list(executor.map(self.fitness_function, population))

                fitness_individuals = list(zip(fitness_scores, population))
                fitness_individuals.sort(key=lambda x: x[0])
                best_fitness, best_individual = fitness_individuals[0]
                self.fitness_history.append(best_fitness)
                self.logger.info(f"Generation {generation + 1}/{generations}: Best Fitness = {best_fitness}")

                if best_fitness < self.best_fitness:
                    self.best_fitness = best_fitness
                    self.best_bin_dict = best_individual.copy()
                    self.logger.info(f"New best fitness {best_fitness} found.")

                # Report progress
                progress_percentage = int(((generation + 1) / generations) * 100)
                if progress_callback:
                    progress_callback(progress_percentage, f"Generation {generation + 1}/{generations}: Best Fitness = {best_fitness}")

                if best_fitness == 0:
                    self.logger.info("✅ Optimal solution found via Genetic Algorithm.")
                    end_time = time.time()
                    self.times.append(end_time - start_time)
                    return best_individual

                # Selection
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

        self.logger.warning("❌ Optimal solution not found via Genetic Algorithm within the given generations.")
        self.logger.info(f"Best Fitness Achieved: {self.best_fitness}")
        return self.best_bin_dict

    def simulated_annealing(self, progress_callback: Optional[Callable[[int, str], None]] = None) -> Dict[str, int]:
        """
        Performs Simulated Annealing to find the best binning configuration.
        """
        initial_temperature = self.initial_temperature
        cooling_rate = self.cooling_rate
        iterations = self.iterations
        neighbors_per_iteration = self.neighbors_per_iteration

        current_solution = {
            col: random.choice(self.possible_bins_per_column[col]) for col in self.columns
        }
        current_fitness = self.fitness_function(current_solution)
        best_solution = current_solution.copy()
        best_fitness = current_fitness
        temperature = initial_temperature

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            for iteration in range(iterations):
                start_time = time.time()
                neighbors = []
                for _ in range(neighbors_per_iteration):
                    neighbor = current_solution.copy()
                    col = random.choice(list(neighbor.keys()))
                    old_bins = neighbor[col]
                    neighbor[col] = random.choice(self.possible_bins_per_column[col])
                    self.logger.info(f"Simulated Annealing Mutation: Column '{col}' from {old_bins} bins to {neighbor[col]} bins.")
                    neighbors.append(neighbor)

                fitness_results = list(executor.map(self.fitness_function, neighbors))

                min_fitness = min(fitness_results)
                min_index = fitness_results.index(min_fitness)
                best_neighbor = neighbors[min_index]

                if (
                    min_fitness < current_fitness
                    or random.uniform(0, 1) < math.exp((current_fitness - min_fitness) / max(temperature, 1e-10))
                ):
                    current_solution = best_neighbor
                    current_fitness = min_fitness
                    self.logger.info(f"Simulated Annealing Iteration {iteration + 1}: Accepted new solution with Fitness = {current_fitness}")

                    if current_fitness < best_fitness:
                        best_solution = current_solution.copy()
                        best_fitness = current_fitness
                        self.logger.info(f"Simulated Annealing Iteration {iteration + 1}: New Best Fitness = {best_fitness}")

                temperature *= cooling_rate
                self.fitness_history.append(best_fitness)
                self.logger.info(f"Simulated Annealing Iteration {iteration + 1}/{iterations}: Best Fitness = {best_fitness}")

                progress_percentage = int(((iteration + 1) / iterations) * 100)
                if progress_callback:
                    progress_callback(progress_percentage, f"Iteration {iteration + 1}/{iterations}: Best Fitness = {best_fitness}")

                if best_fitness == 0:
                    self.logger.info("✅ Optimal solution found via Simulated Annealing.")
                    end_time = time.time()
                    self.times.append(end_time - start_time)
                    break
                end_time = time.time()
                self.times.append(end_time - start_time)

        if best_fitness != 0:
            self.logger.warning("❌ Optimal solution not found via Simulated Annealing within the given iterations.")
            self.logger.info(f"Best Fitness Achieved: {best_fitness}")

        self.best_fitness = best_fitness
        self.best_bin_dict = best_solution.copy()
        return best_solution

    def find_best_binned_data(self, progress_callback: Optional[Callable[[int, str], None]] = None) -> Tuple[Optional[Dict[str, int]], Optional[pd.DataFrame]]:
        """
        Finds the best binning configuration to achieve the selected privacy model using the chosen optimizer.
        """
        self.logger.info("🔍 Starting Random Sampling Phase...")
        bin_dict, binned_df = self._random_sampling_phase(progress_callback=progress_callback)
        if bin_dict:
            self.best_bin_dict = bin_dict.copy()
            self.best_binned_df = binned_df.copy()
            self.logger.info("🔍 Random Sampling Phase succeeded.")
            return bin_dict, binned_df

        self.logger.info(f"🚀 Starting Optimization Phase using {self.optimizer.capitalize()}...")
        if self.optimizer == "genetic":
            best_bin_dict = self.genetic_algorithm(progress_callback=progress_callback)
        elif self.optimizer == "simulated_annealing":
            best_bin_dict = self.simulated_annealing(progress_callback=progress_callback)
        else:
            raise ValueError("Unsupported optimizer. Choose 'genetic' or 'simulated_annealing'.")

        binned_df, _ = bin_columns(best_bin_dict, self.original_data)
        self.best_binned_df = binned_df.copy()

        return best_bin_dict, binned_df

    def _random_sampling_phase(self, progress_callback: Optional[Callable[[int, str], None]] = None) -> Tuple[Optional[Dict[str, int]], Optional[pd.DataFrame]]:
        """
        Performs random sampling to find a binning configuration that satisfies the selected privacy model.
        """
        total_combinations = np.prod(
            [len(self.possible_bins_per_column[col]) for col in self.columns]
        )
        self.logger.info(f"Total possible combinations: {total_combinations}")

        max_iterations = min(self.max_iterations, total_combinations)
        self.logger.info(f"Sampling up to {max_iterations} random combinations.")

        tried_bin_dicts = set()
        progress_interval = 100

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
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

            chunk_size = 100
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

                fitness_results = list(executor.map(self.fitness_function, bin_dicts))

                for bin_dict, fitness in zip(bin_dicts, fitness_results):
                    bin_dicts_processed += 1

                    if fitness == 0:
                        binned_df, _ = bin_columns(bin_dict, self.original_data)
                        self.logger.info(f"✅ Desired privacy model achieved with bin_dict: {bin_dict}")
                        return bin_dict, binned_df
                    else:
                        if fitness < self.best_fitness:
                            self.best_fitness = fitness
                            self.best_bin_dict = bin_dict.copy()

                    if bin_dicts_processed % progress_interval == 0:
                        self.logger.info(f"🔄 Random Sampling Progress: {bin_dicts_processed}/{max_iterations} combinations tried.")
                        if progress_callback:
                            progress_percentage = int((bin_dicts_processed / max_iterations) * 100)
                            progress_callback(progress_percentage, f"Random Sampling: {bin_dicts_processed}/{max_iterations} tried.")

                    if bin_dicts_processed >= max_iterations:
                        break

        self.logger.warning(
            f"❌ Desired privacy model not achieved after {max_iterations} random sampling iterations."
        )
        self.logger.info(f"Best Fitness Achieved: {self.best_fitness}")
        return None, None

    def check_privacy(self, binned_df: pd.DataFrame) -> Tuple[bool, Dict[str, any]]:
        """
        Performs detailed privacy checks on the binned DataFrame.
        """
        privacy_achieved = False
        privacy_details = {}

        try:
            if binned_df.empty:
                self.logger.error("Binned DataFrame is empty. Privacy cannot be achieved.")
                return False, {"Error": "Binned DataFrame is empty."}

            if self.privacy_model == "k_anonymity":
                group_sizes = binned_df.groupby(list(binned_df.columns)).size()
                min_group_size = group_sizes.min()
                privacy_achieved = min_group_size >= self.k
                privacy_details['Minimum Group Size'] = min_group_size
                privacy_details['Required k'] = self.k

            elif self.privacy_model == "l_diversity":
                total_penalties = calculate_l_diversity(binned_df, self.sensitive_attributes, self.l)
                privacy_achieved = total_penalties == 0
                privacy_details['Total L-Diversity Penalty'] = total_penalties
                privacy_details['Required l'] = self.l

            elif self.privacy_model == "t_closeness":
                total_penalties = calculate_t_closeness(binned_df, self.sensitive_attributes, self.t, self.original_data)
                privacy_achieved = total_penalties == 0.0
                privacy_details['Total T-Closeness Penalty'] = total_penalties
                privacy_details['Required t'] = self.t

            self.logger.info(f"Privacy Achievement: {'Achieved' if privacy_achieved else 'Not Achieved'}")
            return privacy_achieved, privacy_details

        except Exception as e:
            self.logger.error(f"Error during privacy checks: {e}", exc_info=True)
            return False, {"Error": str(e)}

    def get_optimization_summary(self) -> Dict[str, any]:
        """
        Returns a summary of the optimization process, including privacy model specific metrics.
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

    def get_logs(self) -> str:
        """
        Returns the accumulated logs for display.
        """
        return "\n".join(self.logs)

    def get_privacy_recommendations(self) -> List[str]:
        """
        Provides recommendations to the user if privacy was not achieved.
        """
        recommendations = []
        if self.privacy_model == "k_anonymity":
            recommendations.append("Increase the number of bins to reduce data granularity.")
            recommendations.append("Consider removing or generalizing quasi-identifiers.")
        elif self.privacy_model == "l_diversity":
            recommendations.append("Increase the diversity of sensitive attributes.")
            recommendations.append("Adjust binning to balance group sizes and diversity.")
        elif self.privacy_model == "t_closeness":
            recommendations.append("Allow for greater t value to relax closeness constraints.")
            recommendations.append("Reassess the distribution of sensitive attributes.")

        recommendations.append("Adjust the number of bins or select different columns for binning.")
        recommendations.append("Preprocess the data to reduce the number of unique values where possible.")
        return recommendations

    def plot_k_anonymity_compliance(self) -> plt.Figure:
        """
        Plots the distribution of group sizes to visualize k-anonymity compliance.
        """
        try:
            binned_df, _ = bin_columns(self.best_bin_dict, self.original_data)
            group_sizes = binned_df.groupby(list(binned_df.columns)).size()
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.hist(group_sizes, bins=range(1, group_sizes.max() + 2), color='purple', alpha=0.7, edgecolor='black')
            ax.axvline(x=self.k, color='red', linestyle='dashed', linewidth=2, label=f'k = {self.k}')
            ax.set_xlabel("Group Size")
            ax.set_ylabel("Number of Groups")
            ax.set_title("K-Anonymity Compliance: Distribution of Group Sizes")
            ax.legend()
            ax.grid(True)
            return fig
        except Exception as e:
            self.logger.error(f"Error plotting k-anonymity compliance: {e}", exc_info=True)
            return plt.Figure()

    def plot_l_diversity_compliance(self) -> plt.Figure:
        """
        Plots the diversity metrics to visualize l-diversity compliance.
        """
        try:
            binned_df, _ = bin_columns(self.best_bin_dict, self.original_data)
            diversity_metrics = []
            for _, group in binned_df.groupby(list(binned_df.columns)):
                for attr in self.sensitive_attributes:
                    unique_values = group[attr].nunique()
                    diversity_metrics.append(unique_values)

            fig, ax = plt.subplots(figsize=(10, 6))
            ax.hist(diversity_metrics, bins=range(1, max(diversity_metrics) + 2), color='teal', alpha=0.7, edgecolor='black')
            ax.axvline(x=self.l, color='red', linestyle='dashed', linewidth=2, label=f'l = {self.l}')
            ax.set_xlabel("Number of Unique Sensitive Attribute Values")
            ax.set_ylabel("Frequency")
            ax.set_title("L-Diversity Compliance: Distribution of Unique Values per Group")
            ax.legend()
            ax.grid(True)
            return fig
        except Exception as e:
            self.logger.error(f"Error plotting l-diversity compliance: {e}", exc_info=True)
            return plt.Figure()

    def plot_t_closeness_compliance(self) -> plt.Figure:
        """
        Plots the t-closeness metrics to visualize compliance.
        """
        try:
            binned_df, _ = bin_columns(self.best_bin_dict, self.original_data)
            closeness_metrics = []
            overall_distributions = {}
            for attr in self.sensitive_attributes:
                overall_distributions[attr] = self.original_data[attr].value_counts(normalize=True)

            for _, group in binned_df.groupby(list(binned_df.columns)):
                for attr in self.sensitive_attributes:
                    bin_distribution = group[attr].value_counts(normalize=True)
                    combined = pd.concat([overall_distributions[attr], bin_distribution], axis=1).fillna(0)
                    distance = wasserstein_distance(combined.iloc[:, 0], combined.iloc[:, 1])
                    closeness_metrics.append(distance)

            fig, ax = plt.subplots(figsize=(10, 6))
            ax.hist(closeness_metrics, bins=30, color='orange', alpha=0.7, edgecolor='black')
            ax.axvline(x=self.t, color='red', linestyle='dashed', linewidth=2, label=f't = {self.t}')
            ax.set_xlabel("Wasserstein Distance")
            ax.set_ylabel("Frequency")
            ax.set_title("T-Closeness Compliance: Distribution of Wasserstein Distances")
            ax.legend()
            ax.grid(True)
            return fig
        except Exception as e:
            self.logger.error(f"Error plotting t-closeness compliance: {e}", exc_info=True)
            return plt.Figure()



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

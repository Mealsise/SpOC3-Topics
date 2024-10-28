from typing import Any

import time  # For time.time()
from datetime import datetime  # For datetime.now()
import random  # For random.seed()

# PyGMO (pg) components
import pygmo as pg  # For pg.algorithm and pg.population

# Project-specific imports
from chromosome_operations import ChromosomeController, PopulationFactory, DistanceMetric
from file_handling import PopulationFileManager
from output_helpers import print_progress_bar_with_time


# Default configuration values used throughout the system.

# Tolerance level for detecting changes in fitness improvement. Used in evolutionary algorithms 
# to decide when to stop further optimization if the fitness does not improve significantly.
DEFAULT_NO_CHANGE_TOLERANCE = 1e-5

# Number of cycles allowed to run without significant improvement before stopping early.
# Expressed as a fraction of the total number of cycles (i.e., 0.5 means half of the cycles).
DEFAULT_PATIENCE = 0.5

# The default number of cycles for evolutionary algorithms. Determines how many iterations
# are run during the evolutionary process.
DEFAULT_NUMBER_OF_CYCLES = 1000

# Default population size for generating populations in evolutionary algorithms.
DEFAULT_POPULATION_SIZE = 100

# Default number of populations to generate when using the `generate_and_save_multiple_populations` function.
DEFAULT_NUM_POPULATIONS = 10

# Default metric for calculating similarity
DEFAULT_DISTANCE_METRIC = "manhattan"

class PgEvolver:
    def __init__(self, algorithm: pg.algorithm):
        """
        Initialize with a given algorithm.

        Args:
            algorithm (pg.algorithm): The PyGMO algorithm (e.g., pg.sade, pg.de, etc.).
        """
        self.algorithm = algorithm

    def __call__(self, population: pg.population, state: dict) -> tuple:
        """
        Evolve the population using the algorithm and return the updated population.

        Args:
            population (pg.population): The current population to evolve.
            state (dict): The current state (can be ignored in this case).

        Returns:
            Tuple: (evolved population, unchanged state)
        """
        population = self.algorithm.evolve(population)
        return population, state

class CollectAlive:
    def __init__(self, chromosome_controller: ChromosomeController):
        """
        Initialize the CollectAlive step with a ChromosomeController instance.

        Args:
            chromosome_controller (ChromosomeController): An instance of the ChromosomeController to extract satellite information.
        """
        self.chromosome_controller = chromosome_controller

    def __call__(self, population: pg.population, state: dict) -> tuple:
        """
        Collects all alive drones from the population and stores them in the state.

        Args:
            population (pg.population): The current population.
            state (dict): The state dictionary where alive drones will be stored.

        Returns:
            Tuple: (unchanged population, updated state)
        """
        alive_drones: list[list[float]] = []

        # Loop over each config and extend the alive_drones list with found satellites
        for config in population.get_x():
            alive_satellites = self.chromosome_controller.extract_alive_satellites(config)
            alive_drones.extend(alive_satellites)

        # Store the alive drones in the state
        state['alive_drones'] = alive_drones

        return population, state

class Resurrect:
    def __init__(self, chromosome_controller: ChromosomeController, distance_metric: str = DEFAULT_DISTANCE_METRIC):
        """
        Initialize the Resurrect step with a ChromosomeController instance.

        Args:
            chromosome_controller (ChromosomeController): An instance of the ChromosomeController to set satellite components.
        """
        self.chromosome_controller = chromosome_controller
        self.distance_metric_name = distance_metric

    def __call__(self, population: pg.population, state: dict) -> tuple:
        """
        Finds dead drones in the population and replaces them with the most similar alive drones from the state.

        Args:
            population (pg.population): The current population.
            state (dict): The state dictionary where alive drones are stored.

        Returns:
            Tuple: (updated population, unchanged state)
        """
        alive_drones = state.get('alive_drones', [])

        # Early return
        if not alive_drones:
            return population, state

        for config in population.get_x():
            dead_indices = self.chromosome_controller.get_dead_indices(config)

            for index in dead_indices:
                dead_satellite = config[index]

                # Find the most similar alive satellite using the chosen distance metric
                most_similar = min(
                    alive_drones,
                    key=lambda alive_drone: DistanceMetric.calculate(dead_satellite, alive_drone, self.distance_metric_name)
                )

                # Replace dead satellite components with the most similar alive satellite
                self.chromosome_controller.set_all_components(config, index, most_similar)

        return population, state

class EvolutionRunner:
    def __init__(
            self,
            population: pg.population,
            steps: list,
            cycles: int = DEFAULT_NUMBER_OF_CYCLES,
            patience: float = DEFAULT_PATIENCE,
            no_change_tolerance: float = DEFAULT_NO_CHANGE_TOLERANCE
    ):
        """
        Initializes the EvolutionRunner with the given population, steps, and parameters for evolution.

        Args:
            population (pg.population): The initial population to evolve.
            steps (list): A list of evolutionary steps to apply during each cycle (e.g., mutation, crossover).
            cycles (int): The maximum number of evolution cycles to run. Defaults to DEFAULT_NUMBER_OF_CYCLES .
            patience (float): The fraction of cycles to wait without improvement before stopping early. Defaults to DEFAULT_PATIENCE.
            no_change_tolerance (float): The threshold for detecting improvement in fitness. Defaults to DEFAULT_NO_CHANGE_TOLERANCE.

        Attributes:
            population (pg.population): The current population being evolved.
            steps (list): The list of steps to be applied during evolution.
            cycles (int): The number of cycles to run the evolution.
            patience (float): How many cycles can pass without improvement before stopping early. As a percentage of total cycles.
            no_change_tolerance (float): The amount of change in fitness needed to reset the patience.
            fitness_progress (list): A list to track the fitness score after each cycle.
            best_chromosome (Any): Stores the best chromosome found during the evolution.
            best_fitness (float): Tracks the best fitness score found so far.
            no_improvement_cycles (int): Counter for how many cycles have passed without fitness improvement.
        """
        self.population = population
        self.steps = steps
        self.cycles = cycles
        self.patience = patience
        self.no_change_tolerance = no_change_tolerance
        self.fitness_progress = []
        self.best_chromosome = None
        self.best_fitness = float('inf')
        self.no_improvement_cycles = 0

    def run_cycle(self) -> None:
        """
        Runs a single evolution cycle by applying all steps to the population.

        Each step is applied in sequence, potentially modifying the population. The
        state dictionary is passed along with the population for use by the steps, if needed.

        Returns:
            None
        """
        state = {}
        for step in self.steps:
            self.population, state = step(self.population, state)

    def track_fitness(self) -> None:
        """
        Tracks the current fitness of the population and updates the best fitness.

        The fitness of the current population's champion is evaluated. If there is no
        significant improvement based on `no_change_tolerance`, a counter (`no_improvement_cycles`)
        is incremented. Otherwise, the best fitness is updated and the counter reset.

        Returns:
            None
        """
        current_fitness = self.population.champion_f[0]
        self.fitness_progress.append(current_fitness)
        self.best_chromosome = self.population.champion_x

        if abs(current_fitness - self.best_fitness) < self.no_change_tolerance:
            self.no_improvement_cycles += 1
        else:
            self.best_fitness = current_fitness
            self.no_improvement_cycles = 0

    def hasConverged(self) -> bool:
        """
        Checks if the number of cycles without improvement has reached the patience threshold.

        If the number of cycles without significant fitness improvement exceeds the
        patience threshold, the evolution is considered converged and will stop early.

        Returns:
            bool: True if the evolution has converged, False otherwise.
        """
        return self.no_improvement_cycles >= self.patience * self.cycles

    def run_evolution(self) -> tuple[list[int], list[int]]:
        """
        Runs the evolution process for a defined number of cycles or until convergence.

        The evolution process runs for `cycles` number of iterations unless convergence
        is reached earlier, as determined by the `hasConverged` method. The fitness is
        tracked during each cycle, and progress is printed using a progress bar.

        Returns:
            tuple: A tuple containing:
                - fitness_progress (list): A list of fitness values for each cycle.
                - best_chromosome (Any): The chromosome with the best fitness found.
        """
        start_time = time.time()

        for cycle in range(self.cycles):
            # Run a single evolution cycle
            self.run_cycle()

            # Track the fitness of the population
            self.track_fitness()

            # Print progress
            suffix = f"{self.best_fitness:.5f}"
            print_progress_bar_with_time(cycle, self.cycles, start_time, suffix=suffix)

            # Check if evolution has converged (end due to no improvement)
            if self.hasConverged():
                print(f"Stopping early after {cycle} cycles due to lack of improvement for the last {self.no_improvement_cycles} cycles.")
                break

        return self.fitness_progress, self.best_chromosome

class PopulationManager:
    def __init__(
            self, 
            population_factory: PopulationFactory, 
            satellite_pool_filename: str, 
            orbit_name: str, 
            population_size: int = DEFAULT_POPULATION_SIZE, 
            base_dir: str = "initial_populations", 
            format: str = "json"
    ):
        """
        Initializes the PopulationManager with necessary parameters for population generation and file management.

        Args:
            population_factory: A factory object that generates populations.
            satellite_pool_filename (str): The name of the satellite pool file to include in metadata.
            orbit_name (str): The orbit name for file structure.
            population_size (int): The size of each population to generate. Default is 100.
            base_dir (str): The base directory where populations will be saved. Default is "initial_populations".
            format (str): The file format to save populations ('json' or 'csv'). Default is 'json'.

        Attributes:
            population_factory: The factory object responsible for generating populations.
            satellite_pool_filename (str): The name of the satellite pool file.
            orbit_name (str): The orbit name used for organizing file structure.
            population_size (int): The size of each population to generate.
            factory_name (str): The name of the chromosome factory used to generate the population.
            base_dir (str): The base directory for saving populations.
            format (str): The format used for saving populations ('json' or 'csv').
            file_manager (PopulationFileManager): Manages the saving and loading of population files.
        """
        self.population_factory = population_factory
        self.satellite_pool_filename = satellite_pool_filename
        self.orbit_name = orbit_name
        self.population_size = population_size
        self.factory_name = population_factory.chromosome_factory.get_name()
        self.base_dir = base_dir
        self.format = format

        # Initialize the PopulationFileManager for saving/loading populations
        self.file_manager = PopulationFileManager(
            directory=f"{self.base_dir}/{self.orbit_name}/{self.factory_name}", 
            format=self.format, 
            population_size=self.population_size
        )

    def _create_metadata(self, population_index: int, pop_time_elapsed: float, seed_value: str) -> dict:
        """
        Helper function to create metadata for the population.

        Args:
            population_index (int): The index of the population.
            pop_time_elapsed (float): Time taken to generate the population.
            seed_value (str): The seed used for population generation.

        Returns:
            dict: A dictionary containing metadata about the generated population.
        """
        current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        return {
            'chromosome_factory': self.factory_name,
            'satellite_pool_filename': self.satellite_pool_filename,
            'population_size': self.population_size,
            'generation': population_index,
            'generation_time_sec': pop_time_elapsed,
            'timestamp': current_time,
            'seed': seed_value
        }

    def _get_population_filename(self, population_index: int) -> str:
        """
        Generates the filename for the population.

        Args:
            population_index (int): The index of the population.

        Returns:
            str: The generated filename.
        """
        return f"{self.population_size}_{population_index}"

    def _file_exists(self, filename: str) -> bool:
        """
        Checks if a population file already exists.

        Args:
            filename (str): The filename to check.

        Returns:
            bool: True if the file exists, otherwise False.
        """
        return self.file_manager.file_exists(filename)

    def _generate_seed(self, population_index: int) -> str:
        """
        Generates a seed string for reproducibility based on the population index.

        This seed ensures that each population generation is reproducible by using
        the population index to create a unique seed string. This seed is also used 
        in the metadata to ensure consistency across the generation process.

        Args:
            population_index (int): The index of the population, used to generate a unique seed.

        Returns:
            str: The generated seed string, which can be used to set the random seed and 
                is also included in the metadata for reproducibility.
        """
        return f"population_generation_{population_index}"

    def _generate_population(self) -> Any:
        """
        Generates a population using the population factory.

        Returns:
            Any: The generated population.
        """
        self.start_time = time.time()  # Save the start time to calculate generation time later
        return self.population_factory(self.population_size)

    def _extract_chromosomes(self, population: Any) -> list:
        """
        Extracts chromosomes from the population, converting to list if necessary.

        Args:
            population (Any): The generated population.

        Returns:
            list: The list of chromosomes.
        """
        chromosomes = population.get_x()
        if isinstance(chromosomes, np.ndarray):
            chromosomes = chromosomes.tolist()
        return chromosomes

    def _save_population(self, chromosomes: list, metadata: dict, filename: str) -> None:
        """
        Saves the population and its metadata using the file manager.

        Args:
            chromosomes (list): The list of chromosomes to save.
            metadata (dict): Metadata to associate with the population.
            filename (str): The filename to save the population under.

        Returns:
            None
        """
        self.file_manager.save_population(chromosomes, metadata, filename)

    def generate_and_save_population(self, population_index: int) -> None:
        """
        Generates a population, attaches metadata, and saves it to disk.

        This method first checks if a population file with the specified index already exists.
        If not, it generates the population, sets the seed for reproducibility, extracts the
        chromosome data, prepares metadata, and saves the population along with the metadata.

        Args:
            population_index (int): The index of the population to generate and save (used for filename).

        Returns:
            None
        """
        filename = self._get_population_filename(population_index)

        # Skip generation if the file already exists
        if self._file_exists(filename):
            print(f"File {filename} already exists. Skipping generation.")
            return

        # Generate and set reproducibility seed
        seed = self._generate_seed(population_index)
        random.seed(seed)

        # Generate population
        population = self._generate_population()

        # Prepare metadata
        chromosomes = self._extract_chromosomes(population)
        elapsed_time = time.time() - self.start_time
        metadata = self._create_metadata(population_index, elapsed_time, seed)

        # Save population and metadata
        self._save_population(chromosomes, metadata, filename)

        print(f"Saved population {population_index} to {filename}")

    def generate_and_save_multiple_populations(self, num_populations: int = DEFAULT_NUM_POPULATIONS) -> None:
        """
        Generates and saves multiple populations sequentially.

        This method generates a specified number of populations and saves them to disk.
        It prints the total time taken to generate and save the populations at the end.

        Args:
            num_populations (int): The number of populations to generate and save. Defaults to 10.

        Returns:
            None
        """
        if num_populations <= 0:
            print(f"Invalid number of populations: {num_populations}. No populations generated.")
            return

        start_time = time.time()

        for population_index in range(1, num_populations + 1):
            self.generate_and_save_population(population_index)

        # Print total time taken
        total_time_elapsed = time.time() - start_time
        print(f"Finished generating and saving {num_populations} populations in {total_time_elapsed:.2f} seconds.")
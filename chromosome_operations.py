import random
import math
import time
import numpy as np
import pygmo as pg
from typing import List, Callable, Tuple
from collections import Counter
from numba import jit
from collections import defaultdict
from file_handling import *
from output_helpers import print_progress_bar_with_time


def set_expansion_bounds(udp, x_vals, y_vals, z_vals, xv_vals, yv_vals, zv_vals, expansion_factor=0.5):
    lower_bounds = []
    upper_bounds = []
    for vals in (x_vals, y_vals, z_vals, xv_vals, yv_vals, zv_vals):
        lower_bounds.append(min(vals) - expansion_factor)
        upper_bounds.append(max(vals) + expansion_factor)

    actual_bounds = [1,1,1,10, 10,10]
    for i in range(6):
        if lower_bounds[i] < -actual_bounds[i]:
            lower_bounds[i] = -actual_bounds[i]
        if lower_bounds[i] > actual_bounds[i]:
            lower_bounds[i] = actual_bounds[i]

    udp.set_custom_bounds(lower_bounds, upper_bounds)
    udp.set_custom_bounds_switch(True)


# General constants
DEAD_RANGE_MARKER = 1000          # Used to indicate a dead range or no valid value.
POOL_SIZE = 500                   # The target number of satellites to collect for the pool.

# Satellite selection constants
DISTANT_SELECTION_COUNT = 25      # Number of most distant satellites to select first.
DIFFERENTIATED_SELECTION_COUNT = 30  # Number of satellites selected using the differentiated method.

# Subset selection constants
RANDOM_SELECTION_SUBSET_SIZE = 100  # Subset size for random satellite selection in fitness-based methods.


TIGHT_BOUND_EXPANSION = 0.1



class ChromosomeController:
    def __init__(self, udp):
        """
        Initialize the ChromosomeController with the UDP instance and number of satellites.

        Args:
            udp: The UDP instance containing satellite information.
            n_sat (int): The total number of satellites.
        """
        self.udp = udp
        self.n_sat = udp.n_sat
        self.lower_bounds, self.upper_bounds = udp.get_bounds()

    def make_dead_config(self) -> List[float]:
        """
        Create a configuration where all satellites are set to an invalid state.

        Returns:
            List[float]: A configuration with invalid satellite data.
        """
        return [100 for _ in range(self.n_sat * 6)]

    def get_component_bounds(self, component: int):
        bound_index = self.n_sat * component
        return self.lower_bounds[bound_index], self.upper_bounds[bound_index]

    def get_component(self, config: List[float], index: int, component: int) -> float:
        """
        Retrieves the requested component for a satellite at a given index.

        Args:
            config (List[float]): The configuration data for all satellites.
            index (int): The satellite index (0-based).
            component (int): The component to retrieve (0: X, 1: Y, 2: Z, 3: Xv, 4: Yv, 5: Zv).

        Returns:
            float: The value of the requested component for the satellite at the given index.
        """
        return config[index + self.n_sat * component]

    def set_component(self, config: List[float], index: int, component: int, value: float) -> None:
        """
        Sets the value for the requested component for a satellite at a given index.

        Args:
            config (List[float]): The configuration data for all satellites.
            index (int): The satellite index (0-based).
            component (int): The component to set (0: X, 1: Y, 2: Z, 3: Xv, 4: Yv, 5: Zv).
            value (float): The value to set for the specified component.
        """
        config[index + self.n_sat * component] = value

    def get_all_components(self, config: List[float], index: int) -> List[float]:
        """
        Retrieves all components for a satellite at a given index.

        Args:
            config (List[float]): The satellite configuration.
            index (int): The satellite index (0-based).

        Returns:
            List[float]: A list containing X, Y, Z, Xv, Yv, Zv for the satellite.
        """
        return [self.get_component(config, index, i) for i in range(6)]

    def set_all_components(self, config: List[float], index: int, values: List[float]) -> None:
        """
        Sets all components for a satellite at a given index.

        Args:
            config (List[float]): The satellite configuration.
            index (int): The satellite index (0-based).
            values (List[float]): A list of values to set for X, Y, Z, Xv, Yv, Zv.
        """
        for i, value in enumerate(values):
            self.set_component(config, index, i, value)

    def get_alive_indices(self, config: List[float]) -> List[int]:
        """
        Get the indices of alive satellites based on the UDP instance and satellite configuration.

        Args:
            config (List[float]): The satellite configuration.

        Returns:
            List[int]: A list of indices of alive satellites.
        """
        lost_satellites = self.udp.getLost(config)
        flattened_lost = set(item for sublist in lost_satellites for item in sublist)
        return [index for index in range(self.n_sat) if index not in flattened_lost]

    def get_dead_indices(self, config: List[float]) -> List[int]:
        """
        Get the indices of dead satellites based on the UDP instance and satellite configuration.

        Args:
            config (List[float]): The satellite configuration.

        Returns:
            List[int]: A list of indices of dead satellites.
        """
        return [index for index in range(self.n_sat) if index not in self.get_alive_indices(config)]

    def extract_alive_satellites(self, config: List[float]) -> List[List[float]]:
        """
        Extracts the [x, y, z, xv, yv, zv] components for all alive satellites.

        Args:
            config (List[float]): The satellite configuration.

        Returns:
            List[List[float]]: A list of configurations for alive satellites.
        """
        alive_indices = self.get_alive_indices(config)
        return [self.get_all_components(config, index) for index in alive_indices]

    def find_extreme_ranges(self, satellite, component, lower_value, upper_value) -> Tuple[float, float, float, float]:
        """
        Find the most extreme alive and dead satellite ranges by varying a component between lower and upper values.
        Resets the specific component to the original satellite value after variation.

        Args:
            satellite (List[float]): The satellite's parameters (x, y, z, xv, yv, zv).
            component (int): The component index (0: x, 1: y, 2: z, etc.).
            lower_value (float): The lower bound value for the component.
            upper_value (float): The upper bound value for the component.

        Returns:
            Tuple[float, float, float, float]: 
            - (alive_min, dead_min, alive_max, dead_max)
            alive_min: the most extreme alive value close to the lower bound.
            dead_min: the first dead value just past alive_min.
            alive_max: the most extreme alive value close to the upper bound.
            dead_max: the first dead value just past alive_max.
        """
        if lower_value == DEAD_RANGE_MARKER or upper_value == DEAD_RANGE_MARKER:
            return (DEAD_RANGE_MARKER, DEAD_RANGE_MARKER, DEAD_RANGE_MARKER, DEAD_RANGE_MARKER)

        config = self.make_dead_config()
        for index in range(self.n_sat):
            self.set_all_components(config, index, satellite)

        # Vary the component within the lower and upper values
        component_values = np.linspace(lower_value, upper_value, num=self.n_sat)
        for index, value in enumerate(component_values):
            self.set_component(config, index, component, value)

        # Find the indices of alive satellites
        alive_indices = self.get_alive_indices(config)

        if len(alive_indices) == 0:
            return (DEAD_RANGE_MARKER, DEAD_RANGE_MARKER, DEAD_RANGE_MARKER, DEAD_RANGE_MARKER)

        # Determine alive min, alive max, and the closest dead values
        alive_min = component_values[alive_indices[0]]  # The first alive value
        alive_max = component_values[alive_indices[-1]]  # The last alive value

        # Find the dead ranges just beyond the alive extremes
        dead_min = lower_value  # Initialize dead_min with the lower bound
        dead_max = upper_value  # Initialize dead_max with the upper bound

        if alive_indices[0] > 0:
            dead_min = component_values[alive_indices[0] - 1]  # First dead before alive_min
        if alive_indices[-1] < len(component_values) - 1:
            dead_max = component_values[alive_indices[-1] + 1]  # First dead after alive_max

        return alive_min, dead_min, alive_max, dead_max

    def generate_satellite_variants(self, satellite, n_iterations: int = 2) -> List[List[float]]:
        """
        Refine the most extreme upper and lower bounds for all 6 components (x, y, z, xv, yv, zv) over multiple iterations.

        Args:
            satellite (List[float]): The original satellite parameters [x, y, z, xv, yv, zv].
            n_iterations (int): The number of refinement iterations.

        Returns:
            List[List[float]]: A list of 13 satellite configurations, including:
                            [original, extreme_x_lower, extreme_x_upper, ..., extreme_zv_lower, extreme_zv_upper]
        """
        variants  = [satellite]

        for component in range(6):
            lower_bound, upper_bound = self.get_component_bounds(component)

            lower_alive_bound, lower_dead_bound, upper_alive_bound, upper_dead_bound = self.find_extreme_ranges(
                satellite, component, lower_bound, upper_bound
            )

            # Skip if the entire range is dead
            if lower_alive_bound == DEAD_RANGE_MARKER or upper_alive_bound == DEAD_RANGE_MARKER:
                # Define a smaller range around satellite[component]
                delta = (upper_bound - lower_bound) / self.n_sat
                lower_value = max(float(satellite[component]) - delta, lower_bound)
                upper_value = min(float(satellite[component]) + delta, upper_bound)

                # Try finding extreme ranges within this smaller range
                lower_alive_bound, lower_dead_bound, upper_alive_bound, upper_dead_bound = self.find_extreme_ranges(
                    satellite, component, lower_value, upper_value
                )

                # If still dead after refining, skip this component
                if lower_alive_bound == DEAD_RANGE_MARKER or upper_alive_bound == DEAD_RANGE_MARKER:
                    continue
    
            for _ in range(n_iterations - 1):
                lower_alive_bound, lower_dead_bound, _, _ = self.find_extreme_ranges(
                    satellite, component, lower_alive_bound, lower_dead_bound
                )
                _, _, upper_alive_bound, upper_dead_bound = self.find_extreme_ranges(
                    satellite, component, upper_alive_bound, upper_dead_bound
                )

            # Only append if lower_alive_bound is not 1000
            if lower_alive_bound != DEAD_RANGE_MARKER:
                variants.append([*satellite[:component], lower_alive_bound, *satellite[component + 1:]])

            # Only append if upper_alive_bound is not 1000
            if upper_alive_bound != DEAD_RANGE_MARKER:
                variants.append([*satellite[:component], upper_alive_bound, *satellite[component + 1:]])

        return variants

    def generate_config_within_bounds(self) -> List[float]:
        """
        Generate a random configuration within the bounds of the UDP instance.

        Returns:
            List[float]: A configuration that respects the UDP bounds.
        """
        return [random.uniform(lower, upper) for lower, upper in zip(self.lower_bounds, self.upper_bounds)]

    def decompose(self, config: List[float]) -> List[List[float]]:
        """
        Decomposes the configuration into a list of satellite parameters for all satellites.

        Args:
            config (List[float]): The satellite configuration.

        Returns:
            List[List[float]]: A list containing [x, y, z, xv, yv, zv] for each satellite.
        """
        return [self.get_all_components(config, index) for index in range(self.n_sat)]

class DistanceMetric:
    @staticmethod
    def get_metrics() -> list[str]:
        """
        Return a list of available metric names.

        Returns:
            list: A list of metric names.
        """
        return list(DistanceMetric.get_metric_functions().keys())

    @staticmethod
    def get_metric_functions() -> dict:
        """
        Return a dictionary mapping metric names to their corresponding functions.

        Returns:
            dict: A dictionary of available metric functions.
        """
        return {
            "manhattan": DistanceMetric.manhattan,
            "geometric": DistanceMetric.geometric,
            "velocity": DistanceMetric.velocity,
            "position": DistanceMetric.position,
            "orbital": DistanceMetric.orbital
        }

    @staticmethod
    def calculate(sat1: List[int], sat2: List[int], metric_name: str) -> float:
        """
        Calculate the distance between two satellites using the specified metric.

        Args:
            sat1: List of [x, y, z, xv, yv, zv] for the first satellite.
            sat2: List of [x, y, z, xv, yv, zv] for the second satellite.
            metric_name: Name of the metric to use for distance calculation.
        
        Returns:
            float: The distance between the satellites' parameters.

        Raises:
            ValueError: If the metric_name is not recognized.
        """
        metrics = DistanceMetric.get_metric_functions()

        if metric_name not in metrics:
            raise ValueError(f"Unknown metric: {metric_name}. Valid options are: {list(metrics.keys())}")

        return metrics[metric_name](sat1, sat2)

    @staticmethod
    def manhattan(sat1: List[int], sat2: List[int]) -> float:
        """Calculate the Manhattan distance (L1 distance) between two satellites' parameters."""
        return sum(abs(a - b) for a, b in zip(sat1, sat2))

    @staticmethod
    def geometric(sat1: List[int], sat2: List[int]) -> float:
        """Calculate the Euclidean (geometric) distance between two satellites."""
        position_distance = math.sqrt(
            (sat1[0] - sat2[0]) ** 2 +
            (sat1[1] - sat2[1]) ** 2 +
            (sat1[2] - sat2[2]) ** 2
        )
        velocity_distance = math.sqrt(
            (sat1[3] - sat2[3]) ** 2 +
            (sat1[4] - sat2[4]) ** 2 +
            (sat1[5] - sat2[5]) ** 2
        )
        return position_distance + velocity_distance

    @staticmethod
    def velocity(sat1: List[int], sat2: List[int]) -> float:
        """Calculate the Euclidean distance between two satellites based on their velocities."""
        return math.sqrt(
            (sat1[3] - sat2[3]) ** 2 +
            (sat1[4] - sat2[4]) ** 2 +
            (sat1[5] - sat2[5]) ** 2
        )

    @staticmethod
    def position(sat1: List[int], sat2: List[int]) -> float:
        """Calculate the Euclidean distance between two satellites' positions."""
        return math.sqrt(
            (sat1[0] - sat2[0]) ** 2 +
            (sat1[1] - sat2[1]) ** 2 +
            (sat1[2] - sat2[2]) ** 2
        )

    @staticmethod
    def orbital(sat1: List[int], sat2: List[int]) -> float:
        """Calculate the orbital distance between two satellites."""
        radial_distance_1 = math.sqrt(sat1[0] ** 2 + sat1[1] ** 2 + sat1[2] ** 2)
        radial_distance_2 = math.sqrt(sat2[0] ** 2 + sat2[1] ** 2 + sat2[2] ** 2)
        velocity_magnitude_1 = math.sqrt(sat1[3] ** 2 + sat1[4] ** 2 + sat1[5] ** 2)
        velocity_magnitude_2 = math.sqrt(sat2[3] ** 2 + sat2[4] ** 2 + sat2[5] ** 2)
        return abs(radial_distance_1 - radial_distance_2) + abs(velocity_magnitude_1 - velocity_magnitude_2)

    @staticmethod
    def find_average(drone: List[int], drones: List[List[int]], metric_name: str) -> float:
        """
        Finds the average distance between a drone and a list of drones.
        """
        distances = [DistanceMetric.calculate(drone, other_drone, metric_name) for other_drone in drones]
        return sum(distances) / len(distances) if distances else 0

    @staticmethod
    def find_most_similar(drone: List[int], drones: List[List[int]], metric_name: str) -> List[int]:
        """
        Finds the most similar drone to the given drone based on the specified distance metric.
        """
        return min(drones, key=lambda other_drone: DistanceMetric.calculate(drone, other_drone, metric_name))

    @staticmethod
    def find_least_similar(drone: List[int], drones: List[List[int]], metric_name: str) -> List[int]:
        """
        Finds the least similar drone to the given drone based on the specified distance metric.
        """
        return max(drones, key=lambda other_drone: DistanceMetric.calculate(drone, other_drone, metric_name))

class ChromosomeFactory:
    def __init__(self, udp, satellite_pool: List[List[float]] = None, pool_name: str = "NoPool", generation_method: str = "differentiated", metric_name: str = "manhattan"):
        """
        Initialize the ChromosomeFactory with the necessary parameters.

        Args:
            satellite_pool (List[List[float]]): A list of satellite configurations.
            udp: The UDP instance containing satellite information (e.g., tiny_dro_udp, large_dro_udp).
            generation_method (str): The generation method to use (e.g., "differentiated", "alive_pool", "random_bounds").
        """
        self.udp = udp
        self.satellite_pool = satellite_pool if satellite_pool != None else []
        self.pool_name = pool_name
        self.n_sat = udp.n_sat
        self.generation_method = generation_method
        self.metric_name = metric_name
        self.chromosome_controller = ChromosomeController(self.udp)
        self.grid = []  # Initialize the grid attribute

    def set_grid(self, grid: List[List[int]]):
        """
        Set the grid for chromosome generation.

        Args:
            grid (List[List[int]]): The grid to set.
        """
        self.grid = grid

    def get_grid(self) -> List[List[int]]:
        """
        Get the grid, generating it if it does not already exist.

        Returns:
            List[List[int]]: The grid for chromosome generation.
        """
        if not self.grid:
            print("Grid is not set. Generating grid...")
            self.grid = self.generate_grid()  # Generate the grid if it isn't set
        return self.grid

    def generate_grid(self) -> List[List[int]]:
        """
        Generate the grid based on the satellite pool.

        Returns:
            List[List[int]]: The generated grid.
        """
        # Implement logic to generate grid positions based on satellite configurations
        # For example, use the udp to calculate grid positions for each satellite in the pool
        return [
            self.udp.get_grid_positions_at_times_single_satellite(satellite)
            for satellite in self.satellite_pool
        ]

    @staticmethod
    def get_generation_methods() -> dict:
        """
        Return a dictionary of available generation methods.

        Returns:
            dict: A dictionary mapping method names to their corresponding functions.
        """
        return {
            "from-bounds": ChromosomeFactory.from_bounds,
            "from-pool": ChromosomeFactory.from_pool,
            "from-pool-differentiated": ChromosomeFactory.from_pool_differentiated,
            # "from-pool-squad-based": ChromosomeFactory.from_pool_squad_based,
            "from-pool-best_squad_based": ChromosomeFactory.from_pool_best_squad_based,
            # "from-pool-fitness-based": ChromosomeFactory.from_pool_fitness_based,
            # "from-pool-hybrid-differentiated-and-fitness": ChromosomeFactory.from_pool_hybrid_differentiated_and_fitness,
            # "from-pool-fitness-based-top-distant": ChromosomeFactory.from_pool_fitness_based_top_distant
            "from-pool-fitness-based-growing-subset": ChromosomeFactory.from_pool_fitness_based_growing_subset,
            "from-grid": ChromosomeFactory.from_grid
        }

    @staticmethod
    def get_generation_method_keys() -> list:
        """
        Return a list of available generation method names.

        Returns:
            list: A list of method names.
        """
        return list(ChromosomeFactory.get_generation_methods().keys())

    def create_chromosome_from_indices(self, indices: List[int]) -> List[float]:
        """
        Create a new chromosome based on the specified indices from the satellite pool.

        Args:
            indices (List[int]): A list of indices specifying which satellites to use.

        Returns:
            List[float]: A new chromosome created from the specified satellites in the satellite pool.
        """
        new_chromosome = [0.0 for _ in range(self.n_sat * 6)]
        for satellite_index, selected_index in enumerate(indices):
            selected_satellite = self.satellite_pool[selected_index]
            self.chromosome_controller.set_all_components(new_chromosome, satellite_index, selected_satellite)
        return new_chromosome

    def __call__(self):
        return self.generate()

    def get_name(self) -> str:
        """
        Return a meaningful name for this ChromosomeFactory based on its configuration.

        Returns:
            str: A descriptive name for the factory, e.g., 'from_pool_differentiated_manhattan' or 'from_pool_differentiated_tiny_pool_manhattan'.
        """
        if "pool" in self.generation_method and self.pool_name:
            if "differentiated" in self.generation_method:
                return f"{self.generation_method}-{self.pool_name}-{self.metric_name}"
            else:
                return f"{self.generation_method}-{self.pool_name}"
        else:
            return self.generation_method

    def generate(self) -> List[float]:
        """
        Generate a chromosome based on the specified generation type.

        Returns:
            List[float]: A new chromosome generated using the chosen method.
        """
        methods = ChromosomeFactory.get_generation_methods()

        if self.generation_method not in methods:
            raise ValueError(f"Unknown generation method: {self.generation_method}. Valid options are: {self.get_generation_methods()}")

        return methods[self.generation_method](self)

    def from_bounds(self) -> List[float]:
        """
        Create a random chromosome within the bounds defined by the UDP instance.

        Returns:
            List[float]: A new random chromosome within the UDP's bounds.
        """
        lower_bounds, upper_bounds = self.udp.get_bounds()
        return [random.uniform(lower, upper) for lower, upper in zip(lower_bounds, upper_bounds)]

    def from_pool(self) -> List[float]:
        random_indices = random.sample(range(len(self.satellite_pool)), self.n_sat)
        return self.create_chromosome_from_indices(random_indices)

    def from_pool_differentiated(self, metric_name: str = "manhattan") -> List[float]:
        """
        Create a new chromosome by selecting the most differentiated satellites from the alive satellite list.

        Args:
            metric_name (str): The name of the distance metric to use (e.g., "manhattan", "geometric").

        Returns:
            List[float]: A new chromosome with differentiated alive satellite data.
        """
        new_chromosome = [0.0 for _ in range(self.n_sat * 6)]

        # Select the first satellite randomly
        selected_indices = [random.randint(0, len(self.satellite_pool) - 1)]
        first_satellite = self.satellite_pool[selected_indices[0]]

        # Calculate closeness using the specified metric
        closeness = [DistanceMetric.calculate(first_satellite, satellite, metric_name)
                     for satellite in self.satellite_pool]

        # Select the most differentiated satellites
        for _ in range(self.n_sat - 1):
            selected_indices.append(closeness.index(max(closeness)))
            selected_satellite = self.satellite_pool[selected_indices[-1]]

            # Update closeness
            for index in range(len(closeness)):
                close = DistanceMetric.calculate(self.satellite_pool[index], selected_satellite, metric_name)
                if close < closeness[index]:
                    closeness[index] = close

        # Populate the new chromosome
        for index in range(len(selected_indices)):
            satellite = self.satellite_pool[selected_indices[index]]
            self.chromosome_controller.set_all_components(new_chromosome, index, satellite)

        return new_chromosome

    def from_pool_fitness_based(self) -> List[float]:
        """
        Create a new chromosome by selecting the most fitness-improving satellites from the pool.
        
        Args:
            metric_name (str): The name of the distance metric to use for initial differentiation (optional).
        
        Returns:
            List[float]: A new chromosome with satellites selected based on udp.fitness improvements.
        """
        # Start with a dead config (all satellites are set to an invalid state)
        new_chromosome = self.make_dead_config(self.n_sat)
        
        selected_indices = []
        
        # Loop to select n_sat satellites based on fitness improvements
        for _ in range(self.n_sat):
            best_satellite_index = None
            best_fitness = float('inf')  # Start with a very high fitness

            # Get a random subset of 50 satellite indices from the pool
            subset_size = min(5, len(self.satellite_pool))  # Ensure we don't exceed the pool size
            random_indices = random.sample(range(len(self.satellite_pool)), subset_size)

            # Iterate through the random subset of satellites
            for i in random_indices:
                if i in selected_indices:
                    continue  # Skip already selected satellites
                
                satellite = self.satellite_pool[i]
                
                # Create a temporary config with the current satellite added
                temp_config = new_chromosome[:]
                self.set_all_components(temp_config, len(selected_indices), satellite)
                
                # Compute fitness of the temporary configuration
                fitness_value = self.udp.fitness(temp_config)[0]
                
                # Select the satellite that improves the fitness the most
                if fitness_value < best_fitness:
                    best_fitness = fitness_value
                    best_satellite_index = i

            # Add the best satellite found to the new chromosome
            if best_satellite_index is not None:
                selected_indices.append(best_satellite_index)
                self.set_all_components(new_chromosome, len(selected_indices) - 1, self.satellite_pool[best_satellite_index])
        
        return new_chromosome

    def from_pool_fitness_based_growing_subset(self) -> List[float]:
        """
        Create a new chromosome by selecting the most fitness-improving satellites from the pool.
        The subset size for fitness evaluation grows with each iteration, culminating in full pool evaluation.
        
        Returns:
            List[float]: A new chromosome with satellites selected based on udp.fitness improvements.
        """
        # Start with a dead config (all satellites are set to an invalid state)
        new_chromosome = self.chromosome_controller.make_dead_config()
        
        selected_indices = []
        pool_size = len(self.satellite_pool)

        # Constants to control the subset size growing behavior
        INITIAL_SUBSET_SIZE = 5  # Start with a very small subset
        FINAL_SUBSET_FULL_POOL = 5  # Number of satellites selected from the full pool
        step_increase = (pool_size - INITIAL_SUBSET_SIZE) // (self.n_sat - FINAL_SUBSET_FULL_POOL)  # Increment for subset size

        # Loop to select n_sat satellites based on fitness improvements
        for selection_step in range(self.n_sat):
            best_satellite_index = None
            best_fitness = float('inf')  # Start with a very high fitness

            # For the last FINAL_SUBSET_FULL_POOL satellites, use the entire pool
            if selection_step >= self.n_sat - FINAL_SUBSET_FULL_POOL:
                subset_size = pool_size
            else:
                # Grow the subset size gradually
                subset_size = INITIAL_SUBSET_SIZE + step_increase * selection_step

            # Get a random subset of satellite indices from the pool
            random_indices = random.sample(range(pool_size), min(subset_size, pool_size))

            # Iterate through the random subset of satellites
            for i in random_indices:
                if i in selected_indices:
                    continue  # Skip already selected satellites

                satellite = self.satellite_pool[i]
                
                # Create a temporary config with the current satellite added
                temp_config = new_chromosome[:]
                self.chromosome_controller.set_all_components(temp_config, len(selected_indices), satellite)
                
                # Compute fitness of the temporary configuration
                fitness_value = self.udp.fitness(temp_config)[0]
                
                # Select the satellite that improves the fitness the most
                if fitness_value < best_fitness:
                    best_fitness = fitness_value
                    best_satellite_index = i

            # Add the best satellite found to the new chromosome
            if best_satellite_index is not None:
                selected_indices.append(best_satellite_index)
                self.chromosome_controller.set_all_components(new_chromosome, len(selected_indices) - 1, self.satellite_pool[best_satellite_index])

        return new_chromosome

    def from_pool_fitness_based_top_distant(self, metric_name: str = "manhattan") -> List[float]:
        """
        Create a new chromosome by selecting the most fitness-improving satellites from the top distant ones.
        
        Args:
            metric_name (str): The name of the distance metric to use for selecting the most distant satellites.
        
        Returns:
            List[float]: A new chromosome with satellites selected based on distance first, then fitness.
        """
        new_chromosome = self.make_dead_config()
        selected_indices = []
        
        # Step 1: Select the most distant satellites based on the metric
        first_satellite_index = random.randint(0, len(self.satellite_pool) - 1)
        first_satellite = self.satellite_pool[first_satellite_index]

        # Calculate closeness using the specified metric and select the most distant
        closeness = [DistanceMetric.calculate(first_satellite, satellite, metric_name)
                    for satellite in self.satellite_pool]
        
        # Get the top DISTANT_SELECTION_COUNT most distant satellites
        top_distant_indices = sorted(range(len(closeness)), key=lambda x: closeness[x], reverse=True)[:DISTANT_SELECTION_COUNT]

        # Step 2: Fitness-based selection from the top distant satellites
        for _ in range(self.n_sat):
            best_satellite_index = None
            best_fitness = float('inf')

            for i in top_distant_indices:
                if i in selected_indices:
                    continue

                satellite = self.satellite_pool[i]
                temp_config = new_chromosome[:]
                self.set_all_components(temp_config, len(selected_indices), satellite)

                # Compute fitness of the temporary configuration
                fitness_value = self.udp.fitness(temp_config)[0]

                # Select the satellite that improves the fitness the most
                if fitness_value < best_fitness:
                    best_fitness = fitness_value
                    best_satellite_index = i

            # Add the best satellite found to the new chromosome
            if best_satellite_index is not None:
                selected_indices.append(best_satellite_index)
                self.set_all_components(new_chromosome, len(selected_indices) - 1, self.satellite_pool[best_satellite_index])

        return new_chromosome
    
    def from_pool_hybrid_differentiated_and_fitness(self, metric_name: str = "manhattan") -> List[float]:
        """
        Create a new chromosome by selecting a subset via differentiation, then using random fitness-based selection.

        Args:
            metric_name (str): The name of the distance metric to use for the differentiated selection.
        
        Returns:
            List[float]: A new chromosome with satellites selected first via differentiation, then fitness-based.
        """
        new_chromosome = self.make_dead_config()
        selected_indices = []

        # Step 1: Select the first DIFFERENTIATED_SELECTION_COUNT  satellites via differentiation
        first_satellite_index = random.randint(0, len(self.satellite_pool) - 1)
        first_satellite = self.satellite_pool[first_satellite_index]

        # Calculate closeness using the specified metric
        closeness = [DistanceMetric.calculate(first_satellite, satellite, metric_name)
                    for satellite in self.satellite_pool]
        
        top_distant_indices = sorted(range(len(closeness)), key=lambda x: closeness[x], reverse=True)[:DIFFERENTIATED_SELECTION_COUNT ]

        # Add these differentiated satellites to the new chromosome
        for index in top_distant_indices:
            if len(selected_indices) < self.n_sat:
                selected_indices.append(index)
                self.set_all_components(new_chromosome, len(selected_indices) - 1, self.satellite_pool[index])

        # Step 2: Select the remaining satellites based on random subset fitness evaluation
        while len(selected_indices) < self.n_sat:
            best_satellite_index = None
            best_fitness = float('inf')

            # Get a random subset of RANDOM_SELECTION_SUBSET_SIZE  satellite indices from the pool
            random_indices = random.sample(range(len(self.satellite_pool)), RANDOM_SELECTION_SUBSET_SIZE )

            for i in random_indices:
                if i in selected_indices:
                    continue

                satellite = self.satellite_pool[i]
                temp_config = new_chromosome[:]
                self.set_all_components(temp_config, len(selected_indices), satellite)

                # Compute fitness of the temporary configuration
                fitness_value = self.udp.fitness(temp_config)[0]

                # Select the satellite that improves the fitness the most
                if fitness_value < best_fitness:
                    best_fitness = fitness_value
                    best_satellite_index = i

            # Add the best satellite found to the new chromosome
            if best_satellite_index is not None:
                selected_indices.append(best_satellite_index)
                self.set_all_components(new_chromosome, len(selected_indices) - 1, self.satellite_pool[best_satellite_index])

        return new_chromosome

    def from_pool_squad_based(self, squad_size: int = 5, final_size: int = 40, num_initial_squads: int = 100) -> List[float]:
        """
        Create a new chromosome using a squad-based approach. Squads of satellites are evaluated,
        and the best squads are combined into larger groups until the final group of `final_size` satellites is formed.
        
        Args:
            squad_size (int): Number of satellites per initial squad (e.g., 5).
            final_size (int): Final number of satellites in the chromosome (e.g., 40).
            num_initial_squads (int): Number of initial squads to evaluate.
        
        Returns:
            List[float]: A new chromosome formed by the best-performing squads.
        """
        selected_indices = []
        
        # Step 1: Create initial squads and evaluate fitness
        initial_squads = []
        for _ in range(num_initial_squads):
            if len(self.satellite_pool) < squad_size:
                raise ValueError("Satellite pool size is too small to form the required squad size.")
            
            squad = random.sample(range(len(self.satellite_pool)), squad_size)
            temp_config = self.make_dead_config()
            
            # Set squad components and calculate fitness
            for index in range(squad_size):
                satellite = self.satellite_pool[squad[index]]
                self.set_all_components(temp_config, index, satellite)
            
            fitness_value = self.udp.fitness(temp_config)[0]
            initial_squads.append((squad, fitness_value))
        
        if not initial_squads:
            raise ValueError("No valid squads were created.")
        
        # Sort initial squads by fitness (best first)
        initial_squads.sort(key=lambda x: x[1])
        best_squads = initial_squads[:num_initial_squads // 2]  # Take top 50%
        
        # Step 2: Repeat the process of combining squads until reaching final size
        while len(best_squads[0][0]) < final_size:
            best_squads = self.combine_squads(best_squads)  # Combine squads until they reach final_size
        
        # Step 3: Create the final chromosome from the best-performing squad (size 40)
        best_final_squad = best_squads[0][0]
        new_chromosome = self.make_dead_config()
        
        for idx, satellite_index in enumerate(best_final_squad):
            satellite = self.satellite_pool[satellite_index]
            self.set_all_components(new_chromosome, idx, satellite)

        return new_chromosome

    def calculate_fitness_from_indices(self, satellite_indices: List[int]) -> float:
        """
        Calculate the fitness for a given list of satellite indices.

        Args:
            satellite_indices (List[int]): A list of satellite indices to include in the configuration.

        Returns:
            float: The fitness value of the configuration.
        """
        temp_config = self.chromosome_controller.make_dead_config()
        
        # Set the satellite components in the config
        for idx, satellite_index in enumerate(satellite_indices):
            if idx >= self.n_sat:  # Ensure we don't exceed the number of satellites
                break
            satellite = self.satellite_pool[satellite_index]
            self.chromosome_controller.set_all_components(temp_config, idx, satellite)
        
        # Calculate and return the fitness of the configuration
        fitness_value = self.udp.fitness(temp_config)[0]
        return fitness_value

    def from_pool_best_squad_based(self):
        squads = self.make_squads(10, 40)
        # print(squads[0])
        # print(squads[-1])
        twenty = self.combine_groups(squads, squads)
        # print(twenty[0])
        # print(twenty[-1])
        # print(len(twenty))
        thirty = self.combine_groups(twenty, squads)
        # print(thirty[0])
        # print(thirty[-1])
        # print(len(thirty))
        forty = self.combine_groups(thirty, squads)
        # print(forty[0])
        # print(forty[-1])
        # print(len(forty))

        config = self.chromosome_controller.make_dead_config()
        
        # Set the satellite components in the config
        for idx, satellite_index in enumerate(forty[0][0]):
            if idx >= self.n_sat:  # Ensure we don't exceed the number of satellites
                break
            satellite = self.satellite_pool[satellite_index]
            self.chromosome_controller.set_all_components(config, idx, satellite)
        return config
    
    def make_squads(self, size = 5, number_of_squads = 100):
        squads = []
        indices = list(range(len(self.satellite_pool)))
        found_fitnesses = [0.0 for _ in range(10)]

        while len(squads) < number_of_squads:
            squad = random.sample(indices, size)
            squad.sort()
            fitness = self.calculate_fitness_from_indices(squad)

            if fitness not in found_fitnesses:
                found_fitnesses.append(fitness)
                found_fitnesses.sort()
                squads = [squad for squad in squads if squad[1] <= found_fitnesses[9]]
                # if fitness <= found_fitnesses[9]:
                #     print(f"Found fitness: {fitness}, len{len(squads)}")

            if fitness <= found_fitnesses[9]:
                squads.append((squad, fitness))
            #     best_fitness = fitness
            #     squads = [(squad, best_fitness)]
            #     print(fitness)
            # elif fitness == best_fitness:
            #     squads.append((squad, fitness))
            #     squads.append((squad, fitness))
            #     squads.append((squad, fitness))
        squads.sort(key=lambda x: x[1])
        return squads

    def combine_groups(self, group_A: List[Tuple[List[int], float]], group_B: List[Tuple[List[int], float]]) -> List[Tuple[List[int], float]]:
        """
        Combine each squad from group A with each squad from group B, ensuring no overlapping satellites.
        Only append squads with the best fitness, and restart the output if a better fitness is found.

        Args:
            group_A (List[Tuple[List[int], float]]): A list of (squad, fitness) tuples where squad is a list of satellite indices.
            group_B (List[Tuple[List[int], float]]): A list of (squad, fitness) tuples where squad is a list of satellite indices.

        Returns:
            List[Tuple[List[int], float]]: A list of new squads with the best fitness, sorted by fitness.
        """
        combined_squads = []
        best_fitness = float('inf')  # Start with a high best fitness

        # Iterate over each squad in group A
        for squad_A, fitness_A in group_A:
            # Iterate over each squad in group B
            for squad_B, fitness_B in group_B:
                # Ensure no overlap in satellites between squad_A and squad_B
                if len(set(squad_A).intersection(set(squad_B))) == 0:
                    # Combine the squads
                    combined_squad = squad_A + squad_B
                    combined_squad = combined_squad[:self.n_sat]  # Limit to max size of `n_sat`
                    
                    # Create the config for fitness calculation
                    temp_config = self.chromosome_controller.make_dead_config()
                    for idx, satellite_index in enumerate(combined_squad):
                        satellite = self.satellite_pool[satellite_index]
                        self.chromosome_controller.set_all_components(temp_config, idx, satellite)

                    # Calculate the fitness of the combined squad
                    fitness_value = self.udp.fitness(temp_config)[0]
                    
                    # # If a new best fitness is found, reset the output list
                    # if fitness_value < best_fitness:
                    #     best_fitness = fitness_value
                    #     combined_squads = [(combined_squad, fitness_value)]  # Reset with the new best squad
                    # elif fitness_value == best_fitness:
                    #     if sorted(combined_squads) == combined_squads:
                    #         # If the fitness matches the best, add it to the list
                    combined_squads.append((combined_squad, fitness_value))

        # Sort combined squads by fitness (optional if best fitness is always the same)
        combined_squads.sort(key=lambda x: x[1])  # Best fitness first
        combined_squads = combined_squads[:50]
        return combined_squads
    
    def combine_squads(self, best_squads: List[Tuple[List[int], float]]) -> List[Tuple[List[int], float]]:
        """
        Combine the best-performing squads to form larger squads.

        Args:
            best_squads (List[Tuple[List[int], float]]): A list of (squad, fitness) tuples where squad is a list of satellite indices.
        
        Returns:
            List[Tuple[List[int], float]]: The new combined squads, sorted by fitness.
        """
        new_squads = []

        # Combine squads in pairs
        for i in range(0, len(best_squads), 2):
            if i + 1 < len(best_squads):
                combined_squad = best_squads[i][0] + best_squads[i + 1][0]
                combined_squad = combined_squad[:self.n_sat]  # Limit to max size of `n_sat`
                
                temp_config = self.chromosome_controller.make_dead_config(self.n_sat)
                for idx, satellite_index in enumerate(combined_squad):
                    satellite = self.satellite_pool[satellite_index]
                    self.chromosome_controller.set_all_components(temp_config, idx, satellite, self.n_sat)
                
                fitness_value = self.udp.fitness(temp_config)[0]
                new_squads.append((combined_squad, fitness_value))
        
        # Sort new squads by fitness and select the top 50% again
        new_squads.sort(key=lambda x: x[1])  # Best fitness first
        return new_squads#[:len(new_squads) // 2]
    @staticmethod
    def create_all_methods(udp, satellite_pools: dict) -> dict:
        """
        Create all possible ChromosomeFactory instances with different generation methods and metrics.
        For methods containing 'pool', a factory will be created for each pool.
        For methods not containing 'pool', the satellite_pool will be empty.

        Args:
            udp: The UDP instance containing satellite information (e.g., tiny_dro_udp, large_dro_udp).
            satellite_pools: A dictionary where keys are pool names and values are lists of satellite configurations.

        Returns:
            dict: A dictionary where keys are descriptive names and values are ChromosomeFactory instances.
        """
        methods = ChromosomeFactory.get_generation_method_keys()
        metrics = DistanceMetric.get_metrics()

        all_factories = {}

        for method in methods:
            if "pool" in method:
                # For methods containing 'pool', create a factory for each pool
                for pool_name, satellite_pool in satellite_pools.items():
                    for metric in metrics:
                        # Differentiated methods need metrics
                        if "differentiated" in method:
                            factory = ChromosomeFactory(udp, satellite_pool, pool_name=pool_name, generation_method=method, metric_name=metric)
                            factory_name = factory.get_name()
                        else:
                            factory = ChromosomeFactory(udp, satellite_pool, pool_name=pool_name, generation_method=method)
                            factory_name = factory.get_name()

                        all_factories[factory_name] = factory
            else:
                # For methods not containing 'pool', create a single factory with an empty satellite pool
                factory = ChromosomeFactory(udp, generation_method=method)
                all_factories[method] = factory

        return all_factories

    def generate_fitness_data(self, number_of_factories: int = 0) -> List[float]:
        """
        Generate fitness data using the factory and return fitness values.
        If number_of_factories is specified, it will generate that many chromosomes.
        Otherwise, it will continue until the change in the running average is less than 1%.

        Args:
            number_of_factories (int): Number of chromosomes to generate (optional).

        Returns:
            List[float]: List of fitness values for the generated chromosomes.
        """
        fitness_values = []

        # If a specific number of factories is given, generate that many
        if number_of_factories > 0:
            start = time.time()
            for i in range(number_of_factories):
                fitness_values.append(self.udp.fitness(self.generate())[0])  # Assuming fitness returns a list
                # Print progress bar
                print_progress_bar_with_time(i + 1, number_of_factories, start, prefix='Progress:', length=50)
            return fitness_values
        
        if self.generation_method == "from-bounds":
            for i in range(100):
                fitness_values.append(self.udp.fitness(self.generate())[0])  # Assuming fitness returns a list
            return fitness_values

        # If no number is specified, stop when the change in average is less than 1%
        running_total = 0
        count = 0
        previous_average = 1  # Initialize with a large value for comparison
        current_average = 0

        while abs(previous_average - current_average) >= previous_average / 10:  # 1% change condition
            fitness_value = self.udp.fitness(self.generate())[0]
            fitness_values.append(fitness_value)

            # Update running total and count for calculating the average
            running_total += fitness_value
            count += 1

            # Update averages
            previous_average = current_average
            current_average = running_total / count

        return fitness_values

    def from_grid(self):
        grid_factory = GridBasedChromosomeFactory(
            udp=self.udp,
            grid=self.get_grid(),
            n_sat=self.n_sat
        )

        selected_indices = grid_factory()

        return self.create_chromosome_from_indices(selected_indices)

class PopulationFactory:
    def __init__(self, chromosome_factory: ChromosomeFactory, problem):
        """
        Initialize the PopulationFactory with a ChromosomeFactory instance.
        """
        self.chromosome_factory = chromosome_factory
        self.problem = problem

    def __call__(self, population_size: int = 100) -> "pg.population":
        """
        Generate a new population using the ChromosomeFactory.

        Args:
            population_size (int): The number of individuals to generate.
            problem: The pygmo problem to initialize the population.

        Returns:
            pg.population: A new population with chromosomes generated from the ChromosomeFactory.
        """
        new_population = pg.population(self.problem, population_size)
        start_time = time.time()

        for i in range(population_size):
            # Generate a chromosome and set it in the population
            new_chromosome = self.chromosome_factory.generate()
            new_population.set_x(i, new_chromosome)
            print_progress_bar_with_time(i + 1, population_size, time_started=start_time, prefix='Progress:', suffix='', length=50)

        return new_population

class SatellitePoolGenerator:
    def __init__(self, udp, quiet: bool):
        """
        Initialize the SatellitePoolGenerator with the necessary parameters.

        Args:
            udp: The UDP instance containing satellite information.
            quiet (bool): Whether or not to suppress output.
        """
        self.udp = udp
        self.n_sat = udp.n_sat
        self.quiet = quiet
        self.chromosome_controller = ChromosomeController(udp)

    def collect_alive_satellites(self, target: int) -> List[List[float]]:
        """
        Collect a list of alive satellites based on the target number.

        Args:
            target (int): The number of alive satellites to collect.

        Returns:
            List[List[float]]: A list of satellite configurations.
        """
        # Your logic for collecting alive satellites would go here.
        satellite_pool = []
        found = 0
        generated = 0

        start_time = time.time()
        print_progress_bar_with_time(found, target, time_started=start_time, prefix='Progress:', suffix='', length=50)
        while found < target:
            config = self.chromosome_controller.generate_config_within_bounds()
            alive = self.chromosome_controller.extract_alive_satellites(config)

            generated += self.n_sat
            if alive != []:
                found += len(alive)
                satellite_pool.extend(alive)

            # Prevent division by zero
            generation_percentage = (found / generated * 100) if generated > 0 else 0
            print_progress_bar_with_time(found, target, time_started=start_time, prefix='', suffix=f"{generation_percentage:.2g}%    ", length=50)

        return satellite_pool[:target]

    def generate_random_config_within_bounds_single_satellite(self) -> List[float]:
        """
        Generate a random configuration within bounds for a single satellite.

        Returns:
            List[float]: A list representing the satellite's configuration [x0, y0, z0, vx0, vy0, vz0].
        """
        lower_bounds, upper_bounds = self.udp.get_bounds()
        # Since the bounds are for all satellites, extract bounds for a single satellite
        lower_bounds_single = lower_bounds[:6]
        upper_bounds_single = upper_bounds[:6]
        # Generate random values within the bounds
        config = [random.uniform(lb, ub) for lb, ub in zip(lower_bounds_single, upper_bounds_single)]
        return config

    def collect_unique_alive_satellites(self, target: int) -> List[List[float]]:
        """
        Collect a list of unique alive satellites based on their grid positions at t=0, t=1, t=2.

        Args:
            target (int): The number of unique alive satellites to collect.

        Returns:
            List[List[float]]: A list of satellite configurations.
        """
        satellite_pool = []
        unique_grid_positions = set()
        found = 0
        generated = 0

        start_time = time.time()
        if not self.quiet:
            print(f"Collecting {target} unique alive satellites...")
        while found < target:
            # Generate a random configuration within bounds for a single satellite
            config = self.generate_random_config_within_bounds_single_satellite()
            # Get the grid positions at times t=0, t=1, t=2
            grid_positions = self.udp.get_grid_positions_at_times_single_satellite(config)
            # grid_positions is a numpy array of shape (3, 3)

            # Check if the satellite is alive at all times (not out of bounds)
            is_alive = all(pos[0] is not None for pos in grid_positions)
            if not is_alive:
                generated += 1
                continue  # Skip satellites that are out of bounds at any time

            # Convert grid_positions to a tuple of tuples for hashing
            grid_positions_tuple = tuple(tuple(pos) for pos in grid_positions)

            if grid_positions_tuple not in unique_grid_positions:
                unique_grid_positions.add(grid_positions_tuple)
                satellite_pool.append(config)
                found += 1

                if not self.quiet:
                    elapsed_time = time.time() - start_time
                    print_progress_bar_with_time(found, target, time_started=start_time, prefix='', suffix='', length=50)

                    # print(f"Found {found}/{target} unique satellites. Time elapsed: {elapsed_time:.2f}s")

            generated += 1

        if not self.quiet:
            total_time = time.time() - start_time
            print(f"Finished collecting {found} unique satellites. Total time: {total_time:.2f}s")
            print(f"Total configurations generated: {generated}")

        return satellite_pool

    def collect_satellites_by_nones(self, target: int) -> Dict[int, List[List[float]]]:
        satellites_by_nones = {nones: [] for nones in range(5)}
        total_targets = {nones: target for nones in range(5)}
        total_found = {nones: 0 for nones in range(5)}
        unique_positions_by_nones = {nones: set() for nones in range(5)}

        generated = 0
        start_time = time.time()

        while any(total_found[nones] < total_targets[nones] for nones in range(5)):
            config = self.generate_random_config_within_bounds_single_satellite()
            grid_positions = self.udp.get_grid_positions_at_times_single_satellite(config)

            num_nones = sum(1 for pos in grid_positions if pos[0] is None)

            if num_nones > 4:
                generated += 1
                continue

            if total_found[num_nones] >= total_targets[num_nones]:
                generated += 1
                continue

            # Use position at t=1 for uniqueness
            if grid_positions[1][0] is not None:
                position_t1 = tuple(grid_positions[1])
                if position_t1 not in unique_positions_by_nones[num_nones]:
                    unique_positions_by_nones[num_nones].add(position_t1)
                    satellites_by_nones[num_nones].append(config)
                    total_found[num_nones] += 1
            else:
                # Satellite not alive at t=1; decide how to handle
                continue

            generated += 1

        return satellites_by_nones

    def create_variation(self, satellite_pool: List[List[float]]):
        new_pool = []
        for satellite in satellite_pool:
            new_pool.extend(self.chromosome_controller.generate_satellite_variants(satellite))
        return new_pool

    def calculate_average_fitness_per_satellite(self, satellite_pool: List[List[float]], num_iterations: int = None) -> List[float]:
        """
        Calculate the average fitness for each satellite across random configurations.

        Args:
            satellite_pool (List[List[float]]): List of satellite configurations.
            num_iterations (int, optional): Number of random configurations to generate (default is len(satellite_pool)).

        Returns:
            List[float]: Average fitness for each satellite.
        """
        if num_iterations is None:
            num_iterations = len(satellite_pool)  # Default to the number of satellites if not specified

        fitness_sums = [0.0 for _ in range(len(satellite_pool))]
        counts = [0 for _ in range(len(satellite_pool))]

        for _ in range(num_iterations):
            config = self.chromosome_controller.make_dead_config()
            selected_indices = [random.randint(0, len(satellite_pool) - 1) for _ in range(self.n_sat)]

            for index in range(self.n_sat):
                self.chromosome_controller.set_all_components(config, index, satellite_pool[selected_indices[index]])

            fitness = self.udp.fitness(config)[0]

            for index in selected_indices:
                fitness_sums[index] += fitness
                counts[index] += 1

        average_fitnesses = [fitness_sums[i] / counts[i] if counts[i] > 0 else 0 for i in range(len(satellite_pool))]
        return average_fitnesses

    def average_fitness_refinement(self, satellite_pool: List[List[float]], top_n: int) -> List[List[float]]:
        """
        Perform one iteration of refining the satellite pool based on average fitness.
        Expands the pool, calculates fitness, and keeps only the top N satellites.

        Args:
            satellite_pool (List[List[float]]): List of satellite configurations to refine.
            top_n (int): Number of top satellites to retain based on average fitness.

        Returns:
            List[List[float]]: The refined satellite pool with top N satellites.
        """
        # Step 1: Expand the satellite pool with variations
        expanded_pool = self.create_variation(satellite_pool)

        # Step 2: Calculate average fitness for the expanded pool
        average_fitnesses = self.calculate_average_fitness_per_satellite(expanded_pool)

        # Step 3: Pair satellites with their corresponding fitness and sort by fitness (lower is better)
        satellites_with_fitness = list(zip(expanded_pool, average_fitnesses))
        sorted_satellites = sorted(satellites_with_fitness, key=lambda x: x[1])

        # Step 4: Select the top N satellites based on fitness
        refined_satellites = [satellite for satellite, fitness in sorted_satellites[:top_n]]

        return refined_satellites

    def calculate_average_fill_factors(self, satellite_pool: List[List[float]], num_iterations: int = None) -> List[float]:
        """
        Calculate the average max fill factors for each satellite across random configurations.

        Args:
            satellite_pool (List[List[float]]): List of satellite configurations.
            num_iterations (int, optional): Number of random configurations to generate (default is len(satellite_pool)).

        Returns:
            List[float]: Average max fill factors for each satellite.
        """
        if num_iterations is None:
            num_iterations = len(satellite_pool)

        fill_sums = [0.0 for _ in range(len(satellite_pool))]
        counts = [0 for _ in range(len(satellite_pool))]

        for _ in range(num_iterations):
            config = self.chromosome_controller.make_dead_config()
            selected_indices = [random.randint(0, len(satellite_pool) - 1) for _ in range(self.n_sat)]

            for index in range(self.n_sat):
                self.chromosome_controller.set_all_components(config, index, satellite_pool[selected_indices[index]])

            # Get the max fill factor for the current configuration
            max_fill_factor = min(self.udp.get_fill_factors(config))
            # print(self.udp.satellite_remaining(config))

            for index in selected_indices:
                fill_sums[index] += max_fill_factor
                counts[index] += 1

        average_fill_factors = [fill_sums[i] / counts[i] if counts[i] > 0 else float('inf') for i in range(len(satellite_pool))]

        return average_fill_factors

    def refine_worst_fill_factor(self, satellite_pool: List[List[float]], n_iterations: int) -> List[List[float]]:
        """
        Refine the satellite pool based on the worst fill factor over multiple iterations.

        Args:
            satellite_pool (List[List[float]]): List of satellite configurations.
            n_iterations (int): Number of iterations for refinement.

        Returns:
            List[List[float]]: Refined satellite pool.
        """
        refined_satellites = satellite_pool
        start_time = time.time()

        for iteration in range(n_iterations):
            # Step 1: Expand the satellite pool with variations
            expanded_pool = self.create_variation(refined_satellites)
            # expanded_pool = self.create_variation(expanded_pool)

            # Step 2: Calculate average fill factors for the expanded pool
            average_factors_for_expanded = self.calculate_average_fill_factors(expanded_pool)

            # Step 3: Sort satellites based on their worst average fill factors (lower is better)
            satellites_with_factors = list(zip(expanded_pool, average_factors_for_expanded))
            sorted_satellites = sorted(satellites_with_factors, key=lambda x: x[1], reverse=True)

            # Step 4: Select the top satellites based on worst fill factors
            refined_satellites = [satellite for satellite, factor in sorted_satellites[:len(refined_satellites)]]

            # Find the minimum factor for this iteration
            min_factor = min(average_factors_for_expanded)

            # Print progress bar with time and minimum fill factor
            print_progress_bar_with_time(
                iteration + 1, 
                n_iterations, 
                time_started=start_time,  
                suffix=f"Min Factor: {min_factor:.4f}", 
            )

        return refined_satellites

class SatellitePoolManager:
    def __init__(self, udp, filename_base: str, quiet: bool):
        """
        Initialize the SatellitePoolManager with necessary parameters.

        Args:
            udp: The UDP instance containing satellite information.
            filename_base (str): The base name for the JSON file.
            quiet (bool): Whether or not to suppress output.
        """
        self.udp = udp
        self.n_sat = udp.n_sat
        self.filename_base = filename_base
        self.quiet = quiet
        self.generator = SatellitePoolGenerator(self.udp, self.quiet)
        self.satellite_file_manager = SatelliteFileManager()

    def ensure_satellite_alive_pool(self, target = POOL_SIZE) -> str:
        """
        Ensure an 'alive' satellite pool file exists. If not, generate it and save it.

        Returns:
            List[List[float]]: A list of alive satellite configurations.
        """
        filename = f"{self.filename_base}-alive-pool-{target}"
        if self.satellite_file_manager.satellite_file_exists(filename):
            return filename
        else:
            if not self.quiet:
                print(f"File '{filename}' does not exist. Generating new alive satellite pool.")
            
            # Generate alive satellites using the generator
            alive_satellite_list = self.generator.collect_alive_satellites(target)
            
            # Save the generated satellite pool to a JSON file
            self.satellite_file_manager.save_satellites(filename, alive_satellite_list)

            return filename

    def iterative_fitness_refinement(self, satellite_pool: List[List[float]], top_n: int, iterations: int, save_filename: str) -> List[List[float]]:
        """
        Iteratively refine the satellite pool based on average fitness over multiple iterations.
        Save the best iteration found and revert to the best set if the fitness worsens.

        Args:
            satellite_pool (List[List[float]]): Initial list of satellite configurations.
            top_n (int): Number of top satellites to keep at each step.
            iterations (int): Number of refinement iterations.
            save_filename (str): Filename to save the best satellite pool.

        Returns:
            List[List[float]]: The refined satellite pool after multiple iterations.
        """
        best_avg_fitness = float('inf')  # Initialize the best fitness as infinity (lower is better)
        best_satellite_pool = satellite_pool  # Keep track of the best satellite pool
        start_time = time.time()

        for i in range(1, iterations + 1):
            # Perform one iteration of average fitness refinement
            refined_satellite_pool = self.generator.average_fitness_refinement(satellite_pool, top_n)

            # Calculate the average fitness of the refined pool
            average_fitness = sum(self.generator.calculate_average_fitness_per_satellite(refined_satellite_pool)) / len(refined_satellite_pool)

            # Check if the current iteration is the best found
            if average_fitness < best_avg_fitness:
                best_avg_fitness = average_fitness
                best_satellite_pool = refined_satellite_pool  # Update the best pool
                # Save the best satellite pool
                self.satellite_file_manager.save_satellites(save_filename, best_satellite_pool, quiet=True)
            else:
                satellite_pool = best_satellite_pool  # Revert to the best satellite pool
            print_progress_bar_with_time(
                i + 1, 
                iterations, 
                time_started=start_time,  
                suffix=f"avg fitness: {best_avg_fitness:.4f}", 
            )

        return best_satellite_pool

    def ensure_satellite_average_fitness_pool(self, top_n: int, iterations: int) -> str:
        """
        Ensure an 'average fitness' satellite pool file exists. If not, refine the alive pool and save it.

        Args:
            top_n (int): Number of top satellites to keep after refining.
            iterations (int): Number of refinement iterations.
            target (int): Number of satellites to collect for the pool.

        Returns:
            List[List[float]]: A list of refined satellite configurations.
        """
        filename = f"{self.filename_base}-refined-{top_n}"
        
        # Check if the refined file already exists
        if self.satellite_file_manager.satellite_file_exists(filename):
            return filename
        else:
            if not self.quiet:
                print(f"Refined file '{filename}' does not exist. Refining the alive satellite pool.")

            # Step 1: Ensure that the alive pool exists
            alive_pool_file = self.ensure_satellite_alive_pool()
            alive_pool, _ = self.satellite_file_manager.load_satellites(alive_pool_file)

            # Step 2: Perform the average fitness refinement
            refined_pool = self.iterative_fitness_refinement(
                alive_pool,
                top_n=top_n,
                iterations=iterations,
                save_filename=filename
            )

            return filename

    def ensure_satellite_worst_fill_factor_pool(self, n_iterations: int) -> str:
        """
        Ensure a satellite pool refined by the worst fill factor exists.
        If it doesn't exist, refine the alive pool based on the worst fill factor and save it.

        Args:
            n_iterations (int): Number of iterations for refinement.

        Returns:
            str: The filename of the refined satellite pool based on the worst fill factor.
        """
        filename = f"{self.filename_base}-refined-fill-factor-{500}"

        # Check if the refined file already exists
        if self.satellite_file_manager.satellite_file_exists(filename):
            if not self.quiet:
                print(f"Refined fill factor pool '{filename}' already generated.")
            return filename
        else:
            if not self.quiet:
                print(f"Refined fill factor pool '{filename}' does not exist. Refining the alive satellite pool.")

            # Step 1: Ensure that the alive pool exists
            alive_pool_file = self.ensure_satellite_alive_pool()
            alive_pool, _ = self.satellite_file_manager.load_satellites(alive_pool_file)

            # Step 2: Perform the refinement based on the worst fill factor
            refined_pool = self.generator.refine_worst_fill_factor(
                satellite_pool=alive_pool,
                n_iterations=n_iterations
            )

            # Step 3: Save the refined satellite pool
            self.satellite_file_manager.save_satellites(filename, refined_pool, quiet=self.quiet)

            return filename

    def ensure_all(self, seed_value: int = 42) -> None:
        """
        Ensure that all necessary satellite pools (alive and average fitness pools) are generated.
        This method ensures that both the 'alive' and 'refined' pools are available.

        Args:
            top_n (int): Number of top satellites to keep after refining.
            iterations (int): Number of refinement iterations.
            target (int): Number of satellites to collect for the pool.
            seed_value (int, optional): The seed for random number generation to ensure consistency. Defaults to 42.
        """
        # Set the seed for consistency
        random.seed(seed_value)

        # Ensure that the alive pool exists
        self.ensure_satellite_alive_pool()

        # Set the seed again to ensure consistency for the next operation
        random.seed(seed_value)

        # Ensure that the average fitness pool exists
        self.ensure_satellite_average_fitness_pool(top_n=POOL_SIZE, iterations=10)

        self.ensure_satellite_worst_fill_factor_pool(n_iterations=10)

        if not self.quiet:
            print(f"All necessary satellite pools are ensured for {self.filename_base}.")

    def get_filenames(self, seed_value: int = 42) -> list[str]:
        pools: list[str] = []

        # Set the seed for consistency
        random.seed(seed_value)
        pools.append(self.ensure_satellite_alive_pool())

        random.seed(seed_value)
        pools.append(self.ensure_satellite_average_fitness_pool(top_n=POOL_SIZE, iterations=10))

        random.seed(seed_value)
        pools.append(self.ensure_satellite_worst_fill_factor_pool(n_iterations=10))

        return pools

    def get_grid_pool_name(self, target=POOL_SIZE):
        return f"{self.filename_base}-grid-pool-{target*10}"

    def ensure_satellite_grid_pool(self, target=POOL_SIZE, expansion_factor=TIGHT_BOUND_EXPANSION) -> str:
        """
        Ensure a grid-based satellite pool file exists. If not, generate it and save both the grid positions
        and the corresponding satellite configurations.

        Args:
            target (int): The number of satellites to collect for the pool.

        Returns:
            str: The filename of the grid-based satellite pool.
        """
        filename = self.get_grid_pool_name(target=target)
        print(filename)
        
        # Check if the grid-based file already exists and is valid
        if self.satellite_file_manager.satellite_file_exists(filename):
            grid_data, _ = self.satellite_file_manager.load_satellites(filename)
            if "grid_positions" in grid_data and "satellite_pool" in grid_data:
                if not self.quiet:
                    print(f"Grid pool '{filename}' already exists and is valid.")
                return filename
            else:
                if not self.quiet:
                    print(f"Grid pool '{filename}' is incomplete or corrupted. Regenerating.")
        
        # Generate a new grid-based pool
        if not self.quiet:
            print(f"Grid pool '{filename}' does not exist. Generating a new grid-based pool.")

        # Step 1: Load the alive pool from the earlier step
        alive_pool_file = self.ensure_satellite_alive_pool(target)
        alive_pool, _ = self.satellite_file_manager.load_satellites(alive_pool_file)

        # Step 2: Set expansion bounds on the udp
        self.udp.set_custom_bounds_switch(True)
        set_expansion_bounds(self.udp, *zip(*alive_pool), expansion_factor=expansion_factor)
        poolGen = SatellitePoolGenerator(self.udp, self.quiet)
        unique_pool = poolGen.collect_unique_alive_satellites(target*10)
        self.udp.set_custom_bounds_switch(False)

        # Step 3: Generate the grid representation for the satellite pool
        grid_positions = [
            self.udp.get_grid_positions_at_times_single_satellite(sat) for sat in unique_pool
        ]

        # Step 4: Save the grid positions and the corresponding satellite configurations
        grid_data = {
            "grid_positions": grid_positions,
            "satellite_pool": unique_pool
        }
        self.satellite_file_manager.save_satellites(filename, grid_data, quiet=self.quiet)

        if not self.quiet:
            print(f"Grid-based satellite pool saved to '{filename}'.")

        return filename






@jit(nopython=True)
def GridBasedChromosomeFactory_encode_differences_jit(dx: int, dy: int, dz: int) -> Tuple[int, int, int]:
    # Create a fixed-size array for output
    output = [0, 0, 0]
    planes = [(dx, dy), (dx, dz), (dy, dz)]
    for plane, (a, b) in enumerate(planes):
        if a < 0 or (a == 0 and b < 0):
            a, b = -a, -b
        encoded_value = (
            (plane << 9) |
            ((b < 0) << 8) |
            ((abs(b) & 0b1111) << 4) |
            (a & 0b1111)
        )
        output[plane] = encoded_value  # Assign to the pre-allocated array
    return output[0], output[1], output[2]

@jit(nopython=True)
def GridBasedChromosomeFactory_calculate_encoded_differences_jit(sat1_grid: np.ndarray, sat2_grid: np.ndarray, num_time_steps: int) -> List[Tuple[int, int, int]]:
    encoded_differences = []
    for time_step in range(num_time_steps):
        x1, y1, z1 = sat1_grid[time_step]
        x2, y2, z2 = sat2_grid[time_step]
        dx = x1 - x2
        dy = y1 - y2
        dz = z1 - z2

        encoded_xy, encoded_xz, encoded_yz = GridBasedChromosomeFactory_encode_differences_jit(dx, dy, dz)

        encoded_differences.append((encoded_xy, encoded_xz, encoded_yz))

    return encoded_differences


class GridBasedChromosomeFactory:
    def __init__(self, udp, grid: List[List[List[int]]], n_sat: int):
        self.udp = udp
        self.grid = grid
        self.n_sat = n_sat

        self.num_time_steps = 3

        self.selected_indices = []

        self.current_differences = [Counter() for _ in range(self.num_time_steps)]
        self.current_sums_per_time = [0 for _ in range(self.num_time_steps)]

    def __call__(self):
        return self.generate()

    def generate(self):
        print(len(self.grid))
        self._select_first_satellite()
        self._fill_remaining_satellites()
        best_found = min(self.current_sums_per_time) - 1
        while True:
            index = self.smallest()
            self.remove_index_and_its_differences(index)
            self._fill_remaining_satellites()
            if best_found < min(self.current_sums_per_time):
                best_found = min(self.current_sums_per_time)
                print(min(self.current_sums_per_time))
            else:
                break
        while True:
            index = self.smallest()
            self.remove_index_and_its_differences(index)
            self._fill_remaining_satellites()
            index = self.smallest()
            self.remove_index_and_its_differences(index)
            self._fill_remaining_satellites()
            if best_found < min(self.current_sums_per_time):
                best_found = min(self.current_sums_per_time)
                print(min(self.current_sums_per_time))
            else:
                break
        while True:
            index = self.smallest()
            self.remove_index_and_its_differences(index)
            self._fill_remaining_satellites()
            index = self.smallest()
            self.remove_index_and_its_differences(index)
            self._fill_remaining_satellites()
            index = self.smallest()
            self.remove_index_and_its_differences(index)
            self._fill_remaining_satellites()
            if best_found < min(self.current_sums_per_time):
                best_found = min(self.current_sums_per_time)
                print(min(self.current_sums_per_time))
            else:
                break
        while True:
            index = self.smallest()
            self.remove_index_and_its_differences(index)
            self._fill_remaining_satellites()
            index = self.smallest()
            self.remove_index_and_its_differences(index)
            self._fill_remaining_satellites()
            index = self.smallest()
            self.remove_index_and_its_differences(index)
            self._fill_remaining_satellites()
            index = self.smallest()
            self.remove_index_and_its_differences(index)
            self._fill_remaining_satellites()
            if best_found < min(self.current_sums_per_time):
                best_found = min(self.current_sums_per_time)
                print(min(self.current_sums_per_time))
            else:
                break
        while True:
            index = self.smallest()
            self.remove_index_and_its_differences(index)
            self._fill_remaining_satellites()
            index = self.smallest()
            self.remove_index_and_its_differences(index)
            self._fill_remaining_satellites()
            index = self.smallest()
            self.remove_index_and_its_differences(index)
            self._fill_remaining_satellites()
            index = self.smallest()
            self.remove_index_and_its_differences(index)
            self._fill_remaining_satellites()
            index = self.smallest()
            self.remove_index_and_its_differences(index)
            self._fill_remaining_satellites()
            if best_found < min(self.current_sums_per_time):
                best_found = min(self.current_sums_per_time)
                print(min(self.current_sums_per_time))
            else:
                break

        return self.selected_indices

    def _select_first_satellite(self):
        first_index = random.randint(0, len(self.grid) - 1)
        self.selected_indices.append(first_index)
        self._add_candidate(random.randint(0, len(self.grid) - 1))
        self._add_candidate(random.randint(0, len(self.grid) - 1))
        self._add_candidate(random.randint(0, len(self.grid) - 1))
        self._add_candidate(random.randint(0, len(self.grid) - 1))
        self._add_candidate(random.randint(0, len(self.grid) - 1))

    def _grids_to_search(self):
        # So I can swap out this logic easily
        # return random.sample(range(len(self.grid)), 500)
        return range(len(self.grid))

    def _find_best_candidate(self):
        max_min_total_sum = -1
        max_new_differences = -1
        best_candidate = None

        for candidate_index in self._grids_to_search():
            if candidate_index in self.selected_indices:
                continue
            if self._is_any_overlapping(candidate_index):
                continue

            new_differences, added_sums_per_time, min_total_sum, number_of_new_differences = self._evaluate_candidate(candidate_index)

            # Prioritize candidates with more new differences first
            if number_of_new_differences > max_new_differences or (number_of_new_differences == max_new_differences and min_total_sum > max_min_total_sum):
                max_new_differences = number_of_new_differences
                max_min_total_sum = min_total_sum
                best_candidate = {
                    'index': candidate_index,
                    'new_differences': new_differences,
                    'added_sums_per_time': added_sums_per_time
                }

        return best_candidate

    def _evaluate_candidate(self, candidate_index):
        new_differences = [Counter() for _ in range(self.num_time_steps)]
        added_sums_per_time = [0 for _ in range(self.num_time_steps)]
        number_of_new_differences = 0  # To track newly unique differences

        for selected_index in self.selected_indices:
            diffs_with_time = GridBasedChromosomeFactory_calculate_encoded_differences_jit(
                np.array(self.grid[selected_index]),
                np.array(self.grid[candidate_index]),
                self.num_time_steps
            )
            for t, diffs in enumerate(diffs_with_time):
                new_differences[t][diffs[t]] += 1
                if diffs[t] not in self.current_differences[t]:
                    number_of_new_differences += 1

        total_sums_per_time = []
        for t in range(self.num_time_steps):
            added_sum_t = sum(new_differences[t].values())
            added_sums_per_time[t] = added_sum_t
            total_sum_t = added_sum_t
            total_sums_per_time.append(total_sum_t)

        min_total_sum = max(total_sums_per_time)

        # Return the number of new differences as an additional metric
        return new_differences, added_sums_per_time, min_total_sum, number_of_new_differences

    def _add_candidate(self, candidate_index):
        self.selected_indices.append(candidate_index)

        candidate_grid = np.array(self.grid[candidate_index])

        for selected_index in self.selected_indices[:-1]:  # Exclude the newly added candidate
            diffs_with_time = GridBasedChromosomeFactory_calculate_encoded_differences_jit(
                np.array(self.grid[selected_index]),
                candidate_grid,
                self.num_time_steps
                )
            for time_step, diffs in enumerate(diffs_with_time):
                # Update current differences using the calculated diffs
                for t in range(self.num_time_steps):
                    self.current_differences[time_step][diffs[t]] += 1
                    if self.current_differences[time_step][diffs[t]] == 1:
                        self.current_sums_per_time[t] += 1

    def _add_best_candidate(self):
        best_candidate = self._find_best_candidate()
        if best_candidate is not None:
            self._add_candidate(best_candidate['index'])
    
    def _fill_remaining_satellites(self):
        while len(self.selected_indices) < self.n_sat:
            self._add_best_candidate()

    def _is_any_overlapping(self, candidate_index):
        candidate_grid = self.grid[candidate_index]
        for sat_index in self.selected_indices:
            selected_grid = self.grid[sat_index]
            matching = any(
                sum(
                    candidate_grid[num_time_step][axis] == selected_grid[num_time_step][axis]
                    for axis in range(self.num_time_steps)
                ) > 1
                for num_time_step in range(self.num_time_steps)
            )
        return matching

    def _calculate_encoded_differences(self, sat1_grid: List[List[int]], sat2_grid: List[List[int]]) -> List[Tuple[int, int, int]]:
        return GridBasedChromosomeFactory_calculate_encoded_differences_jit(sat1_grid, sat2_grid, self.num_time_steps)

        # encoded_differences = []
        # for time_step in range(self.num_time_steps):
        #     x1, y1, z1 = sat1_grid[time_step]
        #     x2, y2, z2 = sat2_grid[time_step]
        #     dx = x1 - x2
        #     dy = y1 - y2
        #     dz = z1 - z2

        #     encoded_xy, encoded_xz, encoded_yz = _encode_differences_jit(dx, dy, dz)

        #     encoded_differences.append((encoded_xy, encoded_xz, encoded_yz))
        
        # return encoded_differences

    def calculate_unique_contributions(self, selected_index):
        unique_contribution_count = 0
        candidate_grid = np.array(self.grid[selected_index])

        # Iterate through all previously selected satellites
        for other_index in self.selected_indices:
            if other_index == selected_index:
                continue

            other_grid = np.array(self.grid[other_index])
            diffs_with_time = GridBasedChromosomeFactory_calculate_encoded_differences_jit(
                candidate_grid,
                other_grid,
                self.num_time_steps
            )

            # Check if the differences are unique because of this candidate
            for time_step, diffs in enumerate(diffs_with_time):
                for t in range(self.num_time_steps):
                    if self.current_differences[time_step][diffs[t]] == 1:
                        unique_contribution_count += 1

        return unique_contribution_count

    def smallest(self):
        least_found = 4000
        index_with_smallest_unique_contributions = -1
        for index in self.selected_indices:
            contributions = self.calculate_unique_contributions(index)
            if contributions < least_found:
                least_found = contributions
                index_with_smallest_unique_contributions = index
        return index_with_smallest_unique_contributions

    def remove_index_and_its_differences(self, index_to_remove):
        if index_to_remove not in self.selected_indices:
            return  # Index not in the selected list

        self.selected_indices.remove(index_to_remove)
        candidate_grid = np.array(self.grid[index_to_remove])

        # Update current differences by removing the contribution of the index
        for other_index in self.selected_indices:
            other_grid = np.array(self.grid[other_index])
            diffs_with_time = GridBasedChromosomeFactory_calculate_encoded_differences_jit(
                candidate_grid,
                other_grid,
                self.num_time_steps
            )

            for time_step, diffs in enumerate(diffs_with_time):
                for t in range(self.num_time_steps):
                    if diffs[t] in self.current_differences[time_step]:
                        self.current_differences[time_step][diffs[t]] -= 1
                        if self.current_differences[time_step][diffs[t]] == 0:
                            del self.current_differences[time_step][diffs[t]]
                            self.current_sums_per_time[time_step] -= 1

















# class GridSatellitePool:
#     def __init__(self, udp):
#         self.udp = udp
#         self.n_sat = self.udp.n_sat
#         self.pool = []
#         self.grid_positions = []
#         self.time_steps = 3
#         self.max_n = 0  # This will store the maximum achievable n

#     def add_to_pool(self, new_pool):
#         self.pool.extend(new_pool)
#         self.add_pool_to_grid_positions(new_pool)
#         self.max_n = 0

#     def add_pool_to_grid_positions(self, new_pool):
#         new_grid_positions = [self.udp.get_grid_positions_at_times_single_satellite(sat) for sat in new_pool]
#         self.grid_positions.extend(new_grid_positions)

#     def create_grid_based_chromosome(self, n):
#         """
#         Creates a GridBasedChromosome by randomly selecting satellites from the pool,
#         aiming to achieve the theoretical maximum fitness for n satellites.
#         """
#         # Calculate the theoretical max fitness
#         theoretical_max_fitness = (n * (n - 1) // 2) * 3

#         repeats = 0
#         while True:
#             # Create a new GridBasedChromosome
#             chromosome = GridBasedChromosome(self.udp)

#             # Randomly select n indices from the pool
#             if len(self.pool) < n:
#                 raise ValueError("Not enough satellites in the pool to select the desired number.")

#             selected_indices = random.sample(range(len(self.pool)), n)

#             # Add the selected satellites to the chromosome
#             for index in selected_indices:
#                 selected_satellite = self.pool[index]
#                 selected_grid_position = self.grid_positions[index]
#                 chromosome.add_selected(selected_satellite, selected_grid_position)

#             # Get the fitness achieved
#             fitness_achieved = chromosome.get_fitness()

#             # Check if the achieved fitness is equal to the theoretical max fitness
#             if fitness_achieved == theoretical_max_fitness:
#                 break

#             # Increment the repeat counter
#             repeats += 1

#         # Print the results
#         print(f"Fitness achieved: {fitness_achieved}, Repeats: {repeats}")

#         # Return the created chromosome
#         return chromosome

#     def _calculate_max_n(self):
#         """Calculates the maximum n for which the theoretical max fitness can be achieved within 1 second."""
#         self.max_n = 0
#         n = 0
#         while True:
#             # Calculate the theoretical max fitness for n
#             theoretical_max_fitness = (n * (n - 1) // 2) * 3

#             start_time = time.time()
#             found_solution = False
#             while time.time() - start_time < 1:  # Try for up to 1 second
#                 # Create a temporary GridBasedChromosome
#                 chromosome = GridBasedChromosome(self.udp)

#                 # Randomly select n indices from the pool if there are enough satellites
#                 if len(self.pool) < n:
#                     break

#                 selected_indices = random.sample(range(len(self.pool)), n)

#                 # Add the selected satellites to the chromosome
#                 for index in selected_indices:
#                     selected_satellite = self.pool[index]
#                     selected_grid_position = self.grid_positions[index]
#                     chromosome.add_selected(selected_satellite, selected_grid_position)

#                 # Check if we achieved the theoretical max fitness
#                 fitness_achieved = chromosome.get_fitness()
#                 if fitness_achieved == theoretical_max_fitness:
#                     found_solution = True
#                     break

#             # If we found a solution, increase n and continue, otherwise stop
#             if found_solution:
#                 self.max_n = n
#                 n += 1
#             else:
#                 break

#     def _generate_starting_point(self):
#         """Generates a starting GridBasedChromosome with the maximum achievable n."""
#         n = self.max_n
#         if n == 0:
#             self._calculate_max_n()
#         if n == 0:
#             return GridBasedChromosome(self)

#         # Try to generate a GridBasedChromosome with n satellites
#         while True:
#             chromosome = GridBasedChromosome(self.udp)
#             selected_indices = random.sample(range(len(self.pool)), n)

#             # Add the selected satellites to the chromosome
#             for index in selected_indices:
#                 selected_satellite = self.pool[index]
#                 selected_grid_position = self.grid_positions[index]
#                 chromosome.add_selected(selected_satellite, selected_grid_position)

#             # Check if the fitness achieved matches the theoretical max fitness
#             theoretical_max_fitness = (n * (n - 1) // 2) * 3
#             if chromosome.get_fitness() == theoretical_max_fitness:
#                 return chromosome

#     def generate_chromosome(self):
#         """Generates a full GridBasedChromosome by expanding a starting point."""
#         # Step 1: Generate the starting chromosome
#         chromosome = self._generate_starting_point()
#         print(f"Starting fitness = {chromosome.get_fitness()}")

#         # Step 2: Expand the chromosome until it is full
#         while not chromosome.is_full:
#             # Randomly pick a subset of 100 from the pool
#             subset_indices = random.sample(range(len(self.pool)), min(1000, len(self.pool)))

#             # Find the best candidate from the subset based on evaluate_candidate_fitness
#             best_fitness = float('-inf')
#             best_index = None
#             for index in subset_indices: #range(len(self.pool)):
#                 candidate_grid_position = self.grid_positions[index]
#                 fitness = chromosome.evaluate_candidate_fitness(candidate_grid_position)
#                 if fitness > best_fitness:
#                     best_fitness = fitness
#                     best_index = index

#             # Add the best candidate to the chromosome
#             if best_index is not None:
#                 selected_grid_position = self.grid_positions[best_index]
#                 chromosome.add_selected(best_index, selected_grid_position)
#                 print(f"New fitness = {chromosome.get_fitness()}")

#         return chromosome

#     def generate_chromosome(self):
#         """Generates a full GridBasedChromosome by expanding a starting point, using a layered reading approach."""
#         # Step 1: Generate the starting chromosome
#         chromosome = self._generate_starting_point()
#         print(f"Starting fitness = {chromosome.get_fitness()}")


#         # Step 2: Prepare a list of candidates
#         candidates = [[i, []] for i in range(len(self.pool))]  # [index, [readings]]

#         # Step 3: Expand the chromosome until it is full
#         while not chromosome.is_full:
#             best_fitness = float('-inf')
#             best_candidate = None
#             num_satellites_added = len(chromosome.selected_indices)

#             # Loop through the candidates
#             for candidate in candidates:
#                 index, readings = candidate
#                 # Check the most recent fitness reading layer by layer
#                 possible_max_fitness = None
#                 for depth, reading in enumerate(readings):
#                     # Calculate the maximum possible fitness increase
#                     max_possible_increase = (num_satellites_added - depth) * 3
#                     if reading + max_possible_increase < best_fitness:
#                         # Skip this candidate if the reading at this depth isn't good enough
#                         break
#                     # Update possible_max_fitness if this reading is the highest encountered so far
#                     possible_max_fitness = max(possible_max_fitness or float('-inf'), reading)
#                 else:
#                     # If all layers were checked, use the possible_max_fitness found in readings
#                     if possible_max_fitness is not None and possible_max_fitness > best_fitness:
#                         best_fitness = possible_max_fitness
#                         best_candidate = index

#                 # Evaluate the candidate's fitness if we didn't find a suitable candidate through readings
#                 if possible_max_fitness is None or possible_max_fitness < best_fitness:
#                     candidate_grid_position = self.grid_positions[index]
#                     fitness = chromosome.evaluate_candidate_fitness(candidate_grid_position)

#                     # Update readings: move the most recent fitness to the front
#                     readings.insert(0, fitness)
#                     if len(readings) > num_satellites_added + 1:
#                         readings.pop()  # Keep only as many layers as there are satellites added

#                     # If this candidate is the best found, update the best variables
#                     if fitness > best_fitness:
#                         best_fitness = fitness
#                         best_candidate = index

#             # If a valid best candidate was found, add it to the chromosome
#             if best_candidate is not None:
#                 selected_satellite = self.pool[best_candidate]
#                 selected_grid_position = self.grid_positions[best_candidate]
#                 chromosome.add_selected(selected_satellite, selected_grid_position)
#             print(f"New fitness = {chromosome.get_fitness()}")


#         return chromosome


# from collections import Counter
# from functools import lru_cache


# class GridBasedChromosome:
#     def __init__(self, udp=None):
#         self.udp = udp
#         self.n_sat = udp.n_sat
#         self.selected_indices = []
#         self.grid_positions = []
#         self.difference_counters = [Counter(), Counter(), Counter()]
#         self.difference_lengths = [0, 0, 0]
#         self.memo_cache = {}

#     def add_selected(self, new_index, new_grid_position):
#         new_differences = self.calculate_differences(new_grid_position, self.grid_positions)
#         self.add_differences(new_differences)
#         self.selected_indices.append(new_index)
#         self.grid_positions.append(new_grid_position)

#     def remove_index(self, index):
#         """Removes the satellite at the specified index and returns its details."""
#         # Remove the grid position and selected index
#         removed_grid_pos = self.grid_positions.pop(index)
#         removed_selected_index = self.selected_indices.pop(index)

#         # Calculate and remove the differences for the removed satellite
#         removed_differences = self.calculate_differences(removed_grid_pos)
#         self.remove_differences(removed_differences)

#         # Return the details of the removed satellite
#         return removed_selected_index, removed_grid_pos, removed_differences

#     def extract_index(self, index):
#         """Extracts the selected index and grid position without modifying the state."""
#         selected_index = self.selected_indices[index]
#         grid_position = self.grid_positions[index]
#         return selected_index, grid_position

#     def add_differences(self, new_differences):
#         for t in (0, 1, 2):
#             t_new_differences = new_differences[t]
#             for diff in t_new_differences:
#                 self.difference_counters[t][diff] += 1
#                 # Update the count of unique differences
#                 if self.difference_counters[t][diff] == 1:
#                     self.difference_lengths[t] += 1

#     def remove_differences(self, old_differences):
#         for t in (0, 1, 2):
#             t_old_differences = old_differences[t]
#             for diff in t_old_differences:
#                 if self.difference_counters[t][diff] > 0:
#                     self.difference_counters[t][diff] -= 1
#                     # Update the count of unique differences
#                     if self.difference_counters[t][diff] == 0:
#                         self.difference_lengths[t] -= 1

#     def calculate_differences(self, new_grid_position, grids):
#         differences = ([], [], [])
#         for grid_pos in grids:
#             for plane_index in (0, 1, 2):
#                 planeA = grid_pos[plane_index]
#                 planeB = new_grid_position[plane_index]
#                 xy, xz, yz = self.calculate_differences_planes(planeA, planeB)
#                 differences[0].append(xy)
#                 differences[1].append(xz)
#                 differences[2].append(yz)
#         return differences

#     def calculate_differences_planes(self, grid_pos1, grid_pos2):
#         """Calculate differences for xy, xz, and zy planes between two grid positions."""
#         key = (tuple(grid_pos1), tuple(grid_pos2))
#         if key in self.memo_cache:
#             return self.memo_cache[key]

#         x1, y1, z1 = grid_pos1
#         x2, y2, z2 = grid_pos2
#         # Example calculations for each plane
#         dx = x1 - x2
#         dy = y1 - y2
#         dz = z1 - z2
#         xy = self.encode_plane_difference(0, dx, dy)
#         xz = self.encode_plane_difference(1, dx, dz)
#         yz = self.encode_plane_difference(2, dy, dz)

#         self.memo_cache[key] = (xy, xz, yz)
#         return xy, xz, yz

#     def encode_plane_difference(self, plane, change1, change2):
#         """Encodes the plane, change1, and change2 into a single integer."""
#         if (change1 < 0 or (change1 == 0 and change2 < 0)):
#             change1 = -change1
#             change2 = -change2

#         # Encode the plane (2 bits)
#         plane_bits = plane & 0b11

#         # Encode change1 (4 bits)
#         change1_bits = abs(change1) & 0b1111

#         # Encode change2 (4 bits for magnitude, 1 bit for sign)
#         change2_bits = abs(change2) & 0b1111
#         sign_bit = 1 if change2 < 0 else 0

#         # Combine all the encoded bits: [plane_bits(2)] [change1_bits(4)] [change2_bits(4)] [sign_bit(1)]
#         encoded_value = (plane_bits << 9) | (change1_bits << 5) | (change2_bits << 1) | sign_bit
#         return encoded_value

#     def get_fitness(self):
#         return min(self.difference_lengths)

#     def evaluate_candidate_fitness(self, candidate_grid_position):
#         """Evaluates the fitness if the candidate grid position were to be added."""
#         # Step 1: Calculate differences for the candidate position
#         candidate_differences = self.calculate_differences(candidate_grid_position, self.grid_positions)

#         # Step 2: Determine how many unique differences would be added
#         temp_lengths = self.difference_lengths[:]
#         for t in (0, 1, 2):
#             t_candidate_differences = candidate_differences[t]
#             already_accounted = set()  # Track differences that have been processed
#             for diff in t_candidate_differences:
#                 if diff not in already_accounted:
#                     # Check if this difference is currently in the difference_counters
#                     if self.difference_counters[t][diff] == 0:
#                         # It would be a new unique difference
#                         temp_lengths[t] += 1
#                     # Mark this difference as accounted for
#                     already_accounted.add(diff)

#         # Step 3: Return the minimum of the updated difference lengths as the fitness
#         return min(temp_lengths) - min(self.difference_lengths)

#     def evaluate_satellite_contribution(self):
#         """Evaluates how much each satellite contributes to the fitness."""
#         contributions = []
#         original_fitness = self.get_fitness()

#         # Loop through each satellite
#         for i in range(len(self.selected_indices)):
#             # Step 1: Temporarily remove the satellite using remove_index
#             removed_selected_index, removed_grid_pos, removed_differences = self.remove_index(i)

#             # Step 2: Calculate the fitness without this satellite
#             fitness_without_satellite = self.get_fitness()

#             # Step 3: Add the satellite back
#             self.add_differences(removed_differences)
#             self.selected_indices.insert(i, removed_selected_index)
#             self.grid_positions.insert(i, removed_grid_pos)

#             # Step 4: Calculate the contribution as the difference in fitness
#             contribution = original_fitness - fitness_without_satellite
#             contributions.append(contribution)

#         return contributions

#     @property
#     def is_full(self):
#         """Check if the chromosome is full."""
#         return len(self.selected_indices) == self.n_sat



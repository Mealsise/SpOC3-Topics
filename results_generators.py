import pygmo as pg # type: ignore
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from datetime import datetime
import random
import time
from tabulate import tabulate # type: ignore
import numpy as np
from scipy.stats import ks_2samp, ttest_ind, levene # type: ignore


from file_handling import ensure_directory_exists, load_results_with_metadata, file_exists, save_results_with_metadata, save_plot_metadata, load_plot_metadata, is_generated_from
from chromosome_operations import SatellitePoolManager, ChromosomeFactory, DistanceMetric, ChromosomeController, SatellitePoolGenerator

NUMBER_OF_CHROMOSOMES = 1000
POOL_SIZES = 500
TIGHT_BOUND_EXPANSION = 0.1
PLOT_SIZES = 20, 20



def generate_chromosomes_and_evaluate_fitness(factory: ChromosomeFactory, num_chromosomes: int = 100) -> list[float]:
    """
    Generates a set number of chromosomes using the ChromosomeFactory and evaluates their fitness.
    
    Args:
        factory (ChromosomeFactory): The factory used to generate satellite configurations.
        num_chromosomes (int): Number of chromosomes to generate and evaluate. Default is 100.
    
    Returns:
        List[float]: A list of fitness values for each generated chromosome.
    """
    udp = factory.udp
    fitness_list = []
    
    # Generate and evaluate chromosomes
    for _ in range(num_chromosomes):
        # Generate a chromosome (satellite configuration) from the factory
        chromosome = factory()
        
        # Evaluate the fitness of the chromosome using the UDP
        fitness = udp.fitness(chromosome)
        
        # Store the fitness result
        fitness_list.append(fitness[0])
    
    return fitness_list

# Timing function to generate 500 alive satellites using default and tight bounds
def time_alive_generation(udp, target=500, use_tight_bounds=False):
    """
    Times the generation of alive satellites using either default or custom bounds.
    
    Args:
        udp: The UDP instance.
        target (int): Number of alive satellites to generate.
        use_tight_bounds (bool): Whether to use custom tight bounds.
    
    Returns:
        float: The time taken to generate the target number of alive satellites.
    """
    # Switch between custom tight bounds and default bounds
    udp.set_custom_bounds_switch(tight=use_tight_bounds)
    
    # Initialize the SatellitePoolGenerator
    pool_generator = SatellitePoolGenerator(udp, quiet=True)
    
    # Start timing the generation
    start_time = time.time()
    
    # Generate alive satellites
    satellite_pool = pool_generator.collect_alive_satellites(target)
    
    elapsed_time = time.time() - start_time
    return elapsed_time, satellite_pool

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



def figure_1(udp, orbit_name, number_of_chromosomes=NUMBER_OF_CHROMOSOMES):
    # Set directory and filename based on orbit name
    directory = "figures"
    plot_filepath = f"{directory}/figure_1_{orbit_name}.png"
    random.seed(plot_filepath)

    # Ensure directory exists
    ensure_directory_exists(directory)

    # Generate random initialization data with 'from-bounds' method
    print("Generating random initialization data with 'from-bounds' method.")
    pool_manager = SatellitePoolManager(udp, orbit_name, quiet=False)
    pool_manager.ensure_all()

    # Use "from-bounds" method for chromosome generation and fitness evaluation
    satellite_pool = pool_manager.satellite_file_manager.load_satellites(pool_manager.get_filenames()[0], quiet=True)[0]
    random_fitness_values = generate_chromosomes_and_evaluate_fitness(
        ChromosomeFactory(
            udp=udp,
            satellite_pool=satellite_pool,
            pool_name="from-bounds",
            metric_name="none",
            generation_method="from-bounds"
        ),
        num_chromosomes=number_of_chromosomes
    )

    # Plot the histogram without theoretical and achieved max lines
    plt.figure(figsize=(6, 4))
    plt.hist(random_fitness_values, bins=50, color='lightcoral', alpha=0.7, label=f"Random Initialization ({orbit_name})")

    # Add vertical line for baseline (0) only
    plt.axvline(0, color='gray', linestyle='--', label="Baseline (0)")

    # Set plot titles and labels
    plt.title(f"Fitness Distribution for Random Initialization ({orbit_name})")
    plt.xlabel("Fitness")
    plt.ylabel("Frequency")
    plt.legend(loc='upper left')

    # Adjust x-axis to display fewer tick markers
    plt.xticks(rotation=45)  # Rotate ticks for readability
    x_ticks = plt.gca().get_xticks()  # Get current tick locations
    plt.gca().set_xticks(x_ticks[::2])  # Display every second tick

    # Save and display the plot
    plt.tight_layout()
    plt.savefig(plot_filepath, dpi=300)
    plt.close()
    print(f"Plot saved to {plot_filepath}")

    # Display the saved plot image
    img = mpimg.imread(plot_filepath)
    plt.figure(figsize=(12, 10))
    plt.imshow(img)
    plt.axis('off')
    plt.show()

def figure_2(udp, orbit_name, theoretical_max, achieved_max, number_of_chromosomes=NUMBER_OF_CHROMOSOMES):
    # Set directory and filename based on orbit name
    directory = "figures"
    plot_filepath = f"{directory}/figure_2_{orbit_name}.png"
    random.seed(plot_filepath)

    # Ensure directory exists
    ensure_directory_exists(directory)

    # Generate random initialization data with 'from-pool' method
    print("Generating data using 'from-pool' method.")
    pool_manager = SatellitePoolManager(udp, orbit_name, quiet=False)
    pool_manager.ensure_all()

    # Use "from-pool" method for chromosome generation and fitness evaluation
    satellite_pool = pool_manager.satellite_file_manager.load_satellites(pool_manager.get_filenames()[0], quiet=True)[0]
    random_fitness_values = generate_chromosomes_and_evaluate_fitness(
        ChromosomeFactory(
            udp=udp,
            satellite_pool=satellite_pool,
            pool_name="from-pool",
            metric_name="none",
            generation_method="from-pool"
        ),
        num_chromosomes=number_of_chromosomes
    )

    # Plot the histogram
    plt.figure(figsize=(6, 4))
    plt.hist(random_fitness_values, bins=50, color='skyblue', alpha=0.7, label=f"Alive Pool Filtering ({orbit_name})")

    # Add vertical lines for 0, theoretical max, and achieved max
    plt.axvline(0, color='gray', linestyle='--', label="Baseline (0)")
    plt.axvline(theoretical_max, color='green', linestyle='--', label=f"Theoretical Max ({theoretical_max})")
    plt.axvline(achieved_max, color='red', linestyle='--', label=f"Achieved Max ({achieved_max})")

    # Set plot titles and labels
    plt.title(f"Fitness Distribution for Alive Pool Filtering ({orbit_name})")
    plt.xlabel("Fitness")
    plt.ylabel("Frequency")
    plt.legend(loc='upper left')

    # Save and display the plot
    plt.tight_layout()
    plt.savefig(plot_filepath, dpi=300)
    plt.close()
    print(f"Plot saved to {plot_filepath}")

    # Display the saved plot image
    img = mpimg.imread(plot_filepath)
    plt.figure(figsize=(12, 10))
    plt.imshow(img)
    plt.axis('off')
    plt.show()

def figure_3(udp, tight_udp, orbit_name, target=POOL_SIZES, expansion_factor=TIGHT_BOUND_EXPANSION):
    """
    Generates data if needed and plots box plots comparing satellite component distributions for tight vs. full bounds.
    
    Args:
        udp: The udp object used for satellite generation.
        tight_udp: The udp object with tight bounds set.
        orbit_name (str): The orbit name, used for identifying the results directory.
    """
    # Set directory, filename, and random seed for reproducibility
    directory = "figures"
    plot_filepath = f"{directory}/figure_3_{orbit_name}.png"
    results_filepath = f"results/results_{orbit_name}_tight_vs_full_bounds.json"
    random.seed(plot_filepath)

    # Ensure the directories exist
    ensure_directory_exists(directory)
    ensure_directory_exists("results")  # Ensure results folder exists for the JSON file

    # Step 1: Check if the results file already exists
    if not file_exists(results_filepath):
        print("Generating data for tight vs. full bounds comparison.")

        # Load satellite pool (assuming first pool file)
        pool_manager = SatellitePoolManager(udp, orbit_name, quiet=False)
        pool_manager.ensure_all()
        pool_filename = pool_manager.get_filenames()[0]
        satellite_pool = pool_manager.satellite_file_manager.load_satellites(pool_filename, quiet=True)[0]

        # Step 2: Default bounds generation
        random.seed("Default bounds")
        default_time, default_pool = time_alive_generation(udp, target=target, use_tight_bounds=False)
        print(f"Time with default bounds: {default_time:.2f} seconds for {len(default_pool)} satellites")

        # Step 3: Tight bounds generation
        random.seed("Tight bounds")
        set_expansion_bounds(tight_udp, *zip(*satellite_pool), expansion_factor=expansion_factor)

        tight_time, tight_pool = time_alive_generation(tight_udp, target=target, use_tight_bounds=True)
        print(f"Time with tight bounds: {tight_time:.2f} seconds for {len(tight_pool)} satellites")

        # Save results
        metadata = {
            "orbit": orbit_name,
            "description": "Tight vs. full bounds generation speed comparison.",
            "target": target,
            "expansion_factor": expansion_factor,
            "default_time": default_time,
            "tight_time": tight_time,
            "datetime": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        save_results_with_metadata(results_filepath, metadata, {"default_pool": default_pool, "tight_pool": tight_pool})
        print(f"Results saved to {results_filepath}.")

    # Load the results
    pools_data, _ = load_results_with_metadata(results_filepath)

    # Decompose pools into components for box plots
    default_pool = pools_data['default_pool']
    tight_pool = pools_data['tight_pool']
    x_full, y_full, z_full, xv_full, yv_full, zv_full = zip(*default_pool)
    x_tight, y_tight, z_tight, xv_tight, yv_tight, zv_tight = zip(*tight_pool)

    # Group data for each component to align plots in two rows
    components = {
        'X': (x_full, x_tight),
        'Y': (y_full, y_tight),
        'Z': (z_full, z_tight),
        'XV': (xv_full, xv_tight),
        'YV': (yv_full, yv_tight),
        'ZV': (zv_full, zv_tight)
    }

    # Plot components in a 2x3 grid layout with adjusted spacing
    fig, axs = plt.subplots(2, 3, figsize=(10, 8), gridspec_kw={'width_ratios': [1, 1, 1]})
    fig.suptitle("Comparison of Satellite Components for Full vs. Tight Bounds", fontsize=14)

    # Populate each subplot
    for idx, (component, (full_vals, tight_vals)) in enumerate(components.items()):
        row, col = divmod(idx, 3)
        ax = axs[row, col]
        ax.boxplot([full_vals, tight_vals], widths=0.6, labels=['Full Bounds', 'Tight Bounds'])
        ax.set_title(f'{component} Component', fontsize=12)
        ax.set_ylabel(f'{component} Value')

    # Tight layout adjustments to maximize box plot width
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.subplots_adjust(wspace=0.4, hspace=0.4)
    plt.savefig(plot_filepath, dpi=300)
    plt.close()
    print(f"Box plot saved to {plot_filepath}.")

    # Display the saved plot image
    img = mpimg.imread(plot_filepath)
    plt.figure(figsize=(10, 6))
    plt.imshow(img)
    plt.axis('off')
    plt.show()

def figure_4(udp, orbit_name, num_chromosomes=NUMBER_OF_CHROMOSOMES):
    """
    Generates and plots violin plots comparing spatial distributions for different generation methods.
    
    Args:
        udp: The udp object used for satellite generation.
        orbit_name (str): The orbit name for identifying results.
        num_chromosomes (int): Number of chromosomes to generate per method.
    """
    # Set directory and plot filepath
    directory = "figures"
    plot_filepath = f"{directory}/figure_4_{orbit_name}.png"
    random.seed(plot_filepath)

    # Ensure directory exists
    ensure_directory_exists(directory)

    # Initialize pool manager and load satellite pool
    pool_manager = SatellitePoolManager(udp, orbit_name, quiet=False)
    pool_manager.ensure_all()
    satellite_pool = pool_manager.satellite_file_manager.load_satellites(pool_manager.get_filenames()[0], quiet=True)[0]

    # Generate chromosomes for each method
    factory_from_pool = ChromosomeFactory(
        udp=udp,
        satellite_pool=satellite_pool,
        pool_name="generated_pool",
        generation_method="from-pool"
    )
    factory_from_pool_differentiated = ChromosomeFactory(
        udp=udp,
        satellite_pool=satellite_pool,
        pool_name="generated_pool",
        metric_name="manhattan",
        generation_method="from-pool-differentiated"
    )

    # Collect satellite positions for each method
    from_pool_chromosomes = [factory_from_pool() for _ in range(num_chromosomes)]
    differentiated_chromosomes = [factory_from_pool_differentiated() for _ in range(num_chromosomes)]

    # Extract coordinates: [x, y, z, xv, yv, zv] for each satellite
    from_pool_positions = {
        "x": [], "y": [], "z": [], "xv": [], "yv": [], "zv": []
    }
    differentiated_positions = {
        "x": [], "y": [], "z": [], "xv": [], "yv": [], "zv": []
    }

    for chromosome in from_pool_chromosomes:
        from_pool_positions["x"].extend(chromosome[:udp.n_sat])
        from_pool_positions["y"].extend(chromosome[udp.n_sat:2 * udp.n_sat])
        from_pool_positions["z"].extend(chromosome[2 * udp.n_sat:3 * udp.n_sat])
        from_pool_positions["xv"].extend(chromosome[3 * udp.n_sat:4 * udp.n_sat])
        from_pool_positions["yv"].extend(chromosome[4 * udp.n_sat:5 * udp.n_sat])
        from_pool_positions["zv"].extend(chromosome[5 * udp.n_sat:6 * udp.n_sat])

    for chromosome in differentiated_chromosomes:
        differentiated_positions["x"].extend(chromosome[:udp.n_sat])
        differentiated_positions["y"].extend(chromosome[udp.n_sat:2 * udp.n_sat])
        differentiated_positions["z"].extend(chromosome[2 * udp.n_sat:3 * udp.n_sat])
        differentiated_positions["xv"].extend(chromosome[3 * udp.n_sat:4 * udp.n_sat])
        differentiated_positions["yv"].extend(chromosome[4 * udp.n_sat:5 * udp.n_sat])
        differentiated_positions["zv"].extend(chromosome[5 * udp.n_sat:6 * udp.n_sat])

    # Prepare data for violin plot
    data = {
        "X-Coordinate": [from_pool_positions["x"], differentiated_positions["x"]],
        "Y-Coordinate": [from_pool_positions["y"], differentiated_positions["y"]],
        "Z-Coordinate": [from_pool_positions["z"], differentiated_positions["z"]],
        "XV-Coordinate": [from_pool_positions["xv"], differentiated_positions["xv"]],
        "YV-Coordinate": [from_pool_positions["yv"], differentiated_positions["yv"]],
        "ZV-Coordinate": [from_pool_positions["zv"], differentiated_positions["zv"]]
    }
    labels = ["From Pool", "Distance-Based (Manhattan)"]
    colors = ["skyblue", "salmon"]

    # Plot violin plots side by side with distinct colors
    fig, axs = plt.subplots(2, 3, figsize=(18, 12), sharey=True)
    coords = list(data.keys())
    for idx, ax in enumerate(axs.flatten()):
        parts = ax.violinplot(data[coords[idx]], showmeans=True, showextrema=True)

        # Set color for each violin plot
        for i, pc in enumerate(parts['bodies']):
            pc.set_facecolor(colors[i])
            pc.set_alpha(0.7)

        ax.set_title(f"{coords[idx]}")
        ax.set_xticks([1, 2])
        ax.set_xticklabels(labels)
        ax.set_ylabel("Position/Velocity Value")

    plt.tight_layout()
    plt.savefig(plot_filepath, dpi=300)
    plt.close()
    print(f"Plot saved to {plot_filepath}")

    # Display the saved plot
    img = mpimg.imread(plot_filepath)
    plt.figure(figsize=(12, 8))
    plt.imshow(img)
    plt.axis('off')
    plt.show()

def figure_5(udp, orbit_name, theoretical_max, achieved_max, num_chromosomes=10):
    """
    Generates and plots histogram for fitness-based selection method.
    
    Args:
        udp: The udp object used for satellite generation.
        orbit_name (str): The orbit name for identifying results.
        theoretical_max (float): The theoretical maximum fitness.
        achieved_max (float): The achieved maximum fitness.
        num_chromosomes (int): Number of chromosomes to generate.
    """
    # Set directory and filepath
    directory = "figures"
    plot_filepath = f"{directory}/figure_5_{orbit_name}.png"
    results_filepath = f"results/results_fitness_based_{orbit_name}.json"
    random.seed(plot_filepath)

    # Ensure directories exist
    ensure_directory_exists(directory)
    ensure_directory_exists("results")

    # Check if results already exist or need regenerating
    if not file_exists(results_filepath):
        print("Generating data for fitness-based selection...")

        # Initialize pool manager and load satellite pool
        pool_manager = SatellitePoolManager(udp, orbit_name, quiet=False)
        pool_manager.ensure_all()
        satellite_pool = pool_manager.satellite_file_manager.load_satellites(pool_manager.get_filenames()[0], quiet=True)[0]

        # Generate and evaluate fitness values
        time1 = time.time()
        factory = ChromosomeFactory(
            udp=udp,
            satellite_pool=satellite_pool,
            pool_name="generated_pool",
            metric_name="none",
            generation_method="from-pool-fitness-based-growing-subset"
        )
        fitness_values = generate_chromosomes_and_evaluate_fitness(factory, num_chromosomes=num_chromosomes)

        # Save results with metadata
        metadata = {
            "orbit": orbit_name,
            "description": "Fitness values for from-pool-fitness-based-growing-subset selection.",
            "num_chromosomes": num_chromosomes,
            "Time_taken": time.time() - time1,
            "satellite_pool_file": pool_manager.get_filenames()[0],
            "datetime": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        save_results_with_metadata(results_filepath, metadata, fitness_values)
        print(f"Results saved to {results_filepath}.")
    else:
        # Load existing results
        fitness_values, _ = load_results_with_metadata(results_filepath)
        print("Loaded existing data for fitness-based selection.")

    # Plot the histogram
    plt.figure(figsize=(6, 4))
    plt.hist(fitness_values, bins=5, color='skyblue', alpha=1, label=f"Fitness-Based ({orbit_name})")

    # Add vertical lines for 0, theoretical max, and achieved max
    plt.axvline(0, color='gray', linestyle='--', linewidth=1.2, label="Baseline (0)")
    plt.axvline(theoretical_max, color='green', linestyle='--', linewidth=1.2, label=f"Theoretical Max ({theoretical_max})")
    plt.axvline(achieved_max, color='red', linestyle='--', linewidth=1.2, label=f"Achieved Max ({achieved_max})")

    # Set plot titles and labels
    plt.title(f"Fitness Distribution for Fitness-Based Selection ({orbit_name})")
    plt.xlabel("Fitness")
    plt.ylabel("Frequency")
    plt.legend(loc='upper left')

    # Save and display the plot
    plt.tight_layout()
    plt.savefig(plot_filepath, dpi=300)
    plt.close()
    print(f"Plot saved to {plot_filepath}")

    # Display the saved plot image
    img = mpimg.imread(plot_filepath)
    plt.figure(figsize=(12, 10))
    plt.imshow(img)
    plt.axis('off')
    plt.show()

def figure_pres_1(udp, orbit_name, theoretical_max, achieved_max, num_chromosomes=NUMBER_OF_CHROMOSOMES):
    """
    Generates and plots histogram for distance-based sampling method for presentation.
    
    Args:
        udp: The udp object used for satellite generation.
        orbit_name (str): The orbit name for identifying results.
        theoretical_max (float): The theoretical maximum fitness.
        achieved_max (float): The achieved maximum fitness.
        num_chromosomes (int): Number of chromosomes to generate.
    """
    # Set directory and filepath
    directory = "figures"
    plot_filepath = f"{directory}/figure_pres_1_{orbit_name}.png"
    results_filepath = f"results/results_distance_based_{orbit_name}.json"
    random.seed(plot_filepath)

    # Ensure directories exist
    ensure_directory_exists(directory)
    ensure_directory_exists("results")

    # Check if results already exist or need regenerating
    if not file_exists(results_filepath):
        print("Generating data for distance-based sampling...")

        # Initialize pool manager and load satellite pool
        pool_manager = SatellitePoolManager(udp, orbit_name, quiet=False)
        pool_manager.ensure_all()
        satellite_pool = pool_manager.satellite_file_manager.load_satellites(pool_manager.get_filenames()[0], quiet=True)[0]

        # Generate and evaluate fitness values
        time1 = time.time()
        factory = ChromosomeFactory(
            udp=udp,
            satellite_pool=satellite_pool,
            pool_name="generated_pool",
            metric_name="manhattan",
            generation_method="from-pool-differentiated"
        )
        fitness_values = generate_chromosomes_and_evaluate_fitness(factory, num_chromosomes=num_chromosomes)

        # Save results with metadata
        metadata = {
            "orbit": orbit_name,
            "description": "Fitness values for from-pool-differentiated selection.",
            "num_chromosomes": num_chromosomes,
            "Time_taken": time.time() - time1,
            "satellite_pool_file": pool_manager.get_filenames()[0],
            "datetime": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        save_results_with_metadata(results_filepath, metadata, fitness_values)
        print(f"Results saved to {results_filepath}.")
    else:
        # Load existing results
        fitness_values, _ = load_results_with_metadata(results_filepath)
        print("Loaded existing data for distance-based sampling.")

    # Plot the histogram with adjustments for small dataset
    plt.figure(figsize=(6, 4))
    plt.hist(fitness_values, bins=50, color='skyblue', alpha=1, label=f"Distance-Based ({orbit_name})")

    # Add vertical lines for 0, theoretical max, and achieved max
    plt.axvline(0, color='gray', linestyle='--', linewidth=1.2, label="Baseline (0)")
    plt.axvline(theoretical_max, color='green', linestyle='--', linewidth=1.2, label=f"Theoretical Max ({theoretical_max})")
    plt.axvline(achieved_max, color='red', linestyle='--', linewidth=1.2, label=f"Achieved Max ({achieved_max})")

    # Set plot titles and labels
    plt.title(f"Fitness Distribution for Distance-Based Sampling ({orbit_name})")
    plt.xlabel("Fitness")
    plt.ylabel("Frequency")
    plt.legend(loc='upper left')

    # Save and display the plot
    plt.tight_layout()
    plt.savefig(plot_filepath, dpi=300)
    plt.close()
    print(f"Plot saved to {plot_filepath}")

    # Display the saved plot image
    img = mpimg.imread(plot_filepath)
    plt.figure(figsize=(12, 10))
    plt.imshow(img)
    plt.axis('off')
    plt.show()

def for_presentation(udp, orbit_name):
    # Set up SatellitePoolManager
    pool_manager = SatellitePoolManager(udp, orbit_name, quiet=False)
    pool_manager.ensure_all()
    satellite_pool = pool_manager.satellite_file_manager.load_satellites(pool_manager.get_filenames()[0], quiet=True)[0]

    # Factory for from-pool method
    factory_from_pool = ChromosomeFactory(
        udp=udp,
        satellite_pool=satellite_pool,
        pool_name="generated_pool",
        metric_name="none",
        generation_method="from-pool"
    )
    # Factory for from-pool-differentiated (Manhattan)
    factory_differentiated = ChromosomeFactory(
        udp=udp,
        satellite_pool=satellite_pool,
        pool_name="generated_pool",
        metric_name="manhattan",
        generation_method="from-pool-differentiated"
    )
    # Factory for from-pool-fitness-based-growing-subset
    factory_fitness_based = ChromosomeFactory(
        udp=udp,
        satellite_pool=satellite_pool,
        pool_name="generated_pool",
        metric_name="none",
        generation_method="from-pool-fitness-based-growing-subset"
    )

    # Generate and time from-pool
    pool_chromo = factory_from_pool()
    diff_chromo = factory_differentiated()
    fitness_chromo = factory_fitness_based()
    
    udp.plot(pool_chromo)
    udp.plot(diff_chromo)
    udp.plot(fitness_chromo)

def generate_and_time_factory(factory, num_chromosomes):
    """Generate chromosomes and time the generation process for a given factory."""
    start_time = time.time()
    fitness_values = generate_chromosomes_and_evaluate_fitness(factory, num_chromosomes)
    time_elapsed = time.time() - start_time
    avg_fitness = np.mean(fitness_values)
    return avg_fitness, time_elapsed

def for_table(udp, orbit_name, num_chromosomes_pool=1000, num_chromosomes_fitness_based=10):
    # Set up SatellitePoolManager
    pool_manager = SatellitePoolManager(udp, orbit_name, quiet=False)
    pool_manager.ensure_all()
    satellite_pool = pool_manager.satellite_file_manager.load_satellites(pool_manager.get_filenames()[0], quiet=True)[0]

    # Factory for from-pool method
    factory_from_pool = ChromosomeFactory(
        udp=udp,
        satellite_pool=satellite_pool,
        pool_name="generated_pool",
        metric_name="none",
        generation_method="from-pool"
    )
    # Factory for from-pool-differentiated (Manhattan)
    factory_differentiated = ChromosomeFactory(
        udp=udp,
        satellite_pool=satellite_pool,
        pool_name="generated_pool",
        metric_name="manhattan",
        generation_method="from-pool-differentiated"
    )
    # Factory for from-pool-fitness-based-growing-subset
    factory_fitness_based = ChromosomeFactory(
        udp=udp,
        satellite_pool=satellite_pool,
        pool_name="generated_pool",
        metric_name="none",
        generation_method="from-pool-fitness-based-growing-subset"
    )

    # Generate and time from-pool
    avg_fitness_pool, time_pool = generate_and_time_factory(factory_from_pool, num_chromosomes_pool)

    # Generate and time from-pool-differentiated
    avg_fitness_differentiated, time_differentiated = generate_and_time_factory(factory_differentiated, num_chromosomes_pool)

    # Generate and time from-pool-fitness-based-growing-subset (using fewer chromosomes due to high computational cost)
    avg_fitness_fitness_based, time_fitness_based = generate_and_time_factory(factory_fitness_based, num_chromosomes_fitness_based)

    # Calculate % increase in fitness
    fitness_increase_differentiated = ((avg_fitness_differentiated - avg_fitness_pool) / avg_fitness_pool) * 100
    fitness_increase_fitness_based = ((avg_fitness_fitness_based - avg_fitness_pool) / avg_fitness_pool) * 100

    # Adjust timing for comparable scales
    adjusted_time_fitness_based = (time_fitness_based / num_chromosomes_fitness_based) * num_chromosomes_pool

    # Print results summary
    print(f"Method Comparison for {orbit_name}:\n")
    print(f"{'Method':<35}{'Avg Fitness':<15}{'Time Taken (s)':<20}{'% Increase over Pool':<25}")
    print("-" * 95)
    print(f"{'From-Pool':<35}{avg_fitness_pool:<15.4f}{time_pool:<20.2f}{'-':<25}")
    print(f"{'From-Pool Differentiated (Manhattan)':<35}{avg_fitness_differentiated:<15.4f}{time_differentiated:<20.2f}{fitness_increase_differentiated:<25.2f}")
    print(f"{'From-Pool Fitness-Based Growing Subset':<35}{avg_fitness_fitness_based:<15.4f}{adjusted_time_fitness_based:<20.2f}{fitness_increase_fitness_based:<25.2f}")






































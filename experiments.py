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


from results_generators import generate_chromosomes_and_evaluate_fitness, time_alive_generation


NUMBER_OF_CHROMOSOMES = 1000
POOL_SIZES = 500
TIGHT_BOUND_EXPANSION = 0.1
PLOT_SIZES = 20, 20

def calculate_statistics(values):
    return {
        'min': np.min(values),
        'q1': np.percentile(values, 25),
        'median': np.median(values),
        'q3': np.percentile(values, 75),
        'max': np.max(values),
        'mean': np.mean(values),
        'std_dev': np.std(values)
    }



def compare_pools_statistically(pool1, pool2):
    """
    Compare two pools of satellite data statistically using multiple tests and return results as a table-ready structure.

    Args:
        pool1: First pool of satellite data (list of lists).
        pool2: Second pool of satellite data (list of lists).

    Returns:
        results_table: A list of rows representing the statistical results ready for tabulation or further use.
    """
    # Decompose the pools into individual components (x, y, z, xv, yv, zv)
    x1_vals, y1_vals, z1_vals, xv1_vals, yv1_vals, zv1_vals = zip(*pool1)
    x2_vals, y2_vals, z2_vals, xv2_vals, yv2_vals, zv2_vals = zip(*pool2)

    # Combine all component lists
    components = ["X", "Y", "Z", "XV", "YV", "ZV"]
    pool1_vals = [x1_vals, y1_vals, z1_vals, xv1_vals, yv1_vals, zv1_vals]
    pool2_vals = [x2_vals, y2_vals, z2_vals, xv2_vals, yv2_vals, zv2_vals]

    # Threshold for statistical significance
    alpha = 0.05

    # Store the results in a table-ready structure
    results_table = []
    
    # Perform statistical tests for each component
    for comp_name, vals1, vals2 in zip(components, pool1_vals, pool2_vals):
        # Kolmogorov-Smirnov Test
        ks_stat, ks_pvalue = ks_2samp(vals1, vals2)
        ks_significant = "Yes" if ks_pvalue < alpha else "No"

        # T-test for Means
        t_stat, t_pvalue = ttest_ind(vals1, vals2, equal_var=False)
        t_significant = "Yes" if t_pvalue < alpha else "No"

        # Levene's Test for Variance Comparison
        levene_stat, levene_pvalue = levene(vals1, vals2)
        levene_significant = "Yes" if levene_pvalue < alpha else "No"

        # Add the row to the table
        results_table.append([
            comp_name,
            f"{ks_stat:.4f}", f"{ks_pvalue:.4f}", ks_significant,
            f"{t_stat:.4f}", f"{t_pvalue:.4f}", t_significant,
            f"{levene_stat:.4f}", f"{levene_pvalue:.4f}", levene_significant
        ])

    # Return the table-ready structure
    return results_table

def print_statistical_comparison_table(pool1, pool2):
    """
    Perform statistical comparison of two pools and print the results in a tabulated format.
    
    Args:
        pool1: First pool of satellite data.
        pool2: Second pool of satellite data.
    """
    # Get the statistical comparison results
    results_table = compare_pools_statistically(pool1, pool2)

    # Define the headers for the table
    headers = [
        "Component", 
        "KS Stat", "KS P-value", "KS Significant",
        "T-test Stat", "T-test P-value", "T-test Significant",
        "Levene Stat", "Levene P-value", "Levene Significant"
    ]

    # Print the table using tabulate
    print("\nStatistical Comparison Between Pools:")
    print(tabulate(results_table, headers, tablefmt="grid"))

"""
EXPERIMENT 1 : POOL REFINEMENT COMPARISON
"""

# generation
def experiment_pool_refinement_generate(udp, orbit_name: str, number_of_chromosomes: int = NUMBER_OF_CHROMOSOMES):
    experiment_name = "pool_refinement_comparison"
    directory = f"results/{orbit_name}/{experiment_name}"
    filename = f"{directory}/results"
    ensure_directory_exists(directory)


    # Early return if already completed
    if file_exists(filename):
        print(f"Results already exist. Loading from {filename}.json")
        return load_results_with_metadata(filename)

    # Ensure all pools are initialized
    pool_manager = SatellitePoolManager(udp, orbit_name, quiet=False)
    pool_manager.ensure_all()

    print("Generating results...")
    start_time = time.time()

    seed = f"{experiment_name}_{orbit_name}"
    random.seed(seed)

    pool_files = pool_manager.get_filenames()
    satellite_pools = {
        pool_file: pool_manager.satellite_file_manager.load_satellites(pool_file, quiet=True)[0]
        for pool_file in pool_files
    }

    pool_results = []

    # Step 3: Iterate over pools and evaluate "from-pool" and "from-pool-differentiated"
    for pool_name, satellite_pool in satellite_pools.items():
        print(f"Evaluating pool: from-pool - {pool_name}")

        # Generate chromosomes using the "from-pool" method and evaluate fitness
        pool_results.append({
            "pool_name": f"from-{pool_name}",
            "avg_fitness": generate_chromosomes_and_evaluate_fitness(
                ChromosomeFactory(
                    udp=udp,
                    satellite_pool=satellite_pool,
                    pool_name=pool_name,
                    generation_method="from-pool"
                ),
                num_chromosomes=number_of_chromosomes
            )
        })

        print(f"Evaluating pool: from-pool-differentiated - {pool_name}")

        # Generate chromosomes using the "from-pool-differentiated" method and evaluate fitness
        pool_results.append({
            "pool_name": f"from-{pool_name}-differentiated",
            "avg_fitness": generate_chromosomes_and_evaluate_fitness(
                ChromosomeFactory(
                    udp=udp,
                    satellite_pool=satellite_pool,
                    pool_name=pool_name,
                    generation_method="from-pool-differentiated"
                ),
                num_chromosomes=number_of_chromosomes
            )
        })

    # Step 4: Save results with metadata
    metadata = {
        "orbit": orbit_name,
        "description": "Comparison of fitness results between different satellite pool generation methods.",
        "num_chromosomes": number_of_chromosomes,
        "satellite_pool_files": pool_files,
        "methods": ["from-pool", "from-pool-differentiated"], #, "from-bounds"],
        "elapsed_time_sec": time.time() - start_time,
        "seed": seed,
        "datetime": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    save_results_with_metadata(filename, metadata, pool_results)
    print(f"Results saved to {directory}/{filename}.json.")

    return pool_results, metadata

# plotting
def experiment_pool_refinement_plot(udp, orbit_name: str):
    # Set the directory based on orbit_name
    experiment_name = "pool_refinement_comparison"
    directory = f"results/{orbit_name}/{experiment_name}"

    results_filename = "results"
    plot_filename = "plot_1"
    plot_metadata_filename = f"{plot_filename}_metadata"

    results_filepath = f"{directory}/{results_filename}"
    plot_metadata_filepath = f"{directory}/{plot_metadata_filename}"
    plot_filename_filepath = f"{directory}/{plot_filename}"

    # Ensure directory exists
    ensure_directory_exists(directory)

    # Load pool results and metadata
    pool_results, metadata = load_results_with_metadata(results_filepath)

    existing_plot_metadata = load_plot_metadata(plot_metadata_filepath)

    if is_generated_from(existing_plot_metadata, metadata):
        print(f"Plot already generated from current data. Skipping plot generation.")

        # Load and display the saved plot using matplotlib
        img = mpimg.imread(plot_filename_filepath+".png")
        plt.figure(figsize=(PLOT_SIZES))
        plt.imshow(img)
        plt.axis('off')  # Hide axes for better display
        plt.show()  # Show the image of the existing plot
        return

    # Current plot metadata to save
    current_plot_metadata = {
        "description": "Box plot showing fitness comparison between satellite pool methods.",
        "generated_from": metadata["metadata_id"],  # Unique ID of the data the plot was generated from
    }

    # Initialize a list of colors for different pools
    colors = ['blue', 'orange', 'green', 'red', 'purple', 'brown', 'pink']

    # Create a figure with two subplots: one for histograms and one for boxplots
    fig, axs = plt.subplots(2, 1, figsize=(10, 12))  # Two rows, one column

    # First subplot: Overlaid histograms
    for idx, result in enumerate(pool_results):
        pool_name = result['pool_name']
        avg_fitness = result['avg_fitness']
        
        # Ensure avg_fitness is a flat list (1D array)
        if isinstance(avg_fitness, list) and all(isinstance(item, list) for item in avg_fitness):
            avg_fitness = [item for sublist in avg_fitness for item in sublist]  # Flatten list
        
        # Plot the histogram with more bins and transparency in the first subplot
        axs[0].hist(avg_fitness, bins=20, alpha=0.5, label=pool_name, color=colors[idx % len(colors)])

    # Add labels, title, legend, and grid to the first subplot
    axs[0].set_title("Fitness Distribution for All Pools")
    axs[0].set_xlabel("Fitness")
    axs[0].set_ylabel("Frequency")
    axs[0].legend(loc='upper right')
    axs[0].grid(True)

    # Second subplot: Combined boxplots
    fitness_data = [result['avg_fitness'] for result in pool_results]
    labels = [result['pool_name'] for result in pool_results]

    # Plot boxplots in the second subplot
    axs[1].boxplot(fitness_data, labels=labels)

    # Add labels, title, and grid to the second subplot
    axs[1].set_title("Fitness Boxplot Comparison for All Pools")
    axs[1].set_ylabel("Fitness")
    axs[1].grid(True)

    # Adjust the layout to avoid overlap
    plt.tight_layout()

    # Save the combined plot
    plt.savefig(plot_filename_filepath, dpi=300)
    plt.close()
    print(f"Plot saved to {plot_filename_filepath}.png")

    # Save plot metadata using save_plot_metadata
    save_plot_metadata(plot_metadata_filepath, current_plot_metadata)
    print(f"Plot metadata saved to {plot_metadata_filepath}")

    # Display the generated plot
    img = mpimg.imread(plot_filename_filepath+".png")
    plt.figure(figsize=(PLOT_SIZES))
    plt.imshow(img)
    plt.axis('off')  # Hide axes for better display
    plt.show()

# stats
def experiment_pool_refinement_stats(udp, orbit_name: str):
    # Set the directory based on orbit_name
    experiment_name = "pool_refinement_comparison"
    directory = f"results/{orbit_name}/{experiment_name}"
    
    results_filename = "results"
    stats_filename = "stats_1"
    stats_metadata_filename = f"{stats_filename}_metadata"

    results_filepath = f"{directory}/{results_filename}"
    stats_filepath = f"{directory}/{stats_filename}"
    stats_metadata_filepath = f"{directory}/{stats_metadata_filename}.json"

    # Ensure directory exists
    ensure_directory_exists(directory)

    # Load pool results and metadata
    pool_results, metadata = load_results_with_metadata(results_filepath)

    # Initialize a list to store the stats for each pool (for table printing)
    stats_table = []
    
    # Loop through each pool's results and calculate stats
    for result in pool_results:
        pool_name = result["pool_name"]
        fitness_values = result["avg_fitness"]

        # Calculate statistics
        stats = calculate_statistics(fitness_values)

        # Append stats to the table (in row format)
        stats_table.append([
            pool_name,
            f"{stats['min']:.5f}",
            f"{stats['q1']:.5f}",
            f"{stats['median']:.5f}",
            f"{stats['q3']:.5f}",
            f"{stats['max']:.5f}",
            f"{stats['mean']:.5f}",
            f"{stats['std_dev']:.5f}"
        ])

    # Print the table with headers
    headers = ["Method Name", "Min", "Q1 (25%)", "Median (Q2)", "Q3 (75%)", "Max", "Mean", "Std Dev"]
    print("\nFitness Statistics:")
    print(tabulate(stats_table, headers, tablefmt="grid"))

    # Current stats metadata to save
    current_stats_metadata = {
        "description": "Fitness statistics (min, max, median, etc.) for each pool method.",
        "generated_from": metadata["metadata_id"],  # Use the metadata ID from results
    }

    # Save the statistics to a file
    save_results_with_metadata(stats_filepath, current_stats_metadata, stats_table)
    print(f"Stats saved to {stats_filepath}")


"""
EXPERIMENT 2 : Distance Metric Comparison
"""

def print_distance_metrics():
    metrics = DistanceMetric.get_metrics()

    # Prepare metrics as a list of lists (for tabulate)
    metrics_table = [[metric_name] for metric_name in metrics]
    
    # Print the metrics in a table format
    print("Available Distance Metrics:")
    print(tabulate(metrics_table, headers=["Distance Metric"], tablefmt="grid"))

def experiment_distance_metric_comparison_generate(udp, orbit_name: str, number_of_chromosomes: int = NUMBER_OF_CHROMOSOMES):
    # Set the directory and file paths
    experiment_name = "distance_metric_comparison"
    directory = f"results/{orbit_name}/{experiment_name}"
    filename = "results"
    results_filepath = f"{directory}/{filename}"

    # Ensure directory exists
    ensure_directory_exists(directory)

    # Step 1: Check if the results file already exists
    if file_exists(results_filepath):
        print(f"Comparison already computed. Loading results from {results_filepath}.")
        distance_metric_results, metadata = load_results_with_metadata(results_filepath)
    else:


        # Ensure all pools are initialized
        pool_manager = SatellitePoolManager(udp, orbit_name, quiet=False)
        pool_manager.ensure_all()

        start_time = time.time()
        seed = f"distance_metric_comparison_{orbit_name}"
        random.seed(seed)

        # Step 2: Load satellite pool from file (assuming first pool file)
        pool_filename = pool_manager.get_filenames()[0]
        satellite_pool = pool_manager.satellite_file_manager.load_satellites(pool_filename, quiet=True)[0]

        distance_metric_results = []

        # Step 3: Evaluate fitness for each distance metric
        for metric_name in DistanceMetric.get_metrics():
            print(f"Evaluating for distance metric: {metric_name}")
            fitness = generate_chromosomes_and_evaluate_fitness(
                ChromosomeFactory(
                    udp=udp,
                    satellite_pool=satellite_pool,
                    pool_name=pool_filename,
                    metric_name=metric_name,
                    generation_method="from-pool-differentiated"
                ),
                num_chromosomes=number_of_chromosomes
            )
            
            # Append the results for this metric
            distance_metric_results.append({
                "metric_name": f"differentiated-{metric_name}",
                "avg_fitness": fitness
            })

        # Step 4: Evaluate fitness for the basic "from-pool" method
        print(f"Evaluating method: from-pool")
        from_pool_fitness = generate_chromosomes_and_evaluate_fitness(
            ChromosomeFactory(
                udp=udp,
                satellite_pool=satellite_pool,
                pool_name=pool_filename,
                metric_name="none",  # No metric for basic from-pool method
                generation_method="from-pool"
            ),
            num_chromosomes=number_of_chromosomes
        )

        distance_metric_results.append({
            "metric_name": "from-pool",
            "avg_fitness": from_pool_fitness
        })

        # Step 5: Save results with metadata
        metadata = {
            "orbit": orbit_name,
            "description": "Comparison of fitness results between distance metrics used in from-pool-differentiated.",
            "num_chromosomes": number_of_chromosomes,
            "satellite_pool_files": pool_filename,
            "metrics": [f"from-pool-differentiated-{metric_name}" for metric_name in DistanceMetric.get_metrics()] + ["from-pool"],
            "elapsed_time_sec": time.time() - start_time,
            "seed": seed,
            "datetime": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }

        # Save the results with metadata
        save_results_with_metadata(results_filepath, metadata, distance_metric_results)
        print(f"Results saved to {results_filepath}.")

def experiment_distance_metric_comparison_plot(udp, orbit_name):
    # Set the directory and file paths
    experiment_name = "distance_metric_comparison"
    directory = f"results/{orbit_name}/{experiment_name}"

    results_filename = "results"
    plot_filename = "plot_1"
    plot_metadata_filename = f"{plot_filename}_metadata"

    results_filepath = f"{directory}/{results_filename}"
    plot_metadata_filepath = f"{directory}/{plot_metadata_filename}"
    plot_filename_filepath = f"{directory}/{plot_filename}"

    # Ensure the directory exists
    ensure_directory_exists(directory)

    # Load the distance metric results and metadata
    distance_metric_results, metadata = load_results_with_metadata(results_filepath)

    # Load existing plot metadata to check if plot has already been generated
    existing_plot_metadata = load_plot_metadata(plot_metadata_filepath)

    if is_generated_from(existing_plot_metadata, metadata):
        print(f"Plot already generated from current data. Skipping plot generation.")

        # Load and display the saved plot using matplotlib
        img = mpimg.imread(plot_filename_filepath+".png")
        plt.figure(figsize=(PLOT_SIZES))
        plt.imshow(img)
        plt.axis('off')  # Hide axes for better display
        plt.show()  # Show the image of the existing plot
        return

    # Current plot metadata to save
    current_plot_metadata = {
        "description": "Fitness distribution and boxplot comparison between distance metrics.",
        "generated_from": metadata["metadata_id"],  # Unique ID of the data the plot was generated from
    }

    # Extract data for plotting
    metric_names = [result["metric_name"] for result in distance_metric_results]
    fitness_values = [result["avg_fitness"] for result in distance_metric_results]

    # Create a figure with two subplots: one for histograms and one for boxplots
    fig, axs = plt.subplots(2, 1, figsize=(14, 10))

    # First subplot: Overlaid histograms
    for i, fitness in enumerate(fitness_values):
        axs[0].hist(fitness, bins=50, alpha=0.5, label=metric_names[i])

    axs[0].set_title("Fitness Distribution Comparison")
    axs[0].set_xlabel("Fitness")
    axs[0].set_ylabel("Frequency")
    axs[0].legend(loc='upper right')

    # Second subplot: Combined boxplots
    axs[1].boxplot(fitness_values, labels=metric_names)
    axs[1].set_title("Fitness Boxplot Comparison")
    axs[1].set_ylabel("Fitness")

    # Adjust the layout to avoid overlap
    plt.tight_layout()

    # Save the plot to file
    plt.savefig(plot_filename_filepath, dpi=300)
    plt.close()
    print(f"Plot saved to {plot_filename_filepath}")

    # Save plot metadata using save_plot_metadata
    save_plot_metadata(plot_metadata_filepath, current_plot_metadata)
    print(f"Plot metadata saved to {plot_metadata_filepath}")

    # Display the generated plot
    img = mpimg.imread(plot_filename_filepath+".png")
    plt.figure(figsize=(PLOT_SIZES))

    plt.imshow(img)
    plt.axis('off')  # Hide axes for better display
    plt.show()

def experiment_distance_metric_comparison_stats(udp, orbit_name: str):
    experiment_name = "distance_metric_comparison"
    directory = f"results/{orbit_name}/{experiment_name}"
    
    results_filename = "results"
    stats_filename = "stats_1"
    stats_metadata_filename = f"{stats_filename}_metadata"

    results_filepath = f"{directory}/{results_filename}"
    stats_filepath = f"{directory}/{stats_filename}"
    stats_metadata_filepath = f"{directory}/{stats_metadata_filename}"

    # Ensure directory exists
    ensure_directory_exists(directory)

    # Load distance metric results and metadata
    distance_metric_results, metadata = load_results_with_metadata(results_filepath)

    # Initialize a list to store the stats for each metric (for table printing)
    stats_table = []
    
    # Loop through each metric's results and calculate stats
    for result in distance_metric_results:
        metric_name = result["metric_name"]
        fitness_values = result["avg_fitness"]

        # Calculate statistics
        stats = calculate_statistics(fitness_values)

        # Append stats to the table (in row format)
        stats_table.append([
            metric_name,
            f"{stats['min']:.5f}",
            f"{stats['q1']:.5f}",
            f"{stats['median']:.5f}",
            f"{stats['q3']:.5f}",
            f"{stats['max']:.5f}",
            f"{stats['mean']:.5f}",
            f"{stats['std_dev']:.5f}"
        ])

    # Print the table with headers
    headers = ["Metric Name", "Min", "Q1 (25%)", "Median (Q2)", "Q3 (75%)", "Max", "Mean", "Std Dev"]
    print("\nFitness Statistics:")
    print(tabulate(stats_table, headers, tablefmt="grid"))

    # Current stats metadata to save
    current_stats_metadata = {
        "description": "Fitness statistics (min, max, median, etc.) for each distance metric.",
        "generated_from": metadata["metadata_id"],  # Use the metadata ID from results
    }

    # Save the statistics to a file
    save_results_with_metadata(stats_filepath, current_stats_metadata, stats_table)
    print(f"Stats saved to {stats_filepath}")

"""
EXPERIMENT 3 : Satellite component vs fitness
"""

def experiment_satellite_component_vs_fitness_generate(udp, orbit_name: str, number_of_chromosomes: int = NUMBER_OF_CHROMOSOMES):
    """
    Generates satellite configurations, decomposes them into satellite components, and collects fitness values.
    
    Args:
        udp: The udp object used for generating and evaluating chromosomes.
        orbit_name (str): The orbit name, used for identifying the results directory.
        number_of_chromosomes (int): Number of chromosomes to generate (default = NUMBER_OF_CHROMOSOMES).
    """
    # Set the directory and file paths
    experiment_name = "satellite_component_vs_fitness"
    directory = f"results/{orbit_name}/{experiment_name}"
    results_filepath = f"{directory}/results"

    # Ensure directory exists
    ensure_directory_exists(directory)

    # Step 1: Check if the results file already exists
    if file_exists(results_filepath):
        print(f"Results already generated. Loading results from {results_filepath}.")
        return  # Skip if results already exist

    # Ensure all pools are initialized
    pool_manager = SatellitePoolManager(udp, orbit_name, quiet=False)
    pool_manager.ensure_all()

    # Load satellite pool from file (assuming first pool file)
    pool_filename = pool_manager.get_filenames()[0]
    satellite_pool = pool_manager.satellite_file_manager.load_satellites(pool_filename, quiet=True)[0]
    chromosome_controller = ChromosomeController(udp)

    # Initialize ChromosomeFactory
    chromosome_factory = ChromosomeFactory(
        udp=udp,
        satellite_pool=satellite_pool,
        pool_name=pool_filename,
        generation_method="from-pool"
    )

    # Collect satellite components and corresponding fitness values
    satellite_components = []
    fitness_values = []

    # Generate chromosomes and decompose into satellite components
    for _ in range(number_of_chromosomes):
        chromosome = chromosome_factory()
        decomposed_satellites = chromosome_controller.decompose(chromosome)
        fitness = udp.fitness(chromosome)[0]  # Single float fitness

        # Store the satellite components and fitness
        for satellite in decomposed_satellites:
            satellite_components.append(satellite)
            fitness_values.append(fitness)

    # Save results with metadata
    metadata = {
        "orbit": orbit_name,
        "description": "Satellite components vs fitness experiment.",
        "num_chromosomes": number_of_chromosomes,
        "satellite_pool_files": pool_filename,
        "datetime": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }

    save_results_with_metadata(results_filepath, metadata, {"components": satellite_components, "fitness": fitness_values})
    print(f"Results saved to {results_filepath}.")

def experiment_satellite_component_vs_fitness_plot(udp, orbit_name: str):
    """
    Plots satellite components (x, y, z, xv, yv, zv) against fitness.
    
    Args:
        udp: The udp object used for generating and evaluating chromosomes.
        orbit_name (str): The orbit name, used for identifying the results directory.
    """
    # Set the directory and file paths
    experiment_name = "satellite_component_vs_fitness"
    directory = f"results/{orbit_name}/{experiment_name}"
    plot_filename = "plot_1"
    plot_metadata_filename = f"{plot_filename}_metadata"
    results_filepath = f"{directory}/results"
    plot_filepath = f"{directory}/{plot_filename}.png"
    plot_metadata_filepath = f"{directory}/{plot_metadata_filename}"

    # Ensure the directory exists
    ensure_directory_exists(directory)

    # Load satellite components and fitness results
    satellite_data, metadata = load_results_with_metadata(results_filepath)

    # Check if plot already exists
    existing_plot_metadata = load_plot_metadata(plot_metadata_filepath)
    if is_generated_from(existing_plot_metadata, metadata):
        print(f"Plot already generated. Loading from {plot_filepath}.")
        img = mpimg.imread(plot_filepath)
        plt.figure(figsize=(PLOT_SIZES))
        plt.imshow(img)
        plt.axis('off')
        plt.show()
        return

    # Extract satellite components and fitness values
    satellite_components = satellite_data["components"]
    fitness_values = satellite_data["fitness"]
    components = ['x', 'y', 'z', 'xv', 'yv', 'zv']

    # Decompose satellite components into separate lists
    x_vals, y_vals, z_vals, xv_vals, yv_vals, zv_vals = zip(*satellite_components)

    # Create scatter plots for each component vs fitness
    plt.figure(figsize=(12, 18))
    for i, (component_vals, component_name) in enumerate(zip([x_vals, y_vals, z_vals, xv_vals, yv_vals, zv_vals], components), start=1):
        plt.subplot(3, 2, i)
        plt.scatter(component_vals, fitness_values, s=0.1, color="black")
        plt.title(f'{component_name.upper()} vs Fitness')
        plt.xlabel(f'{component_name.upper()}')
        plt.ylabel('Fitness')

    plt.tight_layout()
    plt.savefig(plot_filepath, dpi=300)
    plt.close()
    print(f"Plots saved to {plot_filepath}.")

    # Save plot metadata
    current_plot_metadata = {
        "description": "Satellite component vs fitness plots.",
        "generated_from": metadata["metadata_id"]
    }
    save_plot_metadata(plot_metadata_filepath, current_plot_metadata)

    img = mpimg.imread(plot_filepath)
    plt.figure(figsize=(PLOT_SIZES))
    plt.imshow(img)
    plt.axis('off')
    plt.show()

def experiment_satellite_component_vs_fitness_stats(udp, orbit_name: str):
    # Set the directory and file paths
    experiment_name = "satellite_component_vs_fitness"
    directory = f"results/{orbit_name}/{experiment_name}"
    results_filepath = f"{directory}/results"
    stats_filepath = f"{directory}/stats_1"
    stats_metadata_filepath = f"{directory}/stats_1_metadata"

    # Ensure directory exists
    ensure_directory_exists(directory)

    # Load satellite components and fitness results
    satellite_data, metadata = load_results_with_metadata(results_filepath)

    # Decompose satellite components into separate lists
    satellite_components = satellite_data["components"]
    components = ['x', 'y', 'z', 'xv', 'yv', 'zv']
    x_vals, y_vals, z_vals, xv_vals, yv_vals, zv_vals = zip(*satellite_components)

    # Initialize table for stats
    stats_table = []

    # Calculate statistics for each component
    for component_vals, component_name in zip([x_vals, y_vals, z_vals, xv_vals, yv_vals, zv_vals], components):
        stats = calculate_statistics(component_vals)
        stats_table.append([
            component_name.upper(),
            f"{stats['min']:.5f}",
            f"{stats['max']:.5f}",
            f"{stats['mean']:.5f}",
            f"{stats['std_dev']:.5f}"
        ])

    # Print stats in a table
    headers = ["Component", "Min", "Max", "Mean", "Std Dev"]
    print("\nSatellite Component Statistics:")
    print(tabulate(stats_table, headers, tablefmt="grid"))

    # Save stats and metadata
    stats_metadata = {
        "description": "Statistics for satellite components vs fitness.",
        "generated_from": metadata["metadata_id"]
    }
    save_results_with_metadata(stats_filepath, stats_metadata, stats_table)
    print(f"Statistics saved to {stats_filepath}.")

"""
EXPERIMENT 4 : Tight vs full bounds
"""

def experiment_tight_vs_full_bounds_generate(udp, tight_udp, orbit_name: str, target=POOL_SIZES, expansion_factor=TIGHT_BOUND_EXPANSION):
    # Set the directory and file paths
    experiment_name = "tight_vs_full_bounds"
    directory = f"results/{orbit_name}/{experiment_name}"
    results_filepath = f"{directory}/results"

    # Ensure directory exists
    ensure_directory_exists(directory)

    # Step 1: Check if the results file already exists
    if file_exists(results_filepath):
        print(f"Results already generated. Loading results from {results_filepath}.")
        return  # Skip if results already exist

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

    # Save results with metadata
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

def experiment_tight_vs_full_bounds_plot(udp, orbit_name: str):
    """
    Plots satellite component distributions for tight vs. full bounds.
    
    Args:
        udp: The udp object used for satellite generation.
        orbit_name (str): The orbit name, used for identifying the results directory.
    """
    # Set the directory and file paths
    experiment_name = "tight_vs_full_bounds"
    directory = f"results/{orbit_name}/{experiment_name}"
    plot_filename = "plot_4_1"
    plot_metadata_filename = f"{plot_filename}_metadata"
    results_filepath = f"{directory}/results"
    plot_filepath = f"{directory}/{plot_filename}.png"
    plot_metadata_filepath = f"{directory}/{plot_metadata_filename}"

    # Ensure the directory exists
    ensure_directory_exists(directory)

    # Load the results
    pools_data, metadata = load_results_with_metadata(results_filepath)

    # Check if plot already exists
    existing_plot_metadata = load_plot_metadata(plot_metadata_filepath)
    if is_generated_from(existing_plot_metadata, metadata):
        print(f"Plot already generated. Loading from {plot_filepath}.")
        img = mpimg.imread(plot_filepath)
        plt.figure(figsize=(PLOT_SIZES))  # Set larger figure size
        plt.imshow(img)
        plt.axis('off')
        plt.show()
        return

    # Decompose pools into components
    default_pool = pools_data['default_pool']
    tight_pool = pools_data['tight_pool']
    x_full, y_full, z_full, xv_full, yv_full, zv_full = zip(*default_pool)
    x_tight, y_tight, z_tight, xv_tight, yv_tight, zv_tight = zip(*tight_pool)

    # Planes to be plotted
    planes = [
        ('x', 'y', x_full, y_full, x_tight, y_tight, 'XY Plane'),
        ('x', 'z', x_full, z_full, x_tight, z_tight, 'XZ Plane'),
        ('y', 'z', y_full, z_full, y_tight, z_tight, 'YZ Plane'),
        ('xv', 'yv', xv_full, yv_full, xv_tight, yv_tight, 'XV vs YV Plane'),
        ('xv', 'zv', xv_full, zv_full, xv_tight, zv_tight, 'XV vs ZV Plane'),
        ('yv', 'zv', yv_full, zv_full, yv_tight, zv_tight, 'YV vs ZV Plane')
    ]

    # Plot the planes for both full and tight bounds
    plt.figure(figsize=(18, 24))  # Adjust the figure size as needed
    for i, (comp1, comp2, full_vals1, full_vals2, tight_vals1, tight_vals2, plane_name) in enumerate(planes, start=1):
        plt.subplot(len(planes), 2, 2*i-1)
        plt.scatter(full_vals1, full_vals2, s=2, color="blue", label="Full Bounds")
        plt.title(f'Full Bounds: {plane_name}')
        plt.xlabel(f'{comp1.upper()}')
        plt.ylabel(f'{comp2.upper()}')

        plt.subplot(len(planes), 2, 2*i)
        plt.scatter(tight_vals1, tight_vals2, s=2, color="green", label="Tight Bounds")
        plt.title(f'Tight Bounds: {plane_name}')
        plt.xlabel(f'{comp1.upper()}')
        plt.ylabel(f'{comp2.upper()}')

    plt.tight_layout()
    plt.savefig(plot_filepath, dpi=300)  # Save the plot with high resolution
    plt.close()
    print(f"Plot saved to {plot_filepath}.")

    # Save plot metadata
    current_plot_metadata = {
        "description": "Tight vs full bounds satellite component plots.",
        "generated_from": metadata["metadata_id"]
    }
    save_plot_metadata(plot_metadata_filepath, current_plot_metadata)

    img = mpimg.imread(plot_filepath)
    plt.figure(figsize=(PLOT_SIZES))  # Set larger figure size
    plt.imshow(img)
    plt.axis('off')
    plt.show()

def experiment_tight_vs_full_bounds_plot_2(udp, orbit_name: str):
    """
    Plots satellite component distributions for tight vs. full bounds with both datasets overlapping.
    
    Args:
        udp: The udp object used for satellite generation.
        orbit_name (str): The orbit name, used for identifying the results directory.
    """
    # Set the directory and file paths
    experiment_name = "tight_vs_full_bounds"
    directory = f"results/{orbit_name}/{experiment_name}"
    plot_filename = "plot_4_2"
    plot_metadata_filename = f"{plot_filename}_metadata"
    results_filepath = f"{directory}/results"
    plot_filepath = f"{directory}/{plot_filename}.png"
    plot_metadata_filepath = f"{directory}/{plot_metadata_filename}"

    # Ensure the directory exists
    ensure_directory_exists(directory)

    # Load the results
    pools_data, metadata = load_results_with_metadata(results_filepath)

    # Check if plot already exists
    existing_plot_metadata = load_plot_metadata(plot_metadata_filepath)
    if is_generated_from(existing_plot_metadata, metadata):
        print(f"Plot already generated. Loading from {plot_filepath}.")
        img = mpimg.imread(plot_filepath)
        plt.figure(figsize=(PLOT_SIZES))
        plt.imshow(img)
        plt.axis('off')
        plt.show()
        return

    # Decompose pools into components
    default_pool = pools_data['default_pool']
    tight_pool = pools_data['tight_pool']
    x_full, y_full, z_full, xv_full, yv_full, zv_full = zip(*default_pool)
    x_tight, y_tight, z_tight, xv_tight, yv_tight, zv_tight = zip(*tight_pool)

    # Planes to be plotted
    planes = [
        ('x', 'y', x_full, y_full, x_tight, y_tight, 'XY Plane'),
        ('x', 'z', x_full, z_full, x_tight, z_tight, 'XZ Plane'),
        ('y', 'z', y_full, z_full, y_tight, z_tight, 'YZ Plane'),
        ('xv', 'yv', xv_full, yv_full, xv_tight, yv_tight, 'XV vs YV Plane'),
        ('xv', 'zv', xv_full, zv_full, xv_tight, zv_tight, 'XV vs ZV Plane'),
        ('yv', 'zv', yv_full, zv_full, yv_tight, zv_tight, 'YV vs ZV Plane')
    ]

    # Plot the planes with both datasets overlapping
    plt.figure(figsize=(14, 20))  # Adjust the figure size as needed
    for i, (comp1, comp2, full_vals1, full_vals2, tight_vals1, tight_vals2, plane_name) in enumerate(planes, start=1):
        plt.subplot(3, 2, i)
        plt.scatter(full_vals1, full_vals2, s=20, color="blue", label="Full Bounds", alpha=0.25)
        plt.scatter(tight_vals1, tight_vals2, s=20, color="green", label="Tight Bounds", alpha=0.25)
        plt.title(f'{plane_name}')
        plt.xlabel(f'{comp1.upper()}')
        plt.ylabel(f'{comp2.upper()}')
        plt.legend()
        plt.grid(False)

    plt.tight_layout()
    plt.savefig(plot_filepath, dpi=300)  # Save the plot with high resolution
    plt.close()
    print(f"Plot saved to {plot_filepath}.")

    # Save plot metadata
    current_plot_metadata = {
        "description": "Tight vs full bounds overlapping satellite component plots.",
        "generated_from": metadata["metadata_id"]
    }
    save_plot_metadata(plot_metadata_filepath, current_plot_metadata)

    img = mpimg.imread(plot_filepath)
    plt.figure(figsize=(PLOT_SIZES))
    plt.imshow(img)
    plt.axis('off')
    plt.show()

def experiment_tight_vs_full_bounds_stats(udp, orbit_name: str):
    """
    Compare tight vs full bounds satellite pools statistically and save the results.
    
    Args:
        udp: The udp object used for satellite generation.
        orbit_name (str): The orbit name, used for identifying the results directory.
    """
    # Set the directory and file paths
    experiment_name = "tight_vs_full_bounds"
    directory = f"results/{orbit_name}/{experiment_name}"
    results_filepath = f"{directory}/results"
    stats_filepath = f"{directory}/stats_1"
    stats_metadata_filepath = f"{directory}/stats_1_metadata"

    # Ensure directory exists
    ensure_directory_exists(directory)

    # Load the results
    pools_data, metadata = load_results_with_metadata(results_filepath)

    # Perform statistical comparison
    default_pool = pools_data['default_pool']
    tight_pool = pools_data['tight_pool']

    # Get the comparison results
    results_table = compare_pools_statistically(default_pool, tight_pool)

    # Define the headers for the table
    headers = [
        "Value", 
        "KS Stat", "KS P-value", "KS Sig",
        "T-test Stat", "T-test P-value", "T-test Sig",
        "Levene Stat", "Levene P-value", "Levene Sig"
    ]

    # Print the statistical comparison table
    print("\nStatistical Comparison Between Pools:")
    print(tabulate(results_table, headers, tablefmt="grid"))

    # Prepare the results for saving
    statistical_results = {
        "headers": headers,
        "results": results_table
    }

    # Save stats and metadata
    stats_metadata = {
        "description": "Statistical comparison of tight vs full bounds satellite pools.",
        "generated_from": metadata["metadata_id"]
    }
    
    # Save the statistical comparison results along with the metadata
    save_results_with_metadata(stats_filepath, stats_metadata, statistical_results)
    print(f"Statistics saved to {stats_filepath}.")

"""
EXPERIMENT 5 : Tight vs full bounds
"""

def experiment_fitness_comparison_default_vs_tight_bounds_generate(udp, orbit_name: str, number_of_chromosomes: int = NUMBER_OF_CHROMOSOMES):
    """
    Generates fitness results for default_pool and tight_pool.
    
    Args:
        udp: The udp object used for satellite generation.
        orbit_name (str): The orbit name, used for identifying the results directory.
        number_of_chromosomes (int): Number of chromosomes to generate (default = NUMBER_OF_CHROMOSOMES).
    """
    # Set the directory and file paths
    experiment_name = "fitness_comparison_default_vs_tight_bounds"
    directory = f"results/{orbit_name}/{experiment_name}"
    results_filepath = f"{directory}/results"

    # Ensure directory exists
    ensure_directory_exists(directory)

    # Step 1: Check if the results file already exists
    if file_exists(results_filepath):
        print(f"Results already generated. Loading results from {results_filepath}.")
        return  # Skip if results already exist

    # Load pools (assuming they have already been generated)
    pools_data, _ = load_results_with_metadata(f"results/{orbit_name}/tight_vs_full_bounds/results")
    default_pool = pools_data['default_pool']
    tight_pool = pools_data['tight_pool']

    # Step 2: Generate fitness results for default_pool
    print("Generating fitness for default_pool...")
    default_fitness = generate_chromosomes_and_evaluate_fitness(
        ChromosomeFactory(
            udp=udp,
            satellite_pool=default_pool,
            pool_name="default_pool",
            generation_method="from-pool-differentiated"
        ),
        num_chromosomes=number_of_chromosomes
    )

    # Step 3: Generate fitness results for tight_pool
    print("Generating fitness for tight_pool...")
    tight_fitness = generate_chromosomes_and_evaluate_fitness(
        ChromosomeFactory(
            udp=udp,
            satellite_pool=tight_pool,
            pool_name="tight_pool",
            generation_method="from-pool-differentiated"
        ),
        num_chromosomes=number_of_chromosomes
    )

    # Save the results with metadata
    metadata = {
        "orbit": orbit_name,
        "description": "Fitness comparison between default and tight bounds.",
        "num_chromosomes": number_of_chromosomes,
        "datetime": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    save_results_with_metadata(results_filepath, metadata, {"default_fitness": default_fitness, "tight_fitness": tight_fitness})
    print(f"Results saved to {results_filepath}.")

def experiment_fitness_comparison_default_vs_tight_bounds_plot(udp, orbit_name: str):
    """
    Plots the comparison of fitness distributions between default_pool and tight_pool.
    
    Args:
        udp: The udp object used for satellite generation.
        orbit_name (str): The orbit name, used for identifying the results directory.
    """
    # Set the directory and file paths
    experiment_name = "fitness_comparison"
    directory = f"results/{orbit_name}/{experiment_name}"
    plot_filename = "fitness_comparison_plot"
    plot_metadata_filename = f"{plot_filename}_metadata"
    results_filepath = f"{directory}/results"
    plot_filepath = f"{directory}/{plot_filename}.png"
    plot_metadata_filepath = f"{directory}/{plot_metadata_filename}"

    # Ensure directory exists
    ensure_directory_exists(directory)

    # Load the results
    pools_data, metadata = load_results_with_metadata(results_filepath)

    # Check if plot already exists
    existing_plot_metadata = load_plot_metadata(plot_metadata_filepath)
    if is_generated_from(existing_plot_metadata, metadata):
        print(f"Plot already generated. Loading from {plot_filepath}.")
        img = mpimg.imread(plot_filepath)
        plt.figure(figsize=(PLOT_SIZES))  # Set larger figure size
        plt.imshow(img)
        plt.axis('off')
        plt.show()
        return

    # Extract fitness values
    default_fitness = pools_data["default_fitness"]
    tight_fitness = pools_data["tight_fitness"]

    # Create a figure for histograms and boxplots
    plt.figure(figsize=(14, 8))
    
    # Subplot 1: Histograms of fitness values
    plt.subplot(1, 2, 1)
    plt.hist(default_fitness, bins=30, alpha=0.5, label='Default Bounds')
    plt.hist(tight_fitness, bins=30, alpha=0.5, label='Tight Bounds')
    plt.title("Fitness Distribution Comparison")
    plt.xlabel("Fitness")
    plt.ylabel("Frequency")
    plt.legend(loc='upper right')
    
    # Subplot 2: Boxplots of fitness values
    plt.subplot(1, 2, 2)
    plt.boxplot([default_fitness, tight_fitness], labels=['Default Bounds', 'Tight Bounds'])
    plt.title("Fitness Boxplot Comparison")
    plt.ylabel("Fitness")

    plt.tight_layout()
    plt.savefig(plot_filepath, dpi=300)  # Save the plot with high resolution
    plt.close()
    print(f"Plot saved to {plot_filepath}.")

    # Save plot metadata
    current_plot_metadata = {
        "description": "Fitness comparison between default and tight bounds.",
        "generated_from": metadata["metadata_id"]
    }
    save_plot_metadata(plot_metadata_filepath, current_plot_metadata)

    img = mpimg.imread(plot_filepath)
    plt.figure(figsize=(PLOT_SIZES))  # Set larger figure size
    plt.imshow(img)
    plt.axis('off')
    plt.show()

def experiment_fitness_comparison_default_vs_tight_bounds_stats(udp, orbit_name: str):
    """
    Prints and saves fitness statistics for default and tight bounds.
    
    Args:
        udp: The udp object used for satellite generation.
        orbit_name (str): The orbit name, used for identifying the results directory.
    """
    # Set the directory and file paths
    experiment_name = "fitness_comparison"
    directory = f"results/{orbit_name}/{experiment_name}"
    stats_filepath = f"{directory}/stats_1"
    results_filepath = f"{directory}/results"

    # Ensure directory exists
    ensure_directory_exists(directory)

    # Load the results
    pools_data, metadata = load_results_with_metadata(results_filepath)

    # Extract fitness values
    default_fitness = pools_data["default_fitness"]
    tight_fitness = pools_data["tight_fitness"]

    # Initialize table for stats
    stats_table = []

    # Calculate statistics
    for fitness_values, pool_name in [(default_fitness, "Default Bounds"), (tight_fitness, "Tight Bounds")]:
        stats = calculate_statistics(fitness_values)
        stats_table.append([
            pool_name,
            f"{stats['min']:.5f}",
            f"{stats['q1']:.5f}",
            f"{stats['median']:.5f}",
            f"{stats['q3']:.5f}",
            f"{stats['max']:.5f}",
            f"{stats['mean']:.5f}",
            f"{stats['std_dev']:.5f}"
        ])

    # Print stats in a table
    headers = ["Pool", "Min", "Q1 (25%)", "Median (Q2)", "Q3 (75%)", "Max", "Mean", "Std Dev"]
    print("\nFitness Statistics:")
    print(tabulate(stats_table, headers, tablefmt="grid"))

    # Save stats and metadata
    stats_metadata = {
        "description": "Fitness statistics comparison between default and tight bounds.",
        "generated_from": metadata["metadata_id"]
    }
    save_results_with_metadata(stats_filepath, stats_metadata, stats_table)
    print(f"Statistics saved to {stats_filepath}.")

def experiment_fitness_comparison_default_vs_tight_bounds_tests(udp, orbit_name: str, alpha=0.05):
    """
    Performs statistical tests (T-test, Kolmogorov-Smirnov, and Levene's test) 
    to compare fitness results from default and tight bounds and displays the results in a table.
    
    Args:
        udp: The udp object used for satellite generation.
        orbit_name (str): The orbit name, used for identifying the results directory.
        alpha (float): Significance level for the tests (default = 0.05).
    """
    # Set the directory and file paths
    experiment_name = "fitness_comparison"
    directory = f"results/{orbit_name}/{experiment_name}"
    results_filepath = f"{directory}/results"

    # Ensure directory exists
    ensure_directory_exists(directory)

    # Load the results
    pools_data, metadata = load_results_with_metadata(results_filepath)

    # Extract fitness values
    default_fitness = pools_data["default_fitness"]
    tight_fitness = pools_data["tight_fitness"]

    # Perform statistical tests and store the results in a table
    print("\nPerforming statistical tests...")

    # T-test
    t_stat, t_pvalue = ttest_ind(default_fitness, tight_fitness, equal_var=False)
    t_sig = "Yes" if t_pvalue < alpha else "No"

    # Kolmogorov-Smirnov Test
    ks_stat, ks_pvalue = ks_2samp(default_fitness, tight_fitness)
    ks_sig = "Yes" if ks_pvalue < alpha else "No"

    # Levene's Test for variance comparison
    levene_stat, levene_pvalue = levene(default_fitness, tight_fitness)
    levene_sig = "Yes" if levene_pvalue < alpha else "No"

    # Create a table with the statistical test results
    stats_table = [
        ["Fitness", f"{ks_stat:.4f}", f"{ks_pvalue:.4f}", ks_sig, f"{t_stat:.4f}", f"{t_pvalue:.4f}", t_sig, f"{levene_stat:.4f}", f"{levene_pvalue:.4f}", levene_sig]
    ]

    # Define headers for the table
    headers = ["Value", "KS Stat", "KS P-value", "KS Sig", "T-test Stat", "T-test P-value", "T-test Sig", "Levene Stat", "Levene P-value", "Levene Sig"]

    # Print the table using tabulate
    print("\nStatistical Test Results:")
    print(tabulate(stats_table, headers, tablefmt="grid"))

    # Save results to file
    stats_metadata = {
        "description": "Statistical comparison between default and tight bounds fitness.",
        "generated_from": metadata["metadata_id"],
    }
    stats_filepath = f"{directory}/stats_2"
    save_results_with_metadata(stats_filepath, stats_metadata, stats_table)
    print(f"Statistics saved to {stats_filepath}.")

"""
EXPERIMENT 6 : Fitness comparison using grid based satellite pool abstraction
"""

def experiment_fitness_comparison_using_grid_based_satellite_pool_abstraction_generate(udp, orbit_name: str, target=POOL_SIZES, number_of_chromosomes: int = NUMBER_OF_CHROMOSOMES):
    """
    Generate the experiment results for fitness comparison using grid-based satellite pool abstraction.

    Args:
        tight_udp: The udp object used for satellite generation with custom bounds.
        orbit_name (str): The orbit name, used for identifying the results directory.
        target (int): The number of satellites to select for generating chromosomes.
    """
    # Set the directory and file paths
    experiment_name = "fitness_comparison_grid_based"
    directory = f"results/{orbit_name}/{experiment_name}"
    results_filepath = f"{directory}/results"

    # Ensure the directory exists
    ensure_directory_exists(directory)

    # Early return if the results file already exists
    if file_exists(results_filepath):
        print(f"Results already exist. Loading from {results_filepath}.json")
        return load_results_with_metadata(results_filepath)

    pool_manager = SatellitePoolManager(udp, orbit_name, quiet=False)
    default_pool_file = pool_manager.ensure_satellite_alive_pool(target)
    default_pool , _ = pool_manager.satellite_file_manager.load_satellites(default_pool_file)

    grid_pool_file = pool_manager.ensure_satellite_grid_pool(target)
    grid_data, _ = pool_manager.satellite_file_manager.load_satellites(grid_pool_file)

    # Step 2: Retrieve the satellite pool and corresponding grid positions
    grid_positions = grid_data["grid_positions"]
    grid_pool = grid_data["satellite_pool"]

    chromo = ChromosomeFactory(
            udp=udp,
            satellite_pool=grid_pool,
            pool_name="grid_pool",
            generation_method="from-grid"
        )
    print(udp.plot(chromo()))
    print(udp.plot(chromo()))
    print(udp.plot(chromo()))
    print(udp.plot(chromo()))
    print(udp.plot(chromo()))


    print(udp.plot(ChromosomeFactory(
            udp=udp,
            satellite_pool=default_pool,
            pool_name="grid_pool",
            generation_method="from-pool-differentiated"
            )()))


    # Step 3: Generate fitness results for the grid pool using a new grid selection method
    print("Generating fitness for grid_pool...")
    grid_fitness = generate_chromosomes_and_evaluate_fitness(
        ChromosomeFactory(
            udp=udp,
            satellite_pool=grid_pool,
            pool_name="grid_pool",
            generation_method="from-grid"
        ),
        num_chromosomes=number_of_chromosomes
    )

    print("Generating fitness for default_pool...")
    default_fitness = generate_chromosomes_and_evaluate_fitness(
        ChromosomeFactory(
            udp=udp,
            satellite_pool=grid_pool,
            pool_name="default_pool",
            generation_method="from-pool-differentiated"
        ),
        num_chromosomes=number_of_chromosomes
    )

    # Save the results with metadata
    metadata = {
        "orbit": orbit_name,
        "description": "Fitness comparison between default pool and grid-based selection.",
        "num_chromosomes": number_of_chromosomes,
        "datetime": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    save_results_with_metadata(results_filepath, metadata, {"default_fitness": default_fitness, "grid_fitness": grid_fitness})
    print(f"Results saved to {results_filepath}.")



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







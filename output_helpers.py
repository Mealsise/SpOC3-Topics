import time
import sys
from datetime import timedelta, datetime
import matplotlib.pyplot as plt
import numpy as np


def format_time(seconds):
    # Helper function to format time in minutes and seconds
    return str(timedelta(seconds=seconds))[:-3]  # Remove microseconds

def format_finish_time(seconds_remaining):
    """Helper function to calculate and format the estimated finish time (hh:mm 24-hour format)."""
    finish_time = datetime.now() + timedelta(seconds=seconds_remaining)
    return finish_time.strftime("%H:%M")  # Format in 24-hour hh:mm format


"""
# def print_progress_bar(iteration: int, total: int, prefix: str = '', suffix: str = '', decimals: int = 1, length: int = 50, fill: str = '█', print_end: str = "\r") -> None:
#     \"""
#     Display a progress bar in the terminal.

#     Args:
#         iteration (int): Current iteration (0 to total).
#         total (int): Total iterations.
#         prefix (str): Prefix string (optional).
#         suffix (str): Suffix string (optional).
#         decimals (int): Number of decimal places to display in the percentage complete (default: 1).
#         length (int): Character length of the progress bar (default: 50).
#         fill (str): Bar fill character (default: '█').
#         print_end (str): End character (default: "\r", carriage return).
#     \"""
#     percent = f"{100 * (iteration / float(total)):.{decimals}f}"
#     filled_length = int(length * iteration // total)
#     bar = fill * filled_length + '-' * (length - filled_length)
    
#     # Print the progress bar
#     print(f'\r{prefix} |{bar}| {percent}% {suffix}', end=print_end)
    
#     # Print a new line when the progress is complete
#     if iteration >= total:
#         print()
"""

def print_progress_bar(iteration: int, total: int, prefix: str = 'Progress:', suffix: str = '', decimals: int = 1, length: int = 50, fill: str = '█', print_end: str = "\r") -> None:
    """
    Display a progress bar in the terminal with color.

    Args:
        iteration (int): Current iteration (0 to total).
        total (int): Total iterations.
        prefix (str): Prefix string (optional).
        suffix (str): Suffix string (optional).
        decimals (int): Number of decimal places to display in the percentage complete (default: 1).
        length (int): Character length of the progress bar (default: 50).
        fill (str): Bar fill character (default: '█').
        print_end (str): End character (default: "\r", carriage return).
    """
    percent = f"{100 * (iteration / float(total)):.{decimals}f}"
    filled_length = int(length * iteration // total)

    # Determine the color based on the progress
    if iteration / total < 0.25:
        color = '\033[91m'  # Red
    elif iteration / total < 0.50:
        color = '\033[33m'  # Orange
    elif iteration / total < 0.75:
        color = '\033[93m'  # Yellow
    elif iteration / total < 1.00:
        color = '\033[92m'  # Green
    else:
        color = '\033[96m'  # Teal (for 100% completion)

    reset_color = '\033[0m'  # Reset to default color

    bar = color + fill * filled_length + '-' * (length - filled_length) + reset_color

    # Print the progress bar
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end=print_end)
    
    # Print a new line when the progress is complete
    if iteration >= total:
        print()


def print_progress_bar_with_time(iteration: int, total: int, time_started: float, prefix: str = '', suffix: str = '', decimals: int = 1, length: int = 50, fill: str = '█', print_end: str = "\r") -> None:
    """
    Wrapper function to call print_progress_bar with elapsed and estimated time in the suffix.
    
    Args:
        iteration (int): Current iteration (0 to total).
        total (int): Total iterations.
        time_started (float): Time the process started (timestamp from time.time()).
        Other parameters: Same as in print_progress_bar.
    """
    # Calculate elapsed time
    time_elapsed = time.time() - time_started

    # Calculate estimated remaining time
    time_per_iteration = time_elapsed / iteration if iteration > 0 else 0
    estimated_total_time = time_per_iteration * total
    time_remaining = estimated_total_time - time_elapsed

    # Format elapsed and estimated remaining time
    # elapsed_str = format_time(time_elapsed)
    # remaining_str = format_time(time_remaining)
    # print(elapsed_str)

    elapsed_time_str = time.strftime("%H:%M:%S", time.gmtime(time_elapsed))
    remaining_time_str = time.strftime("%H:%M:%S", time.gmtime(time_remaining))
    finish_time_str = format_finish_time(time_remaining)


    # Construct the suffix with time information
    time_suffix = f"complete | Elapsed: {elapsed_time_str} | Remaining: {remaining_time_str} |  Finish: {finish_time_str} | {suffix}"

    # Call the original print_progress_bar with the updated suffix
    print_progress_bar(iteration, total, prefix=prefix, suffix=time_suffix, decimals=decimals, length=length, fill=fill, print_end=print_end)




# NOTE: Future improvement - revisit plotting functions to support showing 
# both average fitness progress and individual runs for better visualization.
# Reference: `plot_comparison` function for ideas on handling multiple methods 
# with lighter lines for individual runs and a thicker line for averages.


# def plot_comparison(save_file_name=None, show_zero=True, *methods):
#     """
#     Plot a comparison of multiple methods, showing the average and the spread of individual runs.

#     Args:
#         save_file_name: Name of the file to save the plot as (optional).
#         show_zero: Whether to show the zero fitness line (default is True).
#         methods: Variable-length tuple where each element is a (label, fileName) pair.
#                  The label is used in the legend, and fileName is the CSV file to load the data from.
#     """
#     plt.figure(figsize=(10, 6))

#     # Get the default color cycle from matplotlib rcParams
#     color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
#     # color_cycle = ['#FF0000', '#FFBF00', '#0000FF', '#008000', '#4B0082', '#964B00', '#ED7014']
#     # print(color_cycle)

#     # Initialize color index
#     color_index = 0

#     # Iterate over methods and assign a color to each one
#     for label, fileName in methods:
#         # Load the CSV data
#         data = pd.read_csv(fileName)

#         # Extract the number of cycles and the run data (assuming columns are Cycle, Run_1, Run_2, ..., Average)
#         cycles = data['Cycle'].values
#         runs = data.iloc[:, 1:-1].values  # All run columns (excluding Cycle and Average)
#         average = data['Average'].values  # The average column

#         # Get the next color in the color cycle
#         color = color_cycle[color_index % len(color_cycle)]
#         color_index += 1

#         # Plot the individual runs with a lighter/thinner version of the color (same as the average)
#         for run in runs.T:  # Transpose runs to iterate over each individual run
#             plt.plot(cycles, run, color=color, linewidth=0.5, alpha=0.3)  # Thinner, lighter lines for individual runs

#         # Plot the average with a thicker, fully opaque line using the same color
#         plt.plot(cycles, average, label=label, color=color, linewidth=2)

#     # Show zero fitness line if specified
#     if show_zero:
#         plt.axhline(0, color='black', linewidth=1, linestyle='--')

#     # Labeling
#     plt.xlabel("Cycle")
#     plt.ylabel("Fitness")
#     plt.title("Fitness Progress Comparison")
#     plt.legend()
#     plt.grid(True)

#     # Save the plot as a PNG file if a save_file_name is provided
#     if save_file_name:
#         plt.savefig(save_file_name, format='png')
#         print(f"Saved fitness progress plot as {save_file_name}")

#     # Show the plot
#     plt.show()


def plot_fitness_histograms_and_box_plots(pool_results):
    """
    Plot overlaid histograms and combined boxplots for fitness distributions from different chromosome generation methods.
    
    Args:
        pool_results (List[Dict]): A list of dictionaries containing pool name and average fitness results.
    """
    
    # Initialize a list of colors for different pools
    colors = ['blue', 'orange', 'green', 'red', 'purple', 'brown', 'pink']
    
    # Create a figure for the histograms
    plt.figure(figsize=(10, 6))

    # Iterate over each pool's results to plot histograms
    for idx, result in enumerate(pool_results):
        pool_name = result['pool_name']
        avg_fitness = result['avg_fitness']
        
        # Ensure avg_fitness is a flat list (1D array)
        if isinstance(avg_fitness, list) and all(isinstance(item, list) for item in avg_fitness):
            avg_fitness = [item for sublist in avg_fitness for item in sublist]  # Flatten list
        
        # Plot the histogram with more bins and transparency
        plt.hist(avg_fitness, bins=20, alpha=0.5, label=pool_name, color=colors[idx % len(colors)])

    # Add labels, title, legend, and grid
    plt.title(f"Fitness Distribution for All Pools")
    plt.xlabel("Fitness")
    plt.ylabel("Frequency")
    plt.legend(loc='upper right')
    plt.grid(True)

    # Show the histogram plot
    plt.tight_layout()
    plt.show()

    # Create a figure for combined boxplots
    plt.figure(figsize=(10, 6))

    # Gather fitness data for boxplot
    fitness_data = [result['avg_fitness'] for result in pool_results]
    labels = [result['pool_name'] for result in pool_results]

    # Plot all the boxplots together
    plt.boxplot(fitness_data, labels=labels)

    # Add labels, title, and grid
    plt.title(f"Fitness Boxplot Comparison for All Pools")
    plt.ylabel("Fitness")
    plt.grid(True)

    # Show the boxplot
    plt.tight_layout()
    plt.show()

def calculate_fitness_statistics(fitness_values):
    """
    Calculate and return statistics for a list of fitness values.
    
    Args:
        fitness_values (List[float]): A list of fitness values.
    
    Returns:
        dict: A dictionary containing the min, quartiles, mean, std dev, and max.
    """
    return {
        "min": np.min(fitness_values),
        "q1": np.percentile(fitness_values, 25),
        "median": np.median(fitness_values),
        "q3": np.percentile(fitness_values, 75),
        "max": np.max(fitness_values),
        "mean": np.mean(fitness_values),
        "std_dev": np.std(fitness_values)
    }

def print_fitness_stats(fitness_values, method_name):
    # Calculate statistics for each pool
    stats = calculate_fitness_statistics(fitness_values)
    
    # Print useful statistics for each pool
    print(f"\n--- Statistics for fitness from : {method_name} ---")
    print(f"Min: {stats['min']:.5f}")
    print(f"Q1 (25th percentile): {stats['q1']:.5f}")
    print(f"Median (Q2): {stats['median']:.5f}")
    print(f"Q3 (75th percentile): {stats['q3']:.5f}")
    print(f"Max: {stats['max']:.5f}")
    print(f"Mean: {stats['mean']:.5f}")
    print(f"Standard Deviation: {stats['std_dev']:.5f}")



def plot_distance_metric_comparison(distance_metric_results, save_filename=None):
    """
    Plots the comparison of fitness distributions between different distance metrics and from-pool method.
    
    Args:
        distance_metric_results (list[dict]): A list of results with keys 'metric_name' and 'avg_fitness'.
        save_filename (str): The filename to save the plots (optional).
    """
    # Extract data for plotting
    metric_names = [result["metric_name"] for result in distance_metric_results]
    fitness_values = [result["avg_fitness"] for result in distance_metric_results]
    
    # Create a figure for histograms and boxplots
    plt.figure(figsize=(14, 8))
    
    # Subplot 1: Histograms of fitness values
    plt.subplot(1, 2, 1)
    for i, fitness in enumerate(fitness_values):
        plt.hist(fitness, bins=30, alpha=0.5, label=metric_names[i])
    
    plt.title("Fitness Distribution Comparison")
    plt.xlabel("Fitness")
    plt.ylabel("Frequency")
    plt.legend(loc='upper right')
    
    # Subplot 2: Boxplots of fitness values
    plt.subplot(1, 2, 2)
    plt.boxplot(fitness_values, labels=metric_names)
    plt.title("Fitness Boxplot Comparison")
    plt.ylabel("Fitness")

    # Save the plot to file if save_filename is provided
    if save_filename:
        plt.savefig(save_filename, format='png')
        print(f"Plots saved as {save_filename}")

    # Show the plots
    plt.show()



import time

# Test function
def test_progress_bar():
    start = time.time()
    total = 100
    for i in range(total + 1):
        print_progress_bar_with_time(i, total, time_started=start, prefix='Progress:', suffix='Complete', length=50)
        # print_progress_bar(i, total, prefix='Progress:', suffix='Complete', length=50)
        time.sleep(0.05)

if __name__ == '__main__':
    test_progress_bar()
    test_progress_bar()



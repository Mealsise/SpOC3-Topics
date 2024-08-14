from problems.golomb_simple import udp
import random

def mutate_satellite(satellite, mutation_rate=0.1):
    """
    Mutates the position and velocity of a satellite with a small random change.

    Parameters:
    satellite (list): A list representing the satellite's [x, y, z, dx, dy, dz].
    mutation_rate (float): The maximum percentage change for the mutation (e.g., 0.05 for ±5%).

    Returns:
    list: A new satellite list with mutated values.
    """
    def mutate_value(value):
        if random.random() < 0.5:  # 50% chance to mutate this component
            change = value * random.uniform(-mutation_rate, mutation_rate)
            return value + change
        else:
            return value  # No change

    # Apply mutation to each component of the satellite
    x = mutate_value(satellite[0])
    y = mutate_value(satellite[1])
    z = mutate_value(satellite[2])
    dx = mutate_value(satellite[3])
    dy = mutate_value(satellite[4])
    dz = mutate_value(satellite[5])

    # Return the mutated satellite
    return [x, y, z, dx, dy, dz]


def fitness_impact_of_removal(all_satellites):
    # Encode the full set of satellites
    x_encoded_full = [dx[i] for i in range(6) for dx in all_satellites]
    
    # Get the fitness of the full set
    full_fitness = udp.fitness(x_encoded_full)
    
    # List to store the difference in fitness for each satellite
    fitness_differences = []
    
    # Loop through each satellite
    for i in range(len(all_satellites)):
        # Create a new list of satellites with the i-th satellite removed
        modified_satellites = all_satellites[:i] + all_satellites[i+1:] + [[ 0.0,  0.0,  0.0, 0.0, 0.0,  0.0]]
        
        # Encode the modified set of satellites
        x_encoded_modified = [dx[j] for j in range(6) for dx in modified_satellites]
        
        # Get the fitness of the modified set
        modified_fitness = udp.fitness(x_encoded_modified)
        
        # Calculate the difference in fitness
        fitness_difference = full_fitness[0] - modified_fitness[0]
        fitness_differences.append(fitness_difference)
    
    return fitness_differences

if (__name__ == '__main__'):
    # Example usage:
    satellite_5 = [ 0.1, -0.2, -0.3,  0.0000,  0.001, -0.0117]
    mutated_satellite_5 = mutate_satellite(satellite_5)

    print("Original Satellite:", satellite_5)
    print("Mutated Satellite:", mutated_satellite_5)
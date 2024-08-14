from problems.golomb_simple import udp
import matplotlib.pyplot as plt
import random
from satelliteUtlils import *


# Relative position and velocity of satellites: dx, dy, dz, dvx, dvy, dvz
satellite_1 = [ 0.1,  0.0,  0.1, -0.0042, -0.004,  0.0023 ]
satellite_2 = [ 0.0,  0.2,  0.4, -0.0001,  0.001,  0.0003 ]
satellite_3 = [ 0.0, -0.1,  0.3,  0.0201,  0.025,  0.0001 ]
satellite_4 = [ 0.2,  0.0, -0.2, -0.0001,  0.003, -0.0100 ]
satellite_5 = [ 0.1, -0.2, -0.3,  0.0000,  0.001, -0.0117 ]

satellites = [ satellite_1, satellite_2, satellite_3, satellite_4, satellite_5 ]

def dynamic_mutate_satellite(satellite, mutation_rate):
    def mutate_value(value):
        change = value * random.uniform(-mutation_rate, mutation_rate)
        return value + change

    return [mutate_value(val) for val in satellite]

mutation_rate = 0.1
no_improvement_count = 0

# Print the header for the table
print(f"| {'gen':<4} | {'fitness':<8} | {'improve':<8} | {'notes':<30} |")
print("-" * 60)

for generation in range(100):
    # Encode the current satellites and calculate fitness
    x_encoded = [dx[i] for i in range(6) for dx in satellites]
    current_fitness = udp.fitness(x_encoded)[0]  # Assume fitness is returned as a list
    current_fitness_str = f"{abs(current_fitness):.4f}"

    # Calculate fitness impact of each satellite
    fitness_impacts = fitness_impact_of_removal(satellites)
    smallest_impact_index = random.randint(0, 4) #fitness_impacts.index(max(fitness_impacts))

    # Generate 10 children by mutating the selected satellite
    children = []
    for _ in range(10):
        mutated_satellite = dynamic_mutate_satellite(satellites[smallest_impact_index], mutation_rate)
        new_satellites = satellites[:smallest_impact_index] + [mutated_satellite] + satellites[smallest_impact_index+1:]
        children.append(new_satellites)

    # Evaluate fitness for each child
    child_fitnesses = [udp.fitness([dx[i] for i in range(6) for dx in child])[0] for child in children]
    best_child_index = child_fitnesses.index(max(child_fitnesses))
    best_fitness = child_fitnesses[best_child_index]
    best_fitness_str = f"{abs(best_fitness):.4f}"

    # Determine if there was an improvement
    improved = best_fitness > current_fitness
    improvement_str = "true" if improved else "false"
    notes = ""

    if improved:
        satellites = children[best_child_index]
        no_improvement_count = 0
        notes = "improved"
    else:
        no_improvement_count += 1
        notes = "no improvement"

    # Adaptive mutation rate
    if no_improvement_count > 10:
        mutation_rate = min(mutation_rate * 1.2, 1)
        notes = "increase mutation"
        no_improvement_count = 0
    else:
        mutation_rate = max(mutation_rate * 0.9, 0.01)

    # Controlled reinitialization for diversity
    if generation % 50 == 0 and generation > 0:
        random_satellite_index = random.randint(0, len(satellites) - 1)
        satellites[random_satellite_index] = dynamic_mutate_satellite(satellites[random_satellite_index], mutation_rate=0.5)
        notes = f"reinitialized satellite {random_satellite_index + 1}"

    # Print the current generation status
    print(f"| {generation:<4} | {current_fitness_str:<8} | {improvement_str:<8} | {notes:<30} |")

print("-" * 60)



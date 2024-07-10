#  Dear Examiner, please change the address of the att48.tsp file in order to run the code.

file_path = r'D:\Important Documents\Python file\Assignment 1\att48.tsp'

with open(file_path, 'r') as file:
    tsp_data = file.readlines()

# print(tsp_data[:10])

import numpy as np
import math
import matplotlib.pyplot as plt


def total_distance(tour, distance_matrix):
    return sum(distance_matrix[tour[i - 1], tour[i]] for i in range(len(tour)))

def get_neighbour(tour):
    a, b = np.random.choice(len(tour), 2, replace=False)
    tour[a], tour[b] = tour[b], tour[a]
    return tour

def acceptance_probability(current_energy, new_energy, temperature):
    if new_energy < current_energy:
        return 1
    else:
        return np.exp((current_energy - new_energy) / temperature)

def temperature(fraction):
    return max(0.01, min(1, 1 - fraction))

def simulated_annealing(distance_matrix, initial_tour, max_iterations):
    current_tour = initial_tour
    current_energy = total_distance(current_tour, distance_matrix)
    best_tour = list(current_tour)
    best_energy = current_energy
    
    for k in range(max_iterations):
        T = temperature(k / max_iterations) 
        new_tour = get_neighbour(list(current_tour)) 
        new_energy = total_distance(new_tour, distance_matrix) 
        if acceptance_probability(current_energy, new_energy, T) > np.random.rand():
            current_tour, current_energy = new_tour, new_energy 
        
        if new_energy < best_energy:
            best_tour, best_energy = new_tour, new_energy 
            
    return best_tour, best_energy


def parse_tsp_data(tsp_data):
    node_coord_start = tsp_data.index('NODE_COORD_SECTION\n') + 1
    node_coord_end = tsp_data.index('EOF\n', node_coord_start)
    coord_data = tsp_data[node_coord_start:node_coord_end]
    coordinates = np.array([list(map(float, line.strip().split()[1:])) for line in coord_data])
    return coordinates


coordinates = parse_tsp_data(tsp_data)


def pseudo_euclidean_distance(coord1, coord2):
    xd = coord1[0] - coord2[0]
    yd = coord1[1] - coord2[1]
    rij = math.sqrt((xd**2 + yd**2) / 10.0)
    tij = np.round(rij)
    if tij < rij:
        return tij + 1
    else:
        return tij

distance_matrix = np.array([[pseudo_euclidean_distance(coord1, coord2) 
                            for coord2 in coordinates] for coord1 in coordinates])


initial_tour = np.random.permutation(len(coordinates))

max_iterations = 10000 



# 100 trial runs to tune the parameters
def run_simulated_annealing(distance_matrix, initial_tour, max_iterations):
    best_tour, best_energy = simulated_annealing(distance_matrix, initial_tour, max_iterations)
    return best_tour, best_energy


max_iterations_list = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 900, 2000,
                    2100, 2200, 2300, 2400, 2500, 2600, 2700, 2800, 2900, 3000, 3100, 3200, 3300, 3400, 3500, 3600, 3700, 3800, 3900, 4000,
                    4100, 4200, 4300, 4400, 4500, 4600, 4700, 4800, 4900, 5000, 5100, 5200, 5300, 5400, 5500, 5600, 5700, 5800, 5900, 6000,
                    6100, 6200, 6300, 6400, 6500, 6600, 6700, 6800, 6900, 7000, 7100, 7200, 7300, 7400, 7500, 7600, 7700, 7800, 7900, 8000,
                    8100, 8200, 8300, 8400, 8500, 8600, 8700, 8800, 8900, 9000, 9100, 9200, 9300, 9400, 9500, 9600, 9700, 9800, 9900, 10000]


results = []


for max_iterations in max_iterations_list:
    best_tour, best_energy = run_simulated_annealing(distance_matrix, initial_tour, max_iterations)
    results.append((max_iterations, best_energy))

# Save results to a file
results_file_path = 'D:\Important Documents\Python file\Assignment 1/SA_tuning_results.txt'
with open(results_file_path, 'w') as out_file:
    for max_iterations, energy in results:
        out_file.write(f"Max Iterations: {max_iterations} - Best Energy: {energy}\n")

results_file_path, results 

best_tour, best_energy = simulated_annealing(distance_matrix, initial_tour, max_iterations)

best_tour, best_energy

# Diagram visualisation 
def plot_path(cities, path, title='Best Path'):
    ordered_cities = np.array([cities[i] for i in path])
    plt.figure(figsize=(10, 6))
    plt.plot(ordered_cities[:, 0], ordered_cities[:, 1], 'o-', label='Path')
    plt.plot([ordered_cities[-1, 0], ordered_cities[0, 0]], [ordered_cities[-1, 1], ordered_cities[0, 1]], 'o-')
    plt.title(title)
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.show()

plot_path(coordinates, best_tour, title='Best Tour for att48 TSP')


# Calculate the average distance and standard deviation from the 30 runs
def tsp_data(file_path):
    with open(file_path, 'r') as file:
        tsp_data = file.readlines()

    node_coord_start = tsp_data.index('NODE_COORD_SECTION\n') + 1
    node_coord_end = tsp_data.index('EOF\n', node_coord_start)
    coord_data = tsp_data[node_coord_start:node_coord_end]
    coordinates = np.array([list(map(float, line.strip().split()[1:])) for line in coord_data])
    return coordinates


coordinates_new = tsp_data(r'D:\Important Documents\Python file\Assignment 1\att48.tsp')

distance_matrix_new = np.array([[pseudo_euclidean_distance(coord1, coord2) for coord2 in coordinates_new] for coord1 in coordinates_new])

distances_30_runs_new = []
for run in range(30):
    initial_tour_new = np.random.permutation(len(coordinates_new))
    _, best_energy_new = simulated_annealing(distance_matrix_new, initial_tour_new, 10000)
    distances_30_runs_new.append(best_energy_new)

average_distance_30_new = np.mean(distances_30_runs_new)
std_deviation_30_new = np.std(distances_30_runs_new)

average_distance_30_new, std_deviation_30_new, distances_30_runs_new

print (average_distance_30_new)
print (std_deviation_30_new)
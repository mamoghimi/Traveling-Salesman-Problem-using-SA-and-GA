# Mohammadali Moghimi
# Student ID: 2571855
# Dear Examiner, please change the address of the att48.tsp file in both GA and SA functuions in order to run the code.

def Generic_Algorithm ():
    import numpy as np
    import random
    import matplotlib.pyplot as plt
    import math

    def read_tsp_file(file_path):
        with open(file_path, 'r') as file:
            cities = []
            node_section_found = False
            for line in file:
                if "NODE_COORD_SECTION" in line.strip():
                    node_section_found = True
                    continue
                if node_section_found:
                    parts = line.strip().split()
                    if len(parts) == 3:
                        _, x, y = parts
                        cities.append((float(x), float(y)))
        return cities


    cities_coordinates = read_tsp_file(r'D:\Important Documents\Python file\Assignment 1\att48.tsp')
    num_cities = len(cities_coordinates)
    cities = np.array(cities_coordinates)


    def calc_distance(cityA, cityB):
        return math.sqrt((cityB[0] - cityA[0])**2 + (cityB[1] - cityA[1])**2)


    def total_distance(path, cities):
        distance = 0
        for i in range(len(path)):
            distance += calc_distance(cities[path[i - 1]], cities[path[i]])
        return distance


    def create_initial_population(population_size, num_cities):
        return [random.sample(range(num_cities), num_cities) for _ in range(population_size)]


    def calculate_fitness(population, cities):
        return [1 / total_distance(path, cities) for path in population]


    def select_parents(population, fitness, num_parents):
        parents_indices = np.argsort(fitness)[-num_parents:]
        return [population[i] for i in parents_indices]


    def crossover(parent1, parent2):
        cut = random.randint(0, len(parent1) - 1)
        child = parent1[:cut] + [city for city in parent2 if city not in parent1[:cut]]
        return child


    def mutate(path, mutation_rate):
        if random.random() < mutation_rate:
            swap_indices = random.sample(range(len(path)), 2)
            path[swap_indices[0]], path[swap_indices[1]] = path[swap_indices[1]], path[swap_indices[0]]
        return path


    def genetic_algorithm(cities, population_size=100, num_generations=1000, mutation_rate=0.01):
        population = create_initial_population(population_size, len(cities))
        for _ in range(num_generations):
            fitness = calculate_fitness(population, cities)
            parents = select_parents(population, fitness, population_size // 2)
            next_generation = []
            while len(next_generation) < population_size:
                parent1, parent2 = random.sample(parents, 2)
                child = crossover(parent1, parent2)
                child = mutate(child, mutation_rate)
                next_generation.append(child)
            population = next_generation
        best_index = np.argmax(fitness)
        best_path = population[best_index]
        return best_path, 1 / fitness[best_index]


    def execute_independent_runs(cities, num_runs=30, population_size=50, num_generations=10000, mutation_rate=0.1):
        distances = []
        for _ in range(num_runs):
            _, distance = genetic_algorithm(cities, population_size, num_generations, mutation_rate)
            distances.append(distance)
        
        average_distance = np.mean(distances)
        std_deviation = np.std(distances)
        
        return average_distance, std_deviation, distances

    average_distance, std_deviation, distances = execute_independent_runs(cities, num_runs=30, num_generations=100, mutation_rate=0.1)

    best_path, best_distance = genetic_algorithm(cities, 500, 200, 0.1)
     
    print(f"Average Distance: {average_distance}")
    print(f"Standard Deviation: {std_deviation}")


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

    plot_path(cities, best_path, 'Best Path Found by GA')
    return average_distance,std_deviation
def Simulated_Annealing ():
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

    print (f"Average Distance: {average_distance_30_new}")
    print (f"Standard Deviation: {std_deviation_30_new}")
    return average_distance_30_new,std_deviation_30_new
def wilcoxon (AD_1, AD_2, SD_1, SD_2):
    from scipy.stats import wilcoxon
    w_statistic_ad, p_value_ad = wilcoxon(AD_1, AD_2)
    w_statistic_sd, p_value_sd = wilcoxon(SD_1, SD_2)
    print(f"Average Distance - Wilcoxon test statistic: {w_statistic_ad}, p-value: {p_value_ad}")
    print(f"Standard Deviation - Wilcoxon test statistic: {w_statistic_sd}, p-value: {p_value_sd}")

print ("GA result:")
average_distance_1, standard_deviation_1 = Generic_Algorithm()
print ("--------------------------------------------------------")
print ("SA result:")
average_distance_2, standard_deviation_2 = Simulated_Annealing()
print ("--------------------------------------------------------")
print ("Wilcoxon signed-rank test result:")
wilcoxon(average_distance_1, average_distance_2, standard_deviation_1, standard_deviation_2)
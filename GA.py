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

best_path, best_distance = genetic_algorithm(cities, 50, 200, 0.1)


def execute_independent_runs(cities, num_runs=30, population_size=50, num_generations=10000, mutation_rate=0.1):
    distances = []
    for _ in range(num_runs):
        _, distance = genetic_algorithm(cities, population_size, num_generations, mutation_rate)
        distances.append(distance)
    
    average_distance = np.mean(distances)
    std_deviation = np.std(distances)
    
    return average_distance, std_deviation, distances

average_distance, std_deviation, distances = execute_independent_runs(cities, num_runs=30, num_generations=100, mutation_rate=0.1)

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

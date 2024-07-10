# TSP Solver Using Genetic Algorithm and Simulated Annealing

This project implements two heuristic algorithms, **Genetic Algorithm (GA)** and **Simulated Annealing (SA)**, to solve the Traveling Salesman Problem (TSP). The TSP is a well-known computational problem in which the goal is to find the shortest possible route that visits a set of cities and returns to the origin city.

## Features
- **Genetic Algorithm (GA)**: Uses a population-based approach for finding the shortest path. It includes functions to calculate the distance between cities, generate initial populations, calculate fitness, select parents, perform crossover and mutation, and optimize over generations.
- **Simulated Annealing (SA)**: Employs a probabilistic technique to approximate the global optimum of a given function. It utilizes temperature decay, neighbor generation, and acceptance probability to explore and exploit the solution space.
- **Result Visualization**: Plots the best path found after the optimization runs, allowing visual verification of the algorithm's effectiveness.
- **Performance Analysis**: Executes multiple runs of the algorithms to gather average and standard deviation statistics for distances across runs. This helps in understanding the stability and reliability of the proposed solutions.

## Technologies Used
- Python
- NumPy for numerical operations
- Matplotlib for plotting the results
- Scipy for statistical tests

## How to Use
1. **Setup File Paths**: Adjust the file paths for reading the TSP data file ('att48.tsp') within the code.
2. **Run the Algorithms**: Each algorithm can be run independently using the provided functions `Generic_Algorithm()` for GA and `Simulated_Annealing()` for SA.
3. **View Results**: After running the algorithms, the paths are plotted, and performance metrics are printed out. Comparative statistics using Wilcoxon signed-rank tests are also generated.

## Development Setup
To run this project locally:
1. Clone the repository.
2. Ensure you have Python installed along with the necessary libraries: NumPy, Matplotlib, and Scipy.
3. Load the TSP data file into the same directory as the script, or update the path in the script to where the file is located.
4. Execute the script to see the algorithms in action.

This project is ideal for students, researchers, and hobbyists interested in heuristic optimization algorithms and their application to combinatorial problems like the TSP. It provides a hands-on approach to understanding and implementing GA and SA in Python.

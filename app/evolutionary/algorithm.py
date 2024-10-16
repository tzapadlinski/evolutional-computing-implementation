import random
import matplotlib.pyplot as plt
import numpy as np


population_size = 100
num_vars = 5
generations = 20
p_mutation = 0.1
p_crossover = 0.8
selection_method = 'roulette'
mutation_method = 'point'
crossover_method = 'single'


def fitness_function(chromosome):
    return np.sum(chromosome)


def initialize_population(population_size, num_vars):
    return np.random.randint(2, size=(population_size, num_vars))


def select_parents(population):
    if selection_method == 'roulette':
        fitness_scores = np.array([fitness_function(ind) for ind in population])
        probabilities = fitness_scores / np.sum(fitness_scores)
        return population[np.random.choice(len(population), size=len(population), p=probabilities)]

    elif selection_method == 'tournament':
        selected_parents = []
        for _ in range(len(population)):
            competitors = np.random.choice(len(population), size=3)
            winner = competitors[np.argmax([fitness_function(population[i]) for i in competitors])]
            selected_parents.append(population[winner])
        return np.array(selected_parents)

    elif selection_method == 'best':
        return population[np.argsort([fitness_function(ind) for ind in population])[-len(population) // 2:]]


def crossover(parents):
    offspring = np.empty(parents.shape, dtype=int)
    for k in range(len(parents)):
        if np.random.rand() < p_crossover:
            parent1 = parents[k]
            parent2 = parents[np.random.randint(len(parents))]
            if crossover_method == 'single':
                crossover_point = np.random.randint(1, num_vars)
                offspring[k, :crossover_point] = parent1[:crossover_point]
                offspring[k, crossover_point:] = parent2[crossover_point:]
            elif crossover_method == 'double':
                crossover_point1 = np.random.randint(1, num_vars)
                crossover_point2 = np.random.randint(1, num_vars)
                if crossover_point1 > crossover_point2:
                    crossover_point1, crossover_point2 = crossover_point2, crossover_point1
                offspring[k, :crossover_point1] = parent1[:crossover_point1]
                offspring[k, crossover_point1:crossover_point2] = parent2[crossover_point1:crossover_point2]
                offspring[k, crossover_point2:] = parent1[crossover_point2:]
            elif crossover_method == 'uniform':
                offspring[k] = np.where(np.random.rand(num_vars) < 0.5, parent1, parent2)
        else:
            offspring[k] = parents[k]
    return offspring


def mutate(offspring):
    for idx in range(offspring.shape[0]):
        if np.random.rand() < p_mutation:
            if mutation_method == 'point':
                mutation_point = np.random.randint(0, num_vars)
                offspring[idx, mutation_point] = 1 - offspring[idx, mutation_point]
            elif mutation_method == 'bit':
                mutation_bits = np.random.rand(num_vars) < p_mutation
                offspring[idx] = np.where(mutation_bits, 1 - offspring[idx], offspring[idx])
            elif mutation_method == 'inverse':
                if np.random.rand() < 0.5:
                    offspring[idx] = 1 - offspring[idx]
    return offspring


def genetic_algorithm(population_size, num_vars, generations, p_mutation):
    population = initialize_population(population_size, num_vars)
    print("Початкова популяція:\n", population)

    initial_fitness = [fitness_function(ind) for ind in population]
    print("Фітнес початкової популяції:", initial_fitness)

    for generation in range(generations):
        parents = select_parents(population)
        print("Вибрані батьки:\n", parents)

        offspring = crossover(parents)
        population = mutate(offspring)

        best_idx = np.argmax([fitness_function(ind) for ind in population])
        best_fitness = fitness_function(population[best_idx])
        avg_fitness = np.mean([fitness_function(ind) for ind in population])

        print(f"Generation {generation}: Best Fitness = {best_fitness}, Average Fitness = {avg_fitness}")

    best_idx = np.argmax([fitness_function(ind) for ind in population])
    return population[best_idx], fitness_function(population[best_idx])


best_solution, best_fitness = genetic_algorithm(population_size, num_vars, generations, p_mutation)

print("lepsze rozwiązanie:", best_solution)
print("Najlepsza wartość funkcji:", best_fitness)
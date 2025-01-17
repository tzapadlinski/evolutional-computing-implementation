import argparse
import os
import time
from datetime import datetime

import benchmark_functions as bf
import numpy as np
from matplotlib import pyplot as plt
from pygad import pygad

def main():
    parser = argparse.ArgumentParser(description="Run the Genetic Algorithm with specified parameters.")
    parser.add_argument("--population_size", type=int, default=10, help="Size of the population")
    parser.add_argument("--chromosome_size", type=int, default=24, help="Size of the chromosome")
    parser.add_argument("--num_variables", type=int, default=2, help="Number of variables")
    parser.add_argument("--num_epochs", type=int, default=50, help="Number of generations")
    parser.add_argument("--begin_range", type=float, default=-10.0, help="Lower bound of the variable range")
    parser.add_argument("--end_range", type=float, default=10.0, help="Upper bound of the variable range")
    parser.add_argument("--cross_prob", type=float, default=0.7, help="Crossover probability")
    parser.add_argument("--mutation_prob", type=float, default=0.1, help="Mutation probability")
    parser.add_argument("--uniform_crossover", type=float, default=0.5, help="Uniform crossover probability")
    parser.add_argument("--selection_method", type=str, default="roulette", help="Selection method")
    parser.add_argument("--cross_method", type=str, default="single", help="Crossover method")
    parser.add_argument("--mutation_method", type=str, default="single", help="Mutation method")
    parser.add_argument("--gene_type", type=str, default="real", help="Gene type (real or binary)")
    parser.add_argument("--parent_population", type=int, default=5, help="Number of parents mating")
    parser.add_argument("--elite", type=int, default=5, help="Number of elite solutions")

    args = parser.parse_args()

    population_size = args.population_size
    parent_population = args.parent_population
    generations = args.num_epochs
    p_mutation = args.mutation_prob
    p_crossover = args.cross_prob
    selection_method = args.selection_method
    mutation_method = args.mutation_method
    crossover_method = args.cross_method
    num_variables = args.num_variables
    p_uniform = args.uniform_crossover
    lower_bound = args.begin_range
    upper_bound = args.end_range
    elite = args.elite
    gene_type = args.gene_type

############################ CONSTANTS #############################

    griewank_func = bf.Griewank(n_dimensions=num_variables)
    chromosome_size = 8
    l_bound = -10
    u_bound = 10

    def decode_chromosome(chromosome):
        n = len(chromosome) // chromosome_size
        values = []

        for i in range(n):
            genes_part = chromosome[i * n:(i + 1) * n]
            decimal_value = sum(gene * (2 ** j) for j, gene in enumerate(reversed(genes_part)))
            value = l_bound + decimal_value * (u_bound - l_bound) / (2 ** n - 1)
            values.append(value)

        return np.array(values)

    def fitness_func(ga_instance, solution, solution_idx):
        output = griewank_func(solution)
        fitness = 1.0 / (output + 1.0)
        return fitness

    def fitness_func_binary(ga_instance, solution, solution_idx):
        decoded_solution = decode_chromosome(solution)
        output = griewank_func(decoded_solution)
        fitness = 1.0 / (output + 1.0)
        return fitness

    best_fitness_per_generation = []
    mean_fitness_per_generation = []
    std_fitness_per_generation = []

    def on_generation(ga_instance):
        best_fitness_per_generation.append(ga_instance.best_solution()[1])
        mean_fitness_per_generation.append(np.mean(ga_instance.last_generation_fitness))
        std_fitness_per_generation.append(np.std(ga_instance.last_generation_fitness))

############################## FUNCTIONS ###############################################

    def boundary_mutation(offspring, ga_instance):
        for chromosome_idx in range(offspring.shape[0]):
            boundary_point = np.random.choice([0, offspring.shape[1] - 1])
            offspring[chromosome_idx, boundary_point] = 1 - offspring[chromosome_idx, boundary_point]
        return offspring

    def uniform_mutation(offspring, ga_instance: pygad.GA):
        p_mutation = ga_instance.mutation_probability
        lower_bound = ga_instance.init_range_low
        upper_bound = ga_instance.init_range_high

        for chromosome_idx in range(offspring.shape[0]):
            for gene_index in range(offspring.shape[1]):
                if np.random.rand() < p_mutation:
                    offspring[chromosome_idx, gene_index] = np.random.uniform(lower_bound, upper_bound)
        return offspring

    def gaussian_mutation(offspring, ga_instance):
        p_mutation = ga_instance.mutation_probability
        lower_bound = ga_instance.init_range_low
        upper_bound = ga_instance.init_range_high

        for chromosome_idx in range(offspring.shape[0]):
            for gene_index in range(offspring.shape[1]):
                if np.random.rand() < p_mutation:
                    mutation_value = np.random.normal(0, 1)
                    offspring[chromosome_idx, gene_index] += mutation_value
                    offspring[chromosome_idx, gene_index] = np.clip(offspring[chromosome_idx, gene_index],
                                                                    lower_bound, upper_bound)
        return offspring

    def alpha_blend_cross(parents, offspring_size, ga_instance):
        offspring = np.empty(offspring_size)

        for idx in range(offspring_size[0]):
            parent1 = parents[idx % parents.shape[0], :].copy()
            parent2 = parents[(idx + 1) % parents.shape[0], :].copy()
            alpha = np.random.rand()

            for i in range(len(parent1)):
                d = abs(parent1[i] - parent2[i])
                lower = min(parent1[i], parent2[i]) - alpha * d
                upper = min(parent1[i], parent2[i]) + alpha * d
                offspring[idx, i] = lower + np.random.rand() * (upper - lower)

        return offspring

    def alpha_beta_blend_cross(parents, offspring_size, ga_instance):
        offspring = np.empty(offspring_size)

        for idx in range(offspring_size[0]):
            parent1 = parents[idx % parents.shape[0], :].copy()
            parent2 = parents[(idx + 1) % parents.shape[0], :].copy()
            alpha = np.random.rand()
            beta = np.random.rand()
            for i in range(len(parent1)):
                d = abs(parent1[i] - parent2[i])
                lower = min(parent1[i], parent2[i]) - alpha * d
                upper = min(parent1[i], parent2[i]) + beta * d
                offspring[idx, i] = lower + np.random.rand() * (upper - lower)
        return offspring

    def averaging_cross(parents, offspring_size, ga_instance):
        offspring = np.empty(offspring_size)
        for idx in range(offspring_size[0]):
            parent1 = parents[idx % parents.shape[0], :].copy()
            parent2 = parents[(idx + 1) % parents.shape[0], :].copy()
            for i in range(len(parent1)):
                offspring[idx, i] = (parent1[i] + parent2[i]) / 2.0

        return offspring

    def arithmetic_cross(parents, offspring_size, ga_instance):
        offspring = np.empty(offspring_size)

        for idx in range(offspring_size[0]):
            parent1 = parents[idx % parents.shape[0], :]
            parent2 = parents[(idx + 1) % parents.shape[0], :]

            alpha = np.random.rand()
            beta = 1 - alpha
            offspring[idx, :] = alpha * parent1 + beta * parent2

        return offspring

    def linear_cross(parents, offspring_size, ga_instance):
        offspring = np.empty(offspring_size)

        for idx in range(offspring_size[0]):
            parent1 = parents[idx % parents.shape[0], :]
            parent2 = parents[(idx + 1) % parents.shape[0], :]

            Z = 0.5 * parent1 + 0.5 * parent2
            V = 1.5 * parent1 - 0.5 * parent2
            W = -0.5 * parent1 + 1.5 * parent2
            function_values = [
                Z,
                V,
                W,
            ]
            fitness_values = [ga_instance.fitness_function(child, idx) for idx, child in enumerate(function_values)]
            sorted_indices = np.argsort(fitness_values)
            if ga_instance.optimization_mode == 'max':
                best_fitness_idx = sorted_indices[-1]
            else:
                best_fitness_idx = sorted_indices[0]

            offspring[idx, :] = function_values[best_fitness_idx]

        return offspring
######################################################################################

    custom_crossover_methods = {
        "arithmetic": arithmetic_cross,
        "linear": linear_cross,
        "alpha_blend": alpha_blend_cross,
        "alpha_beta_blend": alpha_beta_blend_cross,
        "averaging": averaging_cross
    }

    custom_mutation_methods = {
        "boundary": boundary_mutation,
        "uniform": uniform_mutation,
        "gaussian": gaussian_mutation
    }

    crossover_method_name = crossover_method
    mutation_method_name = mutation_method

    if args.cross_method in custom_crossover_methods:
        crossover_method = custom_crossover_methods[args.cross_method]
    else:
        crossover_method = args.cross_method

    if args.mutation_method in custom_mutation_methods:
        mutation_method = custom_mutation_methods[args.mutation_method]
    else:
        mutation_method = args.mutation_method

    function = fitness_func
    if gene_type == "binary":
        l_bound = lower_bound
        u_bound = upper_bound
        lower_bound = 0
        upper_bound = 2
        num_variables = num_variables * chromosome_size
        _type = int
        function = fitness_func_binary
    else:
        _type = float

    ga_instance = pygad.GA(num_generations=generations,
                           num_parents_mating=parent_population,
                           fitness_func=function,
                           sol_per_pop=population_size,
                           num_genes=num_variables,
                           gene_type=_type,
                           parent_selection_type=selection_method,
                           crossover_type=crossover_method,
                           mutation_type=mutation_method,
                           init_range_low=lower_bound,
                           init_range_high=upper_bound,
                           mutation_probability=p_mutation,
                           keep_elitism=elite,
                           on_generation=on_generation)
    ga_instance.run()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename_base = f"{selection_method}_{mutation_method_name}_{crossover_method_name}_{timestamp}"

    plt.figure(figsize=(12, 6))
    plt.subplot(2, 1, 1)
    plt.plot(best_fitness_per_generation, label='Best Fitness')
    plt.xlabel('Generation')
    plt.ylabel('Best Fitness')
    plt.title('Best Fitness per Generation')
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(mean_fitness_per_generation, label='Mean Fitness')
    plt.fill_between(range(generations),
                     np.array(mean_fitness_per_generation) - np.array(std_fitness_per_generation),
                     np.array(mean_fitness_per_generation) + np.array(std_fitness_per_generation),
                     color='b', alpha=0.2, label='Std Dev')
    plt.xlabel('Generation')
    plt.ylabel('Fitness')
    plt.title('Mean and Standard Deviation of Fitness per Generation')
    plt.legend()

    plt.tight_layout()

    plot_path = os.path.join(os.getcwd(), f"{filename_base}_fitness_plot.png")
    plt.savefig(plot_path)
    plt.close()

    print(f"Selection Method: {args.selection_method}")
    print(f"Cross Method: {args.cross_method}")
    print(f"Mutation Method: {args.mutation_method}")

if __name__ == "__main__":
    main()
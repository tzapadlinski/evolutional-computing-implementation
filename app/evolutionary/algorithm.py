import os
import time
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

from .chromosome import Chromosome
from .function import Function


class EvolutionaryAlgorithm:

    def __init__(self,
                 population_size,
                 chromosome_size,
                 generations,
                 p_mutation,
                 p_crossover,
                 selection_method,
                 mutation_method,
                 crossover_method,
                 function: 'Function',
                 num_variables,
                 optimization_mode,
                 p_uniform,
                 lower_bound,
                 upper_bound,
                 p_inversion,
                 elite_percentage=0.1,
                 ):
        self.chromosome_size = chromosome_size * num_variables
        self.population = [Chromosome(num_vars=num_variables, lower_bound=lower_bound, upper_bound=upper_bound) for
                           _ in range(population_size)]
        self.generations = generations
        self.p_mutation = p_mutation
        self.p_crossover = p_crossover
        self.selection_method = selection_method
        self.mutation_method = mutation_method
        self.crossover_method = crossover_method
        self.function = function
        self.num_variables = num_variables
        self.optimization_mode = optimization_mode
        self.p_uniform = p_uniform
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.p_inversion = p_inversion
        self.elite_percentage = elite_percentage

    def run(self):
        start_time = time.time()
        best_chromosome = None
        best_fitness_per_generation = []
        mean_fitness_per_generation = []
        std_fitness_per_generation = []
        all_fitness_per_generation = []

        for generation in range(self.generations):
            elite_individuals = self.select_elite()
            parents = self.select_parents()
            offspring = self.crossover(parents)
            offspring = self.mutate(offspring)
            # offspring = self.inverse(offspring)
            self.population = self.update_population(elite_individuals, offspring)

            fitness_scores = [self.function.fit(chromosome.get_value()) for chromosome
                              in self.population]
            best_fitness = max(fitness_scores) if self.optimization_mode == 'max' else min(fitness_scores)
            mean_fitness = np.mean(fitness_scores)
            std_fitness = np.std(fitness_scores)

            best_fitness_per_generation.append(best_fitness)
            mean_fitness_per_generation.append(mean_fitness)
            std_fitness_per_generation.append(std_fitness)
            all_fitness_per_generation.append(fitness_scores)
            best_fitness_idx = fitness_scores.index(best_fitness) if self.optimization_mode == 'max' else fitness_scores.index(best_fitness)
            best_chromosome = self.population[best_fitness_idx]

            print(
                f'Generation {generation + 1}: Best Fitness: {best_fitness}, Mean Fitness: {mean_fitness}, Std Fitness: {std_fitness}')

        execution_time = time.time() - start_time
        print(f"Execution Time: {execution_time:.4f} seconds")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename_base = f"{self.selection_method}_{self.mutation_method}_{self.crossover_method}_{timestamp}"

        fitness_results_path = os.path.join(os.getcwd(), f"{filename_base}_fitness_results.txt")
        with open(fitness_results_path, 'w') as f:
            for generation, fitness_scores in enumerate(all_fitness_per_generation):
                f.write(f"Generation {generation + 1}: {fitness_scores}\n")

        plt.figure(figsize=(12, 6))
        plt.subplot(2, 1, 1)
        plt.plot(best_fitness_per_generation, label='Best Fitness')
        plt.xlabel('Generation')
        plt.ylabel('Best Fitness')
        plt.title('Best Fitness per Generation')
        plt.legend()

        plt.subplot(2, 1, 2)
        plt.plot(mean_fitness_per_generation, label='Mean Fitness')
        plt.fill_between(range(self.generations),
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
        return best_fitness_per_generation, mean_fitness_per_generation, std_fitness_per_generation, all_fitness_per_generation, execution_time, best_chromosome

    # ELITE
    def select_elite(self):
        num_elite = int(self.elite_percentage * len(self.population))

        fitness_scores = np.array(
            [self.function.fit(chromosome.get_value()) for chromosome in
             self.population])
        elite_indices = np.argsort(fitness_scores)[-num_elite:] if self.optimization_mode == 'max' else np.argsort(
            fitness_scores)[:num_elite]
        return [self.population[i] for i in elite_indices]

    def update_population(self, elite_individuals, offspring):
        new_population = list(elite_individuals) + list(offspring[:len(self.population) - len(elite_individuals)])
        return np.array(new_population)

    # SELECTION
    def select_parents(self):
        fitness_scores = np.array(
            [self.function.fit(chromosome.get_value()) for chromosome in
             self.population])

        if self.selection_method == 'roulette':
            if self.optimization_mode == 'max':
                min_fitness = np.min(fitness_scores)
                fitness_scores += abs(min_fitness)
                probabilities = fitness_scores / np.sum(fitness_scores)
            else:
                min_fitness = np.min(fitness_scores)
                fitness_scores += abs(min_fitness)
                fitness_scores = np.where(fitness_scores == 0, np.float64(1e-10), fitness_scores)
                inverse_scores = 1 / fitness_scores
                probabilities = inverse_scores / np.sum(inverse_scores)

            selected_indices = np.random.choice(len(self.population), size=len(self.population), p=probabilities)
            return [self.population[i] for i in selected_indices]

        elif self.selection_method == 'tournament':
            selected_parents = []
            for _ in range(len(self.population)):
                competitors = np.random.choice(len(self.population), size=3, replace=False)
                competitor_fitness = [
                    self.function.fit(self.population[i].get_value()) for i in
                    competitors
                ]
                if self.optimization_mode == 'max':
                    winner_index = competitors[np.argmax(competitor_fitness)]
                else:
                    winner_index = competitors[np.argmin(competitor_fitness)]
                selected_parents.append(self.population[winner_index])
            return np.array(selected_parents)
        elif self.selection_method == 'best':
            if self.optimization_mode == 'max':
                sorted_indices = np.argsort(fitness_scores)[-len(self.population) // 2:]
            else:
                sorted_indices = np.argsort(fitness_scores)[:len(self.population) // 2]
            return [self.population[i] for i in sorted_indices]
        else:
            raise ValueError("Invalid selection method")

    # CROSSOVER
    def crossover(self, parents):
        offspring = []
        num_offspring_needed = len(self.population)
        while len(offspring) < num_offspring_needed:
            parent1 = parents[np.random.randint(len(parents))]
            parent2 = parents[np.random.randint(len(parents))]
            if np.random.rand() < self.p_crossover:
                if self.crossover_method == 'arithmetic':
                    offspring.append(self.arithmetic_cross(parent1, parent2))
                elif self.crossover_method == 'linear':
                    offspring.append(self.linear_cross(parent1, parent2))
                elif self.crossover_method == 'alpha-blended':
                    offspring.append(self.alpha_blend_cross(parent1, parent2))
                elif self.crossover_method == 'alpha-beta-blended':
                    offspring.append(self.alpha_beta_blend_cross(parent1, parent2))
                elif self.crossover_method == 'averaging':
                    offspring.append(self.averaging_cross(parent1, parent2))

        offspring = offspring[:num_offspring_needed]
        return np.array(offspring)

    @staticmethod
    def arithmetic_cross(parent1: Chromosome, parent2: Chromosome):
        alpha = np.random.rand()
        beta = 1 - alpha
        offspring_genes = alpha * parent1.get_value() + beta * parent2.get_value()
        return Chromosome(genes=offspring_genes)

    def linear_cross(self, parent1, parent2):
        parent1 = np.array(parent1.get_value())
        parent2 = np.array(parent2.get_value())

        Z = 0.5 * parent1 + 0.5 * parent2
        V = 1.5 * parent1 - 0.5 * parent2
        W = -0.5 * parent1 + 1.5 * parent2

        function_values = [
            Chromosome(genes=Z),
            Chromosome(genes=V),
            Chromosome(genes=W),
        ]

        function_values_sorted = sorted(function_values, key=lambda x: self.function.fit(x.get_value()))

        if self.optimization_mode == 'max':
            best_fit = function_values_sorted[-1]
        else:
            best_fit = function_values_sorted[0]

        return best_fit

    def alpha_blend_cross(self, parent1, parent2):
        parent1 = np.array(parent1.get_value())
        parent2 = np.array(parent2.get_value())
        alpha = np.random.rand()

        offspring = []

        for i in range(len(parent1)):
            d = abs(parent1[i] - parent2[i])
            lower = min(parent1[i], parent2[i]) - alpha * d
            upper = min(parent1[i], parent2[i]) + alpha * d
            offspring.append(lower + np.random.rand() * (upper - lower))

        return Chromosome(genes=offspring)

    def alpha_beta_blend_cross(self, parent1, parent2):
        parent1 = np.array(parent1.get_value())
        parent2 = np.array(parent2.get_value())
        alpha = np.random.rand()
        beta = np.random.rand()

        offspring = []

        for i in range(len(parent1)):
            d = abs(parent1[i] - parent2[i])
            lower = min(parent1[i], parent2[i]) - alpha * d
            upper = min(parent1[i], parent2[i]) + beta * d
            offspring.append(lower + np.random.rand() * (upper - lower))

        return Chromosome(genes=offspring)

    def averaging_cross(self, parent1, parent2):
        parent1 = np.array(parent1.get_value())
        parent2 = np.array(parent2.get_value())

        offspring = [(parent1[i] + parent2[i]) / 2.0 for i in range(len(parent1))]

        return Chromosome(genes=offspring)

    # MUTATION
    def mutate(self, offspring: np.ndarray):
        for idx in range(offspring.shape[0]):
            if np.random.rand() < self.p_mutation:
                if self.mutation_method == 'uniform':
                    self.uniform_mutation(offspring)
                elif self.mutation_method == 'gaussian':
                    self.gaussian_mutation(offspring)
        return offspring

    def uniform_mutation(self, offspring):
        for chromosome in offspring:
            for gene_index in range(len(chromosome.genes)):
                if np.random.rand() < self.p_mutation:
                    chromosome.genes[gene_index] = np.random.uniform(self.lower_bound, self.upper_bound)
        return offspring

    def gaussian_mutation(self, offspring):
        for chromosome in offspring:
            for gene_index in range(len(chromosome.genes)):
                if np.random.rand() < self.p_mutation:
                    mutation_value = np.random.normal(0, 1)
                    chromosome.genes[gene_index] += mutation_value
                    chromosome.genes[gene_index] = np.clip(chromosome.genes[gene_index], self.lower_bound,
                                                           self.upper_bound)
        return offspring

    # INVERSION should be included??????????????????????????????????????????
    def inverse(self, offspring: np.ndarray):
        pass

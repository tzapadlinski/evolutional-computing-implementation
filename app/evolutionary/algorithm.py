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
        self.population = [Chromosome(chromosome_size=chromosome_size * num_variables, num_variables=num_variables) for
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
            offspring = self.inverse(offspring)
            self.population = self.update_population(elite_individuals, offspring)

            fitness_scores = [self.function.fit(chromosome.get_value(self.lower_bound, self.upper_bound)) for chromosome
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
            [self.function.fit(chromosome.get_value(self.lower_bound, self.upper_bound)) for chromosome in
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
            [self.function.fit(chromosome.get_value(self.lower_bound, self.upper_bound)) for chromosome in
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
                    self.function.fit(self.population[i].get_value(self.lower_bound, self.upper_bound)) for i in
                    competitors]
                winner_index = competitors[np.argmax(competitor_fitness)]
                selected_parents.append(self.population[winner_index])
            return np.array(selected_parents)

        elif self.selection_method == 'best':
            sorted_indices = np.argsort(fitness_scores)[-len(self.population) // 2:]
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
                if self.crossover_method == 'single':
                    offspring.append(self.cross_single(parent1, parent2))
                elif self.crossover_method == 'double':
                    offspring.append(self.cross_double(parent1, parent2))
                elif self.crossover_method == 'uniform':
                    offspring.append(self.cross_uniform(parent1, parent2))
                elif self.crossover_method == 'seeded':
                    offspring.append(self.cross_seeded(parent1, parent2))

        offspring = offspring[:num_offspring_needed]
        return np.array(offspring)

    def cross_single(self, parent1: Chromosome, parent2: Chromosome):
        crossover_point = np.random.randint(1, self.chromosome_size)
        p1_genes = parent1.genes
        p2_genes = parent2.genes
        offspring_genes = p1_genes[:crossover_point] + p2_genes[crossover_point:]
        return Chromosome(genes=offspring_genes, num_variables=self.num_variables)

    def cross_double(self, parent1, parent2):
        crossover_point1 = np.random.randint(1, self.chromosome_size)
        crossover_point2 = np.random.randint(1, self.chromosome_size)
        if crossover_point1 > crossover_point2:
            crossover_point1, crossover_point2 = crossover_point2, crossover_point1

        p1_genes = parent1.genes
        p2_genes = parent2.genes

        offspring_genes = (
                p1_genes[:crossover_point1] +
                p2_genes[crossover_point1:crossover_point2] +
                p1_genes[crossover_point2:]
        )
        return Chromosome(genes=offspring_genes, num_variables=self.num_variables)

    def cross_seeded(self, parent1, parent2):
        seed = np.random.randint(0, 2, size=self.chromosome_size)
        offspring_genes = [p1 if s == 0 else p2 for p1, p2, s in zip(parent1.genes, parent2.genes, seed)]
        return Chromosome(genes=offspring_genes, num_variables=self.num_variables)


    def cross_uniform(self, parent1, parent2):
        offspring_genes = [
            parent1.genes[i] if np.random.rand() < self.p_uniform else parent2.genes[i]
            for i in range(self.chromosome_size)
        ]
        return Chromosome(genes=offspring_genes, num_variables=self.num_variables)

    # MUTATION
    def mutate(self, offspring: np.ndarray):
        for idx in range(offspring.shape[0]):
            if np.random.rand() < self.p_mutation:
                if self.mutation_method == 'single':
                    self.single_point_mutation(offspring)
                elif self.mutation_method == 'double':
                    self.two_point_mutation(offspring)
                elif self.mutation_method == 'boundary':
                    self.boundary_mutation(offspring)
        return offspring

    def single_point_mutation(self, offspring):
        for chromosome in offspring:
            mutation_point = np.random.randint(0, self.chromosome_size)
            chromosome.genes[mutation_point] = 1 - chromosome.genes[mutation_point]

    def two_point_mutation(self, offspring):
        for chromosome in offspring:
            point1 = np.random.randint(0, self.chromosome_size)
            point2 = np.random.randint(0, self.chromosome_size)
            if point1 > point2:
                point1, point2 = point2, point1
            for i in range(point1, point2 + 1):
                chromosome.genes[i] = 1 - chromosome.genes[i]

    def boundary_mutation(self, offspring):
        for chromosome in offspring:
            boundary_point = np.random.choice([0, self.chromosome_size - 1])
            chromosome.genes[boundary_point] = 1 - chromosome.genes[boundary_point]

    # INVERSION
    def inverse(self, offspring: np.ndarray):
        for chromosome in offspring:
            if np.random.rand() < self.p_inversion:
                start = np.random.randint(0, self.chromosome_size)
                end = np.random.randint(start, self.chromosome_size)
                chromosome.genes[start:end] = reversed(chromosome.genes[start:end])
        return offspring

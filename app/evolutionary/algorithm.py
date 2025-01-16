import os
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
from pygad import pygad

from .chromosome import RealValueChromosome


def on_generation(ga_instance):
    print(f"Generation = {ga_instance.generations_completed}, Fitness = {ga_instance.best_solution()[1]}")

class EvolutionaryAlgorithm:

    def __init__(self,
                 gene_type,
                 population_size,
                 parent_population,
                 chromosome_size,
                 generations,
                 p_mutation,
                 p_crossover,
                 selection_method,
                 mutation_method,
                 crossover_method,
                 num_variables,
                 optimization_mode,
                 p_uniform,
                 lower_bound,
                 upper_bound,
                 elite=10,
                 ):
        self.gene_type = gene_type
        self.population_size = population_size
        self.parent_population = parent_population
        self.chromosome_size = chromosome_size * num_variables
        self.generations = generations
        self.p_mutation = p_mutation
        self.p_crossover = p_crossover
        self.selection_method = selection_method
        self.mutation_method = mutation_method
        self.crossover_method = crossover_method
        self.num_variables = num_variables
        self.optimization_mode = optimization_mode
        self.p_uniform = p_uniform
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.elite = elite

    def run(self):

        ga_instance = pygad.GA(num_generations=self.generations,
                               num_parents_mating=self.parent_population,
                               sol_per_pop=self.population_size,
                               num_genes=self.chromosome_size,
                               gene_type=self.gene_type,
                               parent_selection_type=self.selection_method,
                               crossover_type=self.crossover_method,
                               mutation_type=self.mutation_method,
                               init_range_low=self.lower_bound,
                               init_range_high=self.upper_bound,
                               mutation_probability=self.p_mutation,)
        ga_instance.run()

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename_base = f"{self.selection_method}_{self.mutation_method}_{self.crossover_method}_{timestamp}"



    def arithmetic_cross(self, parent1: RealValueChromosome, parent2: RealValueChromosome):
        alpha = np.random.rand()
        beta = 1 - alpha
        offspring_genes = alpha * parent1.get_value() + beta * parent2.get_value()
        return RealValueChromosome(genes=offspring_genes)

    def linear_cross(self, parent1, parent2):
        parent1 = np.array(parent1.get_value())
        parent2 = np.array(parent2.get_value())

        Z = 0.5 * parent1 + 0.5 * parent2
        V = 1.5 * parent1 - 0.5 * parent2
        W = -0.5 * parent1 + 1.5 * parent2

        function_values = [
            RealValueChromosome(genes=Z),
            RealValueChromosome(genes=V),
            RealValueChromosome(genes=W),
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

        return RealValueChromosome(genes=offspring)

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

        return RealValueChromosome(genes=offspring)

    def averaging_cross(self, parent1, parent2):
        parent1 = np.array(parent1.get_value())
        parent2 = np.array(parent2.get_value())

        offspring = [(parent1[i] + parent2[i]) / 2.0 for i in range(len(parent1))]

        return RealValueChromosome(genes=offspring)

    # MUTATION
    def mutate(self, offspring: np.ndarray):
        for idx in range(offspring.shape[0]):
            if np.random.rand() < self.p_mutation:
                if self.mutation_method == 'uniform':
                    self.uniform_mutation(offspring)
                elif self.mutation_method == 'gaussian':
                    self.gaussian_mutation(offspring)
        return offspring

    def uniform_mutation(offspring, ga_instance):
        for chromosome in offspring:
            for gene_index in range(len(chromosome.genes)):
                if np.random.rand() < self.p_mutation:
                    chromosome.genes[gene_index] = np.random.uniform(self.lower_bound, self.upper_bound)
        return offspring

    def gaussian_mutation(offspring, ga_instance):
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

import numpy as np

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
        # TODO plotting

        for generation in range(self.generations):
            elite_individuals = self.select_elite()
            parents = self.select_parents()
            offspring = self.crossover(parents)
            offspring = self.mutate(offspring)
            offspring = self.inverse(offspring)
            self.population = self.update_population(elite_individuals, offspring)

        # TODO rest...

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
        if self.selection_method == 'roulette':
            fitness_scores = np.array(
                [self.function.fit(chromosome.get_value(self.lower_bound, self.upper_bound)) for chromosome in
                 self.population])

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
                competitors = np.random.choice(len(self.population), size=3)
                winner = competitors[np.argmax([self.function.fit(self.population[i]) for i in competitors])]
                selected_parents.append(self.population[winner])
            return np.array(selected_parents)

        elif self.selection_method == 'best':
            return self.population[
                np.argsort(
                    [self.function.fit(chromosome.get_value(self.lower_bound, self.upper_bound)) for chromosome in
                     self.population])[
                -len(self.population) // 2:]]

    # CROSSOVER
    def crossover(self, parents):
        offspring = []
        for k in range(len(parents)):
            if np.random.rand() < self.p_crossover:
                parent1 = parents[k]
                parent2 = parents[np.random.randint(len(parents))]
                if self.crossover_method == 'single':
                    offspring.append(self.cross_single(parent1, parent2))
                elif self.crossover_method == 'double':
                    offspring.append(self.cross_double(parent1, parent2))
                elif self.crossover_method == 'uniform':
                    offspring.append(self.cross_uniform(parent1, parent2))
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

    # TODO
    def cross_seeded(self, parent1, parent2):
        pass

    def cross_uniform(self, parent1, parent2):
        offspring_genes = [
            parent1.genes[i] if np.random.rand() < self.p_uniform else parent2.genes[i]
            for i in range(self.chromosome_size)
        ]
        return Chromosome(genes=offspring_genes, num_variables=self.num_variables)

    def cross_seeded(self, parent1, parent2):
        pass

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

import numpy as np

from app.evolutionary.chromosome import Chromosome


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
                 function,
                 optimization_mode
                 ):
        self.chromosome_size = chromosome_size
        self.population = [Chromosome(chromosome_size) for _ in range(population_size)]
        self.generations = generations
        self.p_mutation = p_mutation
        self.p_crossover = p_crossover
        self.selection_method = selection_method
        self.mutation_method = mutation_method
        self.crossover_method = crossover_method
        self.function = function
        self.optimization_mode = optimization_mode

    def run(self):
        # TODO plotting

        for generation in range(self.generations):
            parents = self.select_parents()
            offspring = self.crossover(parents)
            population = self.mutate(offspring)
        # TODO rest...

    # SELECTION
    def select_parents(self):
        if self.selection_method == 'roulette':
            fitness_scores = np.array([self.function.fit(chromosome) for chromosome in self.population])

            if self.optimization_mode == 'max':
                probabilities = fitness_scores / np.sum(fitness_scores)
            else:  # TODO add solution for values < 0
                fitness_scores = np.where(fitness_scores == 0, np.float64(1e-10), fitness_scores)
                inverse_scores = 1 / fitness_scores
                probabilities = inverse_scores / np.sum(inverse_scores)

            return self.population[np.random.choice(len(self.population), size=len(self.population), p=probabilities)]

        elif self.selection_method == 'tournament':
            selected_parents = []
            for _ in range(len(self.population)):
                competitors = np.random.choice(len(self.population), size=3)
                winner = competitors[np.argmax([self.function.fit(self.population[i]) for i in competitors])]
                selected_parents.append(self.population[winner])
            return np.array(selected_parents)

        elif self.selection_method == 'best':
            return self.population[
                np.argsort([self.function.fit(chromosome.getValue()) for chromosome in self.population])[
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
        return offspring

    def cross_single(self, parent1: Chromosome, parent2: Chromosome):
        crossover_point = np.random.randint(1, self.function.num_vars)
        p1_genes = parent1.genes
        p2_genes = parent2.genes
        offspring_genes = p1_genes[:crossover_point] + p2_genes[crossover_point:]
        return Chromosome(genes=offspring_genes)

    def cross_double(self, parent1, parent2):
        crossover_point1 = np.random.randint(1, self.function.num_vars)
        crossover_point2 = np.random.randint(1, self.function.num_vars)
        if crossover_point1 > crossover_point2:
            crossover_point1, crossover_point2 = crossover_point2, crossover_point1

        p1_genes = parent1.genes
        p2_genes = parent2.genes

        offspring_genes = (
                p1_genes[:crossover_point1] +
                p2_genes[crossover_point1:crossover_point2] +
                p1_genes[crossover_point2:]
        )
        return Chromosome(offspring_genes)
    
    #TODO
    def cross_seeded(self, parent1, parent2):
        pass

    #TODO
    def cross_uniform(self, parent1, parent2):
        pass

    # MUTATION
    def mutate(self, offspring):
        for idx in range(offspring.shape[0]):
            if np.random.rand() < self.p_mutation:
                if self.mutation_method == 'single':
                    self.single_point_mutation(offspring)
                elif self.mutation_method == 'double':
                    self.two_point_mutation(offspring)
                elif self.mutation_method == 'boundary':
                    self.boundary_mutation(offspring)
        return offspring

    #TODO
    def single_point_mutation(self, offspring):
        pass

    #TODO
    def two_point_mutation(self, offspring):
        pass

        #TODO
    def boundary_mutation(self, offspring):
        pass

    # TODO INVERSION

    # TODO ELITE

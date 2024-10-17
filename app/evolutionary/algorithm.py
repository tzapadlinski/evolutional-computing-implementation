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
        #TODO plotting

        for generation in range(self.generations):
            parents = self.select_parents()
            offspring = self.crossover(parents)
            population = self.mutate(offspring)
        #TODO rest...


    def select_parents(self):
        if self.selection_method == 'roulette':
            fitness_scores = np.array([self.function.fit(chromosome) for chromosome in self.population])

            if self.optimization_mode == 'max':
                probabilities = fitness_scores / np.sum(fitness_scores)
            else:       #TODO add solution for values < 0
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
            return self.population[np.argsort([self.function.fit(chromosome.getValue()) for chromosome in self.population])[-len(self.population) // 2:]]

    def crossover(self, parents):
        offspring = np.empty(parents.shape)
        for k in range(len(parents)):
            if np.random.rand() < self.p_crossover:
                parent1 = parents[k]
                parent2 = parents[np.random.randint(len(parents))]
                if self.crossover_method == 'single':
                    crossover_point = np.random.randint(1, self.function.num_vars)
                    offspring[k, :crossover_point] = parent1[:crossover_point]
                    offspring[k, crossover_point:] = parent2[crossover_point:]
                elif self.crossover_method == 'double':
                    crossover_point1 = np.random.randint(1, self.function.num_vars)
                    crossover_point2 = np.random.randint(1, self.function.num_vars)
                    if crossover_point1 > crossover_point2:
                        crossover_point1, crossover_point2 = crossover_point2, crossover_point1
                    offspring[k, :crossover_point1] = parent1[:crossover_point1]
                    offspring[k, crossover_point1:crossover_point2] = parent2[crossover_point1:crossover_point2]
                    offspring[k, crossover_point2:] = parent1[crossover_point2:]
                elif self.crossover_method == 'uniform':
                    offspring[k] = np.where(np.random.rand(self.function.num_vars) < 0.5, parent1, parent2)
            else:
                offspring[k] = parents[k]
        return offspring

    def mutate(self, offspring):
        for idx in range(offspring.shape[0]):
            if np.random.rand() < self.p_mutation:
                if self.mutation_method == 'point':
                    mutation_point = np.random.randint(0, self.function.num_vars)
                    offspring[idx, mutation_point] = 1 - offspring[idx, mutation_point]
                elif self.mutation_method == 'bit':
                    mutation_bits = np.random.rand(self.function.num_vars) < self.p_mutation
                    offspring[idx] = np.where(mutation_bits, 1 - offspring[idx], offspring[idx])
                elif self.mutation_method == 'inverse':
                    if np.random.rand() < 0.5:
                        offspring[idx] = 1 - offspring[idx]
        return offspring

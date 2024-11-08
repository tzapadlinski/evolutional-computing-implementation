import unittest
from app.evolutionary.chromosome import Chromosome
from app.evolutionary.algorithm import EvolutionaryAlgorithm
from app.evolutionary.function import Function


class TestEvolutionaryAlgorithm(unittest.TestCase):

    def setUp(self):
        self.population_size = 10
        self.chromosome_size = 24
        self.generations = 50
        self.p_mutation = 0.1
        self.p_crossover = 0.7
        self.selection_method = 'roulette'
        self.mutation_method = 'single'
        self.crossover_method = 'single'
        self.function = Function(2, 'Griewank')
        self.optimization_mode = 'max'
        self.uniform_prob = 0.5
        self.lower_bound = -10
        self.upper_bound = 10
        self.num_variables = 2
        self.algorithm = EvolutionaryAlgorithm(
            self.population_size,
            self.chromosome_size,
            self.generations,
            self.p_mutation,
            self.p_crossover,
            self.selection_method,
            self.mutation_method,
            self.crossover_method,
            self.function,
            self.num_variables,
            self.optimization_mode,
            self.uniform_prob,
            self.lower_bound,
            self.upper_bound,
            0.04,
            0.1
        )

    def test_select_parents(self):
        parents = self.algorithm.select_parents()
        self.assertEqual(len(parents), self.population_size)
        self.assertIsInstance(parents[0], Chromosome)

    def test_crossover(self):
        parents = self.algorithm.select_parents()
        offspring = self.algorithm.crossover(parents)
        self.assertTrue(len(offspring) > 0)
        self.assertIsInstance(offspring[0], Chromosome)

    def test_cross_single(self):
        parent1 = Chromosome(genes=[0, 1] * (self.chromosome_size * self.num_variables // 2), num_variables=self.num_variables)
        parent2 = Chromosome(genes=[1, 0] * (self.chromosome_size * self.num_variables // 2), num_variables=self.num_variables)
        offspring = self.algorithm.cross_single(parent1, parent2)
        self.assertIsInstance(offspring, Chromosome)
        self.assertEqual(len(offspring.genes), self.chromosome_size * self.num_variables)

    def test_cross_double(self):
        parent1 = Chromosome(genes=[0, 1] * (self.chromosome_size * self.num_variables // 2), num_variables=self.num_variables)
        parent2 = Chromosome(genes=[1, 0] * (self.chromosome_size * self.num_variables // 2), num_variables=self.num_variables)
        offspring = self.algorithm.cross_double(parent1, parent2)
        self.assertIsInstance(offspring, Chromosome)
        self.assertEqual(len(offspring.genes), self.chromosome_size * self.num_variables)

    def test_cross_uniform(self):
        parent1 = Chromosome(genes=[0, 1] * (self.chromosome_size * self.num_variables // 2), num_variables=self.num_variables)
        parent2 = Chromosome(genes=[1, 0] * (self.chromosome_size * self.num_variables // 2), num_variables=self.num_variables)
        offspring = self.algorithm.cross_uniform(parent1, parent2)
        self.assertIsInstance(offspring, Chromosome)
        self.assertEqual(len(offspring.genes), self.chromosome_size * self.num_variables)

if __name__ == '__main__':
    unittest.main()
    print("Tests OK!")
import random


class Chromosome:

    def __init__(self, chromosome_size):
        self.genes = [random.choice([0, 1]) for _ in range(chromosome_size)]

    ##TODO implement
    def getValue(self):
        pass

    def cross_single(self, parent1, parent2):
        pass

    def cross_multi(self, parent1, parent2):
        pass

    def cross_uniform(self, parent1, parent2):
        pass
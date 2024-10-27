import random


class Chromosome:

    def __init__(self,chromosome_size=None, genes=None):
        if genes is not None:
            self.genes = genes
        else:
            self.genes = [random.choice([0, 1]) for _ in range(chromosome_size)]

    ##TODO implement
    def getValue(self):
        pass

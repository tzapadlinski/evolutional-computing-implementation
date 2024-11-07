import numpy as np

class Chromosome:

    def __init__(self, chromosome_size=None, genes=None):
        if genes is not None:
            self.genes = genes
        else:
            self.genes = [np.random.randint(2) for _ in range(chromosome_size)]

    def get_value(self, lower_bound, upper_bound):
        n = len(self.genes)
        decimal_value = sum(gene * (2 ** i) for i, gene in enumerate(reversed(self.genes)))
        x = lower_bound + decimal_value * (upper_bound - lower_bound) / (2 ** n - 1)
        return x

# # Test
# a = -10
# b = 10
# bit_array = [0,0,0,0,1,1,0,0,1,0,0,1,0,0,1,0,0,1,0,1,1,0,1,0]
# chromosome = Chromosome(genes=bit_array)
# value = chromosome.get_value(a, b)
# print(f"Chromosome: {chromosome.genes}, Value: {value}")
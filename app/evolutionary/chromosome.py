import numpy as np


class Chromosome:

    def __init__(self, num_vars: int = None, genes: [] = None, lower_bound: int = None, upper_bound: int = None):
        if genes is not None:
            self.genes = genes
        else:
            self.genes = np.random.randint(low=lower_bound, high=upper_bound, size=num_vars)

    def get_value(self):
        return self.genes

# # Test
# a = -10
# b = 10
# bit_array = [0,0,0,0,1,1,0,0,1,0,0,1,0,0,1,0,0,1,0,1,1,0,1,0]
# chromosome = Chromosome(genes=bit_array)
# value = chromosome.get_value(a, b)
# print(f"Chromosome: {chromosome.genes}, Value: {value}")

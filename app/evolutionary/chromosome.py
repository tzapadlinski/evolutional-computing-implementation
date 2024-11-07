import numpy as np

class Chromosome:

    def __init__(self, chromosome_size=None, num_variables=1, genes=None):
        if genes is not None:
            self.genes = genes
        else:
            self.genes = [np.random.randint(2) for _ in range(chromosome_size)]
        self.num_variables = num_variables

    def get_value(self, lower_bound, upper_bound):
        n = len(self.genes) // self.num_variables
        values = []

        for i in range(self.num_variables):
            genes_part = self.genes[i * n:(i + 1) * n]
            decimal_value = sum(gene * (2 ** j) for j, gene in enumerate(reversed(genes_part)))
            value = lower_bound + decimal_value * (upper_bound - lower_bound) / (2 ** n - 1)
            values.append(value)

        return values

# # Test
# a = -10
# b = 10
# bit_array = [0,0,0,0,1,1,0,0,1,0,0,1,0,0,1,0,0,1,0,1,1,0,1,0]
# chromosome = Chromosome(genes=bit_array)
# value = chromosome.get_value(a, b)
# print(f"Chromosome: {chromosome.genes}, Value: {value}")
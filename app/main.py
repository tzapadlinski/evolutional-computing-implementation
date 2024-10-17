population_size = 100
num_vars = 5
generations = 20
p_mutation = 0.1
p_crossover = 0.8
selection_method = 'roulette'
mutation_method = 'point'
crossover_method = 'single'


best_solution, best_fitness = genetic_algorithm(population_size, num_vars, generations, p_mutation)

print("lepsze rozwiązanie:", best_solution)
print("Najlepsza wartość funkcji:", best_fitness)
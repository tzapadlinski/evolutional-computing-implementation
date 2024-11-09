import argparse
import os
import time

import numpy as np
from matplotlib import pyplot as plt

from app.evolutionary.algorithm import EvolutionaryAlgorithm
from app.evolutionary.function import Function

def main():
    parser = argparse.ArgumentParser(description="Run the Genetic Algorithm with specified parameters.")
    parser.add_argument("--population_size", type=int, default=10, help="Size of the population")
    parser.add_argument("--chromosome_size", type=int, default=24, help="Size of the chromosome")
    parser.add_argument("--num_variables", type=int, default=2, help="Number of variables")
    parser.add_argument("--num_epochs", type=int, default=50, help="Number of generations")
    parser.add_argument("--begin_range", type=float, default=-10.0, help="Lower bound of the variable range")
    parser.add_argument("--end_range", type=float, default=10.0, help="Upper bound of the variable range")
    parser.add_argument("--elite_strategy_amount", type=float, default=0.04, help="Percentage of elite strategy")
    parser.add_argument("--cross_prob", type=float, default=0.7, help="Crossover probability")
    parser.add_argument("--mutation_prob", type=float, default=0.1, help="Mutation probability")
    parser.add_argument("--inversion_prob", type=float, default=0.1, help="Inversion probability")
    parser.add_argument("--uniform_crossover", type=float, default=0.5, help="Uniform crossover probability")
    parser.add_argument("--selection_method", type=str, default="roulette", help="Selection method")
    parser.add_argument("--cross_method", type=str, default="single", help="Crossover method")
    parser.add_argument("--mutation_method", type=str, default="single", help="Mutation method")
    parser.add_argument("--function", type=str, default="Griewank", help="Function to optimize")
    parser.add_argument("--optimization_mode", type=str, default="max", help="Optimization mode (max or min)")

    args = parser.parse_args()

    all_best_fitness = []
    all_mean_fitness = []
    all_std_fitness = []
    all_execution_time = []

    for _ in range(10):
        alg = EvolutionaryAlgorithm(
            population_size=args.population_size,
            chromosome_size=args.chromosome_size,
            generations=args.num_epochs,
            p_mutation=args.mutation_prob,
            p_crossover=args.cross_prob,
            selection_method=args.selection_method,
            mutation_method=args.mutation_method,
            crossover_method=args.cross_method,
            function=Function(args.num_variables, args.function),
            num_variables=args.num_variables,
            optimization_mode=args.optimization_mode,
            p_uniform=args.uniform_crossover,
            lower_bound=args.begin_range,
            upper_bound=args.end_range,
            p_inversion=args.inversion_prob,
            elite_percentage=args.elite_strategy_amount,
        )
        best_fitness, mean_fitness, std_fitness, all_fitness, execution_time = alg.run()
        all_best_fitness.append(best_fitness)
        all_mean_fitness.append(mean_fitness)
        all_std_fitness.append(std_fitness)
        all_execution_time.append(execution_time)

    avg_best_fitness = np.mean(all_best_fitness, axis=0)
    avg_mean_fitness = np.mean(all_mean_fitness, axis=0)
    avg_std_fitness = np.mean(all_std_fitness, axis=0)
    avg_execution_time = np.mean(all_execution_time, axis=0)

    worst_run_index = np.argmax([fitness[-1] for fitness in all_best_fitness])
    best_run_index = np.argmin([fitness[-1] for fitness in all_best_fitness])

    best_run_best_fitness = np.array(all_best_fitness[best_run_index])
    best_run_mean_fitness = np.array(all_mean_fitness[best_run_index])
    best_run_std_fitness = np.array(all_std_fitness[best_run_index])

    worst_run_best_fitness = np.array(all_best_fitness[worst_run_index])
    worst_run_mean_fitness = np.array(all_mean_fitness[worst_run_index])
    worst_run_std_fitness = np.array(all_std_fitness[worst_run_index])

    print(f"Selection Method: {args.selection_method}")
    print(f"Cross Method: {args.cross_method}")
    print(f"Mutation Method: {args.mutation_method}")
    print(f"Function: {args.function}")
    print(f"Optimization Mode: {args.optimization_mode}")

    print(f"Average execution Time: {avg_execution_time:.4f} seconds")

    # AVG RUN PLOT
    plt.figure(figsize=(12, 6))
    plt.subplot(2, 1, 1)
    plt.plot(avg_best_fitness, label='Average Best Fitness')
    plt.xlabel('Generation')
    plt.ylabel('Best Fitness')
    plt.title('Average Best Fitness per Generation')
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(avg_mean_fitness, label='Average Mean Fitness')
    plt.fill_between(range(args.num_epochs),
                     avg_mean_fitness - avg_std_fitness,
                     avg_mean_fitness + avg_std_fitness,
                     color='b', alpha=0.2, label='Std Dev')
    plt.xlabel('Generation')
    plt.ylabel('Fitness')
    plt.title('Average Mean and Standard Deviation of Fitness per Generation')
    plt.legend()

    plt.tight_layout()
    plot_path_avg = os.path.join(os.getcwd(), "avg_plot.png")
    plt.savefig(plot_path_avg)

    # BEST RUN PLOT
    plt.figure(figsize=(12, 6))
    plt.subplot(2, 1, 1)
    plt.plot(best_run_best_fitness, label='Best Run Best Fitness')
    plt.xlabel('Generation')
    plt.ylabel('Best Fitness')
    plt.title('Best Run Best Fitness per Generation')
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(best_run_mean_fitness, label='Best Run Mean Fitness')
    plt.fill_between(range(args.num_epochs),
                     best_run_mean_fitness - best_run_std_fitness,
                     best_run_mean_fitness + best_run_std_fitness,
                     color='b', alpha=0.2, label='Std Dev')
    plt.xlabel('Generation')
    plt.ylabel('Fitness')
    plt.title('Best Run Mean and Standard Deviation of Fitness per Generation')
    plt.legend()

    plt.tight_layout()
    plot_path_best = os.path.join(os.getcwd(), "best_plot.png")
    plt.savefig(plot_path_best)

    # WORST RUN PLOT
    plt.figure(figsize=(12, 6))
    plt.subplot(2, 1, 1)
    plt.plot(worst_run_best_fitness, label='Worst Run Best Fitness')
    plt.xlabel('Generation')
    plt.ylabel('Best Fitness')
    plt.title('Worst Run Best Fitness per Generation')
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(worst_run_mean_fitness, label='Worst Run Mean Fitness')
    plt.fill_between(range(args.num_epochs),
                     worst_run_mean_fitness - worst_run_std_fitness,
                     worst_run_mean_fitness + worst_run_std_fitness,
                     color='b', alpha=0.2, label='Std Dev')
    plt.xlabel('Generation')
    plt.ylabel('Fitness')
    plt.title('Worst Run Mean and Standard Deviation of Fitness per Generation')
    plt.legend()

    plt.tight_layout()
    plot_path_worst = os.path.join(os.getcwd(), "worst_plot.png")
    plt.savefig(plot_path_worst)
if __name__ == "__main__":
    main()

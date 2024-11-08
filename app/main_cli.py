import argparse
import time
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

    start_time = time.time()

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

    result = alg.run()

    execution_time = time.time() - start_time

    print(f"Selection Method: {args.selection_method}")
    print(f"Cross Method: {args.cross_method}")
    print(f"Mutation Method: {args.mutation_method}")
    print(f"Function: {args.function}")
    print(f"Optimization Mode: {args.optimization_mode}")
    print(f"Execution Time: {execution_time:.4f} seconds")
    print(f"Result: {result}")

if __name__ == "__main__":
    main()
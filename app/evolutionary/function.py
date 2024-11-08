import benchmark_functions as bf
import opfunu

class Function:
    def __init__(self, num_vars, function_name):
        self.function_name = function_name

        if function_name == 'Griewank':
            self.func = bf.Griewank(n_dimensions=num_vars)
        elif function_name == 'Cigar':
            funcs = opfunu.get_functions_by_classname(function_name)
            self.func = funcs[0](ndim=num_vars)

    def fit(self, x):
        if self.function_name == 'Griewank':
            result = self.func(x)
        elif self.function_name == 'Cigar':
            result = self.func.evaluate(x)
        else:
            raise ValueError(f"Function {self.function_name} not implemented")
        return result
    
    @staticmethod
    def get_available_functions():
        return ['Griewank', 'Cigar']
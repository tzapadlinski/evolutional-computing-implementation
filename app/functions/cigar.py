import numpy as np
import matplotlib.pyplot as plt
import opfunu

def get_cigar_optimum(n_dimensions):
          cigar_function_class = opfunu.get_functions_by_classname('Cigar')

          if not cigar_function_class:
                    raise ValueError("Cigar Function not found in opfunu package")

          cigar_function = cigar_function_class[0](ndim=n_dimensions)

          optimum_arguments = np.zeros(n_dimensions)
          optimum_value = cigar_function.evaluate(optimum_arguments)

          return optimum_value, optimum_arguments

optimum_3 = get_cigar_optimum(3)
optimum_10 = get_cigar_optimum(10)
optimum_20 = get_cigar_optimum(20)
optimum_30 = get_cigar_optimum(30)

# Print the results
print(f"3 Dimensions: Optimum Value = {optimum_3[0]}, Arguments = {optimum_3[1]}")
print(f"10 Dimensions: Optimum Value = {optimum_10[0]}, Arguments = {optimum_10[1]}")
print(f"20 Dimensions: Optimum Value = {optimum_20[0]}, Arguments = {optimum_20[1]}")
print(f"30 Dimensions: Optimum Value = {optimum_30[0]}, Arguments = {optimum_30[1]}")

def cigar_2d_projection(n_dimensions):
          cigar_function_class = opfunu.get_functions_by_classname('Cigar')
          cigar_function = cigar_function_class[0](ndim=n_dimensions)

          def cigar_fixed(x, y):
                    variables = np.zeros(n_dimensions)
                    variables[0] = x
                    variables[1] = y
                    return cigar_function.evaluate(variables)

          return cigar_fixed

x = np.linspace(-600.0, 600.0, 400)
y = np.linspace(-600.0, 600.0, 400)
X, Y = np.meshgrid(x, y)

Z_10 = np.vectorize(cigar_2d_projection(10))(X, Y)
Z_20 = np.vectorize(cigar_2d_projection(20))(X, Y)
Z_30 = np.vectorize(cigar_2d_projection(30))(X, Y)

fig, axes = plt.subplots(1, 3, figsize=(21, 7), subplot_kw={'projection': '3d'})
fig.suptitle('Cigar Function (2D Projections)')

axes[0].plot_surface(X, Y, Z_10, cmap='viridis')
axes[0].set_title('10 zmiennych')
axes[0].set_xlabel('x1')
axes[0].set_ylabel('x2')
axes[0].set_zlabel('f(x)')

axes[1].plot_surface(X, Y, Z_20, cmap='viridis')
axes[1].set_title('20 zmiennych')
axes[1].set_xlabel('x1')
axes[1].set_ylabel('x2')
axes[1].set_zlabel('f(x)')

axes[2].plot_surface(X, Y, Z_30, cmap='viridis')
axes[2].set_title('30 zmiennych')
axes[2].set_xlabel('x1')
axes[2].set_ylabel('x2')
axes[2].set_zlabel('f(x)')

plt.show()



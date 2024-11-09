import numpy as np
import matplotlib.pyplot as plt
import benchmark_functions as bf

def griewank_2d_projection(n_dimensions):
          def griewank_fixed(x, y):
                    variables = [x, y] + [0] * (n_dimensions - 2)
                    return bf.Griewank(n_dimensions=n_dimensions)(variables)
          return griewank_fixed

x = np.linspace(-600.0, 600.0, 400)
y = np.linspace(-600.0, 600.0, 400)
X, Y = np.meshgrid(x, y)

Z_10 = np.vectorize(griewank_2d_projection(10))(X, Y)
Z_20 = np.vectorize(griewank_2d_projection(20))(X, Y)
Z_30 = np.vectorize(griewank_2d_projection(30))(X, Y)

fig, axes = plt.subplots(1, 3, figsize=(21, 7), subplot_kw={'projection': '3d'})
fig.suptitle('Griewank Function (2D Projections)')

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

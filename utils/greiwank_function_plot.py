import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def griewank_function(X, Y):
    """
    Compute the Griewank function for given X and Y.
    Griewank function is defined as:
    f(x) = 1 + (1/4000) * sum(x_i^2) - prod(cos(x_i/sqrt(i)))
    """
    sum_term = (X**2 + Y**2) / 40000
    cos_term = np.cos(X/np.sqrt(1)) * np.cos(Y/np.sqrt(2))
    return 1 + sum_term - cos_term

# Generate grid of points
x = np.linspace(-5, 5, 1000)
y = np.linspace(-5, 5, 1000)
X, Y = np.meshgrid(x, y)

# Compute the Griewank function values
Z = griewank_function(X, Y)

# Create 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot the surface
surf = ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none')

# Add labels and title
ax.set_xlabel('X-axis')
ax.set_ylabel('Y-axis')
ax.set_zlabel('Z-axis')
ax.set_title('Griewank Function')

# Add color bar for reference
fig.colorbar(surf)

# Show plot
plt.show()

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def michalewicz_function(X, Y, m=10):
    Z = np.sin(X) * (np.sin(X**2 / np.pi))**(2 * m) + \
        np.sin(Y) * (np.sin(2 * Y**2 / np.pi))**(2 * m)
    return -Z

# Generate a grid of points
x = np.linspace(0, np.pi, 100)
y = np.linspace(0, np.pi, 100)
X, Y = np.meshgrid(x, y)

# Compute the Michalewicz function on the grid
Z = michalewicz_function(X, Y)

# Plotting the surface
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')
surface = ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none')

# Adding labels and title
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Michalewicz Function')

# Add a color bar
fig.colorbar(surface, ax=ax, shrink=0.5, aspect=5)

plt.show()
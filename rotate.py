import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Define the length of the stick
L = 1.0

# Define the angles of rotation around x and y axes (in degrees)
alpha_deg = 30  # Rotation around x-axis
beta_deg = 45   # Rotation around y-axis

# Convert degrees to radians
alpha = np.deg2rad(alpha_deg)
beta = np.deg2rad(beta_deg)

# Define the initial points of the stick
points = np.array([[0, 0, 0],
                   [0, 0, L]])

# Define rotation matrices for x and y axes
Rx = np.array([[1, 0, 0],
               [0, np.cos(alpha), -np.sin(alpha)],
               [0, np.sin(alpha), np.cos(alpha)]])

Ry = np.array([[np.cos(beta), 0, np.sin(beta)],
               [0, 1, 0],
               [-np.sin(beta), 0, np.cos(beta)]])

# Rotate the points
points_rotated = Rx @ Ry @ points.T

# Plotting
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Original stick
ax.plot(points[:, 0], points[:, 1], points[:, 2], marker='o', color='b')

# Rotated stick
ax.plot(points_rotated[0, :], points_rotated[1, :], points_rotated[2, :], marker='o', color='r')

# Setting the same scale for all axes
ax.set_xlim([-1, 1])
ax.set_ylim([-1, 1])
ax.set_zlim([0, 1])

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

plt.title(f'Rotation around X by {alpha_deg}° and Y by {beta_deg}°')

plt.show()

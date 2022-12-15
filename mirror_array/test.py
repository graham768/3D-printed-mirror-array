import matplotlib.pyplot as plt
import numpy as np
import hexy as hx
from mirror_array.grid import *

coords = hx.get_spiral(np.array((0, 0, 0)), 1, 3)
x, y = hx.cube_to_pixel(coords, 1).T
points = np.array([y, x]).T


def plot_hex_points(points, num=len(points)):
    # Test visualize
    plt.scatter(points.T[0][0:num], points.T[1][0:num])
    plt.axis("equal")
    plt.show()
    
plot_hex_points(points)




target_pos = np.array([0, 0, 40]) # the common focal point

θ = np.deg2rad(0) # This is the angle with respect to the XY plane, normal to the overall structure
ϕ = np.deg2rad(0)

grid_hex_radius = 2
mini_hex_radius = 10

hex_centers = get_hex_grid_centers(grid_hex_radius, mini_hex_radius)
mirror_positions = [mirror_pos_from_xy_base(center, center_height=20, distance_scaling=.2) for center in hex_centers]
mirror_normals = [compute_mirror_normal(mirror_pos, target_pos, θ, ϕ) for mirror_pos in mirror_positions]

hex_prism_grid = make_hex_prism_grid(mirror_positions, mirror_normals, mini_hex_radius * .9)

# The extra bit at the bottom of the grids
base_positions = [mirror_pos_from_xy_base(center, center_height=5, distance_scaling=0) for center in hex_centers]
base_normals = np.tile(np.array([0,0,1]), (len(hex_centers),1))
hex_prism_bases = make_hex_prism_grid(base_positions, base_normals, mini_hex_radius * 1.1, add_mirror_aligners=False)

hex_prism_grid_combined = fuse_models([hex_prism_grid, hex_prism_bases])

hex_prism_grid_combined.save("stl/hex_prism_grid_single_focus.stl")
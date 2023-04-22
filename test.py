from mirror_array.helpers import *
from mirror_array.coords import get_coords
from mirror_array.grid import build_mirror_array

heart_initial = [[599.4664306640625, 119.61083984375]]
heart_coords = get_coords(heart_initial) # 42.02in x 38.43in

mirror_positions, mirror_normals, target_positions = build_mirror_array(heart_coords, use_ground_target=True, filename="mirror_array_heart.stl")


plot_colored_projection(target_positions)
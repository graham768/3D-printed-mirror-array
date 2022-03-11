from mirror_array.helpers import *
from mirror_array.coords import get_coords
from mirror_array.grid import build_mirror_array, divide_hex_grid_flower

regards_initial = [[40,9], [22,6], [7,15], [10,29], [22,35], [34,42], [41,55], [32,68], [19,69], [4,66], [97,46], [94,30], [80,23], [65,31], [79,69], [94,69], [78,47], [61,47], [65,62], [118,66], [118,47], [118,26], [130,28], [143,25], [154,32], [154,49], [153,68], [200,25], [183,27], [175,43], [180,62], [198,70], [212,5], [212,21], [212,38], [212,57], [220,70], [270,65], [270,45], [270,25], [270,8], [282,18], [290,33], [299,50], [312,66], [314,46], [313,27], [314,10], [339,27], [339,45], [341,61], [354,70], [370,62], [373,27], [374,43], [379,70], [423,26], [406,27], [395,43], [400,62], [418,70], [434,7], [434,23], [434,41], [434,58], [440,70], [456,47], [476,47], [495,47], [490,30], [475,23], [460,30], [459,61], [472,69], [490,67], [538,25], [523,23], [512,31], [519,44], [532,49], [541,59], [531,69], [511,68], [118,38], [119,57], [154,58], [465,47], [486,47], [88,47], [69,47], [39,62]]
regards_coords = get_coords(regards_initial, width_in=42, mirror_y=True)

# To generate the final 3D model without dividing it into subvolumes:
positions, normals, target_positions = build_mirror_array(regards_coords, use_ground_target=True, filename="regards.stl")

plot_colored_projection(target_positions)
plot_colored_grid(positions, divide_hex_grid_flower(positions, 14.85))
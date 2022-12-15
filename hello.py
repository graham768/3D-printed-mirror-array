
from mirror_array.helpers import *
from mirror_array.coords import *
from mirror_array.grid import *

hello =[[9.67,13.2],[9.67,13.2],[9.67,24.9],[9.67,36.6],[9.67,48.29],[9.67,59.99],[9.67,51.62],[9.67,39.93],[18.27,36.83],[29.97,36.83],[41.67,36.83],[53.37,36.83],[55.27,27.03],[55.27,15.34],[55.27,22.76],[55.27,34.46],[55.27,46.16],[55.27,57.86],[55.27,61.18],[105.72,59.76],[97.35,59.76],[85.66,59.76],[73.96,59.76],[68.38,51.38],[68.38,39.68],[68.63,28.08],[78.61,24.08],[90.3,24.08],[101.34,25.65],[103.77,36.35],[97.5,41.78],[85.8,41.78],[74.1,41.78],[68.15,41.78],[116.2,9.23],[116.2,14.97],[116.2,26.67],[116.2,38.37],[116.2,50.06],[120.91,59.53],[127.97,59.53],[135.67,9.23],[135.67,13.86],[135.67,25.56],[135.67,37.25],[135.67,48.95],[139.45,59.1],[147.45,59.53],[157.02,29.03],[157.02,32.55],[157.02,44.25],[157.48,55.76],[167.55,59.53],[179.25,59.53],[190.21,57.76],[192.2,46.88],[192.2,35.18],[189.54,24.53],[178.14,23.85],[166.44,23.85],[157.02,29.02]]

hello_coords = get_coords(hello, width_in=20, mirror_x=False, mirror_y=True, debug=True)

# # To generate the final 3D model and use the partitioning scheme:
# positions, normals, target_positions = build_mirror_array(hello_coords, use_ground_target=True, filename="mirror_array.stl", divider_function=divide_hex_grid_flower)

# plot_colored_projection(target_positions)
# plot_colored_grid(positions, divide_hex_grid_flower(positions))
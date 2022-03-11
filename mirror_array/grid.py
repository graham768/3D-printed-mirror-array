import numpy as np
import hexy as hx
from prism import *
from helpers import *

def get_hex_grid_centers(num_hexes_radius, mini_hex_radius):
    """Creates a large hexagonal grid of mini-hexagons. Returns the 2D xy centerpoint for each hexagon.
    Args:
        num_hexes_radius: the radius of the hexagonal grid. A single point is R=1, seven points is R=2, 19 points is R=3, etc.
        mini_hex_radius: the radius of one of the hexagons in the grid (the radius of the hexagonal prisms)
    """
    coords = hx.get_spiral(np.array((0, 0, 0)), 1, num_hexes_radius) # The center is in hexagonal, "fake", xyz cubic coordinates
    x, y = hx.cube_to_pixel(coords, mini_hex_radius).T
    centers = np.array([y, x]).T
    return centers

def mirror_pos_from_xy_base(center, center_height=20, distance_scaling=0.0, x_offset=0.0, y_offset=0.0):
    """Converts xy coordinates of a hexagonal base to the xyz coordinates of mirror centers
    Args:
        center: the center xy coordinate of the base
        center_height: the height of the centermost prism
        distance_scaling: prisms a radius r from the center will have their height adjusted by distance_scaling * r amount
        x_offset: shift all prisms by this amount on x axis
        y_offset: shift all prisms by this amount on y axis
    """
    distance_from_center = np.linalg.norm(center)
    height = center_height + distance_scaling * distance_from_center
    center_vec = np.array([center[0] + x_offset, center[1] + y_offset, height])
    return center_vec

def make_hex_prism_grid(mirror_positions, mirror_normals, mini_hex_radius, add_mirror_aligners=True, x_offset=0.0, y_offset=0.0, verbose=True, no_fuse_models=False):
    """Makes a grid of hexagonal prisms given a list of mirror positions and corresponding normal vectors
    Args:
        mirror_positions: xyz coordinates of the mirror centroids
        mirror_normals: normal vector for each mirror
        mini_hex_radius: radius of each of the hexagonal prisms in the grid
        no_fuse_models: if True, a list of hexagonal prisms meshes will be returned instead of a single fused model (useful for partitioning)
    """
    all_prisms = []
    for mirror_pos, mirror_normal in zip(mirror_positions, mirror_normals):
        hex_prism = create_hex_prism(mirror_pos, mirror_normal, mini_hex_radius, add_mirror_aligners=add_mirror_aligners, aligner_inner_radius=mini_hex_radius-1.0, aligner_thickness=1.0)
        all_prisms.append(hex_prism)
    if verbose:
        print("Created model with {} hex prisms".format(len(all_prisms)))
    if no_fuse_models:
        return all_prisms
    else:
        return fuse_models(all_prisms)




def divide_hex_grid_in_quarters(points):
    """Partitions a hexagonal grid into quarters. Returns a list of partition indices for each point."""
    points = np.array(points)
    # Divide into quarters
    partition_indices = np.zeros(len(points))
    for i, point in enumerate(points):
        x, y, z = point
        if x >= 0 and y >= 0:
            partition_indices[i] = 0
        elif x >= 0 and y < 0:
            partition_indices[i] = 1
        elif x < 0 and y >= 0:
            partition_indices[i] = 2
        else:
            partition_indices[i] = 3
    return partition_indices

def divide_hex_grid_flower(points, hex_radius=None):
    """Partitions a hexagonal grid into a flower pattern (this is what I used for the final product. Returns a list of partition indices for each point."""
    if hex_radius is None: # copied from build_mirror_array()
        mini_hex_radius = (10 * 2.5 / 2) + 1 
        hex_radius = mini_hex_radius * 1.1
    points = np.array(points)
    # Divide into quarters
    partition_indices = np.ones(len(points)) * -1
    for i, point in enumerate(points):
        x, y, z = point
        if np.sqrt(x**2 + y**2) <= 3 * (2*hex_radius + 1) * np.sqrt(3)/2:
            partition_indices[i] = 0
        else:
            θ = np.arctan2(x,y) + pi - 1e-10
            partition_indices[i] = 1 + np.floor(6 * θ / (2 * pi))
    return partition_indices

def partition_and_save_models(hex_prisms, base_prisms, partition_indices, filename = "hex_prism_grid.stl"):
    """Returns a list of fused sub-volumes which can be individually printed
    
    Args:
        hex_prisms: the hexagonal pillars of the model which hold the mirorrs
        base_prisms: the bottom hexagonal bases of each pillar which overlap to form a single base 
        partition_indices: for each hex_prism and base_prism, an index indicating which sub-volume they belong to
    """
    num_sections = len(set(partition_indices))
    for section in range(num_sections):
        base_prisms_section = [base for (i, base) in enumerate(base_prisms) if partition_indices[i]==section]
        hex_prisms_section = [hex_prism for (i, hex_prism) in enumerate(hex_prisms) if partition_indices[i]==section]
        hex_prism_grid_section_combined = fuse_models([*hex_prisms_section, *base_prisms_section])
        section_filename = filename[0:-4] + "_section{}".format(section) + filename[-4:]
        hex_prism_grid_section_combined.save("stl/" + section_filename)  





def build_mirror_array(target_coordinates, filename = "hex_prism_grid.stl", depth = FOCAL_PLANE_DEFAULT, divider_function=None, use_ground_target=False):
    """Builds an array of mirrors to focus on a list of target coordinates and saves the model (or partitioned models) as .stl file(s)
    
    Args:
        target_coordinates: the xy coordinates of the target image
        filename: the filename (or filename pattern) to save the output file(s) as
        depth: the depth of the focal plane
        divider_function: if specified, how to divide up the 3D printed model into subvolumes
        use_ground_target: if True, will project onto the ground (see create_target_positions_ground() above); otherwise will project onto a wall
        
    Returns:
        Lists of respective mirror positions, mirror normals, and corresponding target positions for each mirror
    """
    
    assert is_hexagonal_number(len(target_coordinates)), "coordinates must have hex number of targets!"
    
    # New sorting scheme:
    # 1. Group target coords into hex-number sized buckets based on the distance from the center of mass
    # 2. Sort target coords in each bucket based on angle from vertical-down
    # 3. Match each point to the corresponding mirror in that radius
    target_coords_by_bucket = []
    center_of_mass = np.mean(target_coordinates, axis=0)
    target_coords_sorted_by_radius = sort_by_predicate(target_coordinates, lambda c: np.linalg.norm(c - center_of_mass))
    
    # group target coords by radius into bins
    r_hex = get_hex_radius(len(target_coordinates))
    target_coords_grouped_by_radius = np.split(target_coords_sorted_by_radius, HEX_NUMBERS[0:r_hex])

    # computes the COUNTERCLOCKWISE angle from the bottom. necessary because mirrors are enumerate counterclockwise.
    def angle_from_bottom(coord):
        line_from_com = coord - center_of_mass
        x, y = line_from_com
        angle_from_bottom = -1 * (np.arctan2(x,y) - pi - 1e-10)
        return angle_from_bottom
    
    # sort each bin by angle from bottom and append to master list
    target_coordinates_sorted = []
    for target_coords_in_bin in target_coords_grouped_by_radius:
        target_coords_sorted_by_angle = sort_by_predicate(target_coords_in_bin, lambda c: angle_from_bottom(c))
        target_coordinates_sorted += list(target_coords_sorted_by_angle)
    target_coordinates_sorted = np.array(target_coordinates_sorted)
    
    if use_ground_target:
        target_positions_sorted = create_target_positions_ground(target_coordinates_sorted)
    else:
        target_positions_sorted = create_target_positions_wall(target_coordinates_sorted, depth=depth)

    θ = np.deg2rad(10) # This is the angle with respect to the XY plane, normal to the overall structure
    ϕ = np.deg2rad(0)
    
    v_sun = np.array([sin(ϕ), cos(ϕ) * sin(θ), cos(ϕ) * cos(θ)])

    grid_hex_radius = HEX_NUMBERS.index(len(target_positions_sorted))
    assert grid_hex_radius >= 1
    
    mini_hex_radius = (10 * 2.5 / 2) + 1 # +1 for aligner
    hex_base_radius = mini_hex_radius * 1.1

    hex_centers = get_hex_grid_centers(grid_hex_radius, hex_base_radius)

    mirror_positions = [mirror_pos_from_xy_base(center, center_height=24, distance_scaling=-.07) for center in hex_centers]

    mirror_normals = [compute_mirror_normal(mirror_pos, target_pos, θ, ϕ) for mirror_pos, target_pos in zip(mirror_positions, target_positions_sorted)]

    base_positions = [mirror_pos_from_xy_base(center, center_height=5, distance_scaling=0) for center in hex_centers]
    base_normals = np.tile(np.array([0,0,1]), (len(hex_centers),1))
    
    if divider_function is not None:
        partition_indices = divider_function(mirror_positions)
        hex_prisms = make_hex_prism_grid(mirror_positions, mirror_normals, mini_hex_radius, add_mirror_aligners=True, no_fuse_models=True)
        base_prisms = make_hex_prism_grid(base_positions, base_normals, hex_base_radius * 1.01, add_mirror_aligners=False, no_fuse_models=True)
        partition_and_save_models(hex_prisms, base_prisms, partition_indices, filename=filename)
    else:
        hex_prism_grid = make_hex_prism_grid(mirror_positions, mirror_normals, mini_hex_radius, add_mirror_aligners=True)
        base_grid = make_hex_prism_grid(base_positions, base_normals, hex_base_radius * 1.01, add_mirror_aligners=False)
        hex_prism_grid_combined = fuse_models([hex_prism_grid, base_grid])
        hex_prism_grid_combined.save("stl/" + filename)
    
    return mirror_positions, mirror_normals, target_positions_sorted
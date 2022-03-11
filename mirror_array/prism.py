import numpy as np
from numpy import sin, cos, tan, pi
import stl

def compute_mirror_normal(mirror_pos, target_pos, sun_azimuth_θ, sun_angle_ϕ = 0.0):
    """Returns a unit normal vector for the mirror to shine light on a target position"""
    
    v_target_mirror = mirror_pos - target_pos

    θ = sun_azimuth_θ
    ϕ = sun_angle_ϕ
    v_mirror_sun = np.array([sin(ϕ), cos(ϕ) * sin(θ), cos(ϕ) * cos(θ)])

    mirror_normal = normalize(v_mirror_sun) - normalize(v_target_mirror)
    return normalize(mirror_normal)


def intersection_line_plane(p0, p1, p_co, p_no, epsilon=1e-9):
    """
    Copied from https://stackoverflow.com/questions/5666222/3d-line-plane-intersection
    
    p0, p1: Define the line.
    p_co, p_no: define the plane:
        p_co Is a point on the plane (plane coordinate).
        p_no Is a normal vector defining the plane direction;
            (does not need to be normalized).

    Return a Vector or None (when the intersection can't be found).
    """
    
    def add_v3v3(v0, v1):
        return (v0[0] + v1[0], v0[1] + v1[1], v0[2] + v1[2])

    def sub_v3v3(v0, v1):
        return (v0[0] - v1[0], v0[1] - v1[1], v0[2] - v1[2])

    def dot_v3v3(v0, v1):
        return ((v0[0] * v1[0]) + (v0[1] * v1[1]) + (v0[2] * v1[2]))

    def len_squared_v3(v0):
        return dot_v3v3(v0, v0)

    def mul_v3_fl(v0, f):
        return (v0[0] * f, v0[1] * f, v0[2] * f)

    u = sub_v3v3(p1, p0)
    dot = dot_v3v3(p_no, u)

    if abs(dot) > epsilon:
        # The factor of the point between p0 -> p1 (0 - 1)
        # if 'fac' is between (0 - 1) the point intersects with the segment.
        # Otherwise:
        #  < 0.0: behind p0.
        #  > 1.0: infront of p1.
        w = sub_v3v3(p0, p_co)
        fac = -dot_v3v3(p_no, w) / dot
        u = mul_v3_fl(u, fac)
        return add_v3v3(p0, u)
    else:
        # The segment is parallel to plane.
        return None

def rad_from_deg(deg):
    return deg * np.pi / 180

def normalize(v):
    """Normalizes a vector, returning v / ||v||"""
    return v / np.linalg.norm(v)


def rotation_matrix(axis, theta):
    """
    Return the rotation matrix associated with counterclockwise rotation about
    the given axis by theta radians.
    """
    axis = np.asarray(axis)
    axis = axis / np.sqrt(np.dot(axis, axis))
    a = cos(theta / 2.0)
    b, c, d = -axis * sin(theta / 2.0)
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
    return np.array([[aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
                     [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
                     [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]])


def create_hex_prism(center_pos_vec, normal_vec, triangle_radius, add_mirror_aligners=False, aligner_inner_radius=None, aligner_thickness=0.5):
    """
    Creates a 3D model of a hexagonal prism which can hold a mirror. 
    Assumptions:
    - Base plane is at z=0
    - Triangle radius is center-to-corner and points in +/- x direction
    
    
    Args:
        center_pos_vec: the centroid of the top of the hexagonal prism
        normal_vec: the normal vector of the top face of the prism
        triangle_radius: the distance from the center of the bottom base of the prism to one of the bottom corners
        add_mirror_aligners: if True, will add protruding structures that help with aligning mirrors when gluing them on
        aligner_inner_radius: the distance from the centroid of the top face to the inner edge of the aligner (the radius of your mirror)
        aligner_thickness: the thickness of the aligner
        
    Returns:
        A stl mesh for the hexagonal prism
    """

    # Create base vertices
    base_center = [center_pos_vec[0], center_pos_vec[1], 0.0]
    base_corners = [
        base_center + (triangle_radius * (rotation_matrix([0,0,1], θ) @ np.array([1,0,0])))
        for θ in 2*pi/6 * np.arange(6)
    ]
    base_vertices = np.array([base_center, *base_corners])

    # Create top vertices 
    top_center = np.copy(center_pos_vec)
    # top_corners = [[v[0], v[1], 1000.0] for v in base_corners]
    top_corners = np.array([intersection_line_plane(v, [v[0], v[1], 1000.0], center_pos_vec, normal_vec) for v in base_corners])
    top_vertices = np.array([top_center, *top_corners])

    # Create the mesh vertices and faces
    vertices = np.array([*base_vertices, *top_vertices])
    bottom_faces = [
        [0,1,2],
        [0,2,3],
        [0,3,4],
        [0,4,5],
        [0,5,6],
        [0,6,1]
    ]
    top_faces = [
        [7,8,9],
        [7,9,10],
        [7,10,11],
        [7,11,12],
        [7,12,13],
        [7,13,8]
    ]
    side_faces = [
        [1,9,2],  [1,8,9],
        [2,10,3], [2,9,10],
        [3,11,4], [3,10,11],
        [4,12,5], [4,11,12],
        [5,13,6], [5,12,13],
        [6,1,8],  [6,13,8]
    ]
    faces = np.array([*bottom_faces, *top_faces, *side_faces])

    # Create the mesh
    hex_prism = stl.mesh.Mesh(np.zeros(faces.shape[0], dtype=stl.mesh.Mesh.dtype))
    for i, face in enumerate(faces):
        for j in range(3):
            hex_prism.vectors[i][j] = vertices[face[j],:]
            
    # Add alignment ridges to place mirrors into
    if add_mirror_aligners:
        if aligner_inner_radius is None:
            raise ValueError("Need to specify alignemnt radius!")
            
        aligner_outer_radius = aligner_inner_radius + aligner_thickness
        if aligner_outer_radius > triangle_radius:
            raise ValueError("Aligner thickness plus width too large!")
            
        # Create base vertices
        aligner_center = np.copy(center_pos_vec)
        x_plus_on_face = normalize(np.cross(np.array([0,1,0]), normal_vec))
        aligner_inner_corners = [
            aligner_center + aligner_inner_radius * (rotation_matrix(normal_vec, θ) @ x_plus_on_face)
            for θ in 2*pi/6 * np.array([-1,0,1])
        ]
        aligner_outer_corners = [
            aligner_center + aligner_outer_radius * (rotation_matrix(normal_vec, θ) @ x_plus_on_face)
            for θ in 2*pi/6 * np.array([-1,0,1])
        ]
        aligner_bottom_vertices = np.array([*aligner_inner_corners, *aligner_outer_corners])
        
        unit_normal_vec = normal_vec / np.linalg.norm(normal_vec)
        aligner_top_vertices = aligner_bottom_vertices + (aligner_thickness * unit_normal_vec)
        
        aligner_vertices = np.array([*aligner_bottom_vertices, *aligner_top_vertices])
        aligner_bottom_faces = [
            [0,1,3],
            [3,1,4],
            [1,2,5],
            [1,5,4]
        ]
        aligner_top_faces = list(np.array(aligner_bottom_faces) + 6)
        aligner_side_faces = [
            [8,2,11],
            [2,11,5],
            [8,2,7],
            [2,1,7],
            [1,7,0],
            [7,6,0],
            [0,6,9],
            [0,3,9],
            [3,9,10],
            [3,4,10],
            [4,10,5],
            [10,5,11]
        ]
        aligner_faces = np.array([*aligner_bottom_faces, *aligner_top_faces, *aligner_side_faces])
        # Create the mesh
        aligner_mesh = stl.mesh.Mesh(np.zeros(aligner_faces.shape[0], dtype=stl.mesh.Mesh.dtype))
        for i, face in enumerate(aligner_faces):
            for j in range(3):
                aligner_mesh.vectors[i][j] = aligner_vertices[face[j],:]

        # Fuse with hex prism
        hex_prism = fuse_models([hex_prism, aligner_mesh])
            
    return hex_prism

def fuse_models(models):
    """Fuses together a list of 3D models by concatenating their vertices and faces"""
    all_data = np.concatenate([model.data.copy() for model in models])
    return stl.mesh.Mesh(all_data)
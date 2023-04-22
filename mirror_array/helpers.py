import numpy as np
import matplotlib.pyplot as plt

HEX_NUMBERS = [1 + sum(6 * (r-1) for r in range(1, radius+1)) for radius in range(1, 100)]

def is_hexagonal_number(n):
    # This is the really janky way of doing this but whatever ¯\_(ツ)_/¯
    return n in HEX_NUMBERS

def get_hex_radius(n):
    assert is_hexagonal_number(n), "must be a hex number!"
    return HEX_NUMBERS.index(n)


def sort_by_predicate(arr, predicate):
    '''Sort a numpy array by a predicate (along axis=0) because for some reason this isn't a standard method'''
    l = list(arr)
    l.sort(key=lambda element: predicate(element))
    return np.array(l)

def to_mm(inches):
  return round(inches*25.4,2) if inches else None

def to_inches(mm):
  return round(mm/25.4,2) if mm else None


FOCAL_PLANE_DEFAULT = to_mm(6*12) # 6 feet in mm

VERTICAL_DISTANCE_DEFAULT = to_mm(46) # 46 inch in mm
HORIZONTAL_DISTANCE_DEFAULT = to_mm(46) # 46 inch in mm
        
def create_target_positions_wall(target_coordinates, depth = FOCAL_PLANE_DEFAULT):
    """Converts a list of xy target coordinates into a list of xyz target positions, assuming you are projecting against a vertical wall at a specified focal distance."""
    target_positions = np.array([[c[0], c[1], depth] for c in target_coordinates])
    return target_positions

def create_target_positions_ground(target_coordinates, vertical_distance=VERTICAL_DISTANCE_DEFAULT, horizontal_distance=HORIZONTAL_DISTANCE_DEFAULT):
    """Converts a list of xy target coordinates into a list of xyz target positions, assuming you are projecting against the ground as the focal plane, at a specified horizontal distance and vertical distance from the focal plane."""
    target_positions = np.array([[c[0], -1 * vertical_distance, c[1] + horizontal_distance] for c in target_coordinates])
    return target_positions



# Plotting Functions

def plot_coords(coords, width_in, height_in, max_size_in=20):
    big = max(width_in, height_in)
    while big > max_size_in:
      width_in /= 2
      height_in /= 2
      big = max(width_in, height_in)
    plt.figure(figsize=(width_in, height_in))
    plt.scatter(coords.T[0], coords.T[1])
    plt.show()


def plot_colored_projection(target_positions):
  # This demonstrates the order of the sorted target positions with corresponding mirrors. 
  # Purple dots correspond to the innermost mirrors, red to the outermost.
  plt.scatter(target_positions[:,0], target_positions[:,2], c=np.arange(len(target_positions)), cmap="rainbow")
  plt.axis("equal")
  plt.show()


# Demonstrate the flower-petal model partitioning scheme I'm using
def plot_colored_grid(points, partition_indices):
    plt.figure(figsize=(10,10))
    points = np.array(points)
    # Divide into quarters
    plt.scatter(points.T[0], points.T[1], c=partition_indices, s=300.0, cmap="rainbow")
    for i, p in enumerate(points):
      plt.text(p[0], p[1], int(partition_indices[i]), ha="center", va="center")
    plt.axis("equal")
    plt.show()
import numpy as np
from mirror_array.helpers import *


def get_coords(coords, width_in=None, height_in=None, **kwargs):
  return get_coords_mm(coords, width_mm=to_mm(width_in), height_mm=to_mm(height_in), **kwargs)


def get_coords_mm(coords, width_mm=None, height_mm=None, resize_both=True, mirror_x=True, mirror_y=False, debug=False):

  coords = np.array(coords, dtype=np.float64)

  if width_mm or height_mm:
    coords = scale_coords(coords, width_mm, height_mm, resize_both)
  
  width_mm = max(coords.T[0]) - min(coords.T[0])
  height_mm = max(coords.T[1]) - min(coords.T[1])

  # center the coords around 0
  coords -= np.array([max(coords.T[0])-width_mm/2, max(coords.T[1])-height_mm/2])

  # mirror the image
  coords *= np.array([-1 if mirror_x else 1,-1 if mirror_y else 1])

  if debug:
    width_in = to_inches(width_mm)
    height_in = to_inches(height_mm)
    print(f"Number of mirrors: {len(coords)}")
    print(f"Width: {round(width_mm,2)}mm or {width_in}in")
    print(f"Height: {round(height_mm,2)}mm or {height_in}in")
    plot_coords(coords, width_in, height_in)

  return coords



def scale_coords(coords, width, height, resize_both):
  current_width = max(coords.T[0]) - min(coords.T[0])
  current_height = max(coords.T[1]) - min(coords.T[1])
  if width and height:
    width_ratio = width/current_width
    height_ratio = height/current_height
    coords *= np.array([width_ratio,height_ratio])
  elif width:
    width_ratio = width/current_width
    coords *= width_ratio if resize_both else np.array([width_ratio,1])
  else:
    height_ratio = height/current_height
    coords *= height_ratio if resize_both else np.array([1,height_ratio])
  return coords
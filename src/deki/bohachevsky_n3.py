import torch
import math
from .visualizer import Visualizer

__all__ = ['bohachevskyn3_fn', 'BohachevskyN3']

def bohachevskyn3_fn(X):
  x, y = X
  t1 = torch.pow(x,2) + 2*torch.pow(y,2) 
  t2 = 0.3*torch.cos((3*math.pi*x) + (4*math.pi*y)) * torch.cos(4*math.pi*y)

  return t1 - t2 + 0.3


BohachevskyN3 = Visualizer(bohachevskyn3_fn,
                        x_range = (-5, 5),
                        y_range = (-5, 5),
                        minima = (0, 0),
                        start = (0.5, 0.5), 
                        precision = 200)
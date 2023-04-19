import torch
import math
from .visualizer import Visualizer

__all__ = ['bohachevskyn2_fn', 'BohachevskyN2']

def bohachevskyn2_fn(X):
  x, y = X
  t1 = torch.pow(x,2) + 2*torch.pow(y,2) 
  t2 = 0.3*torch.cos(3*math.pi*x) * torch.cos(4*math.pi*y)

  return t1 - t2 + 0.3


BohachevskyN2 = Visualizer(bohachevskyn2_fn,
                        x_range = (-1, 1),
                        y_range = (-1, 1),
                        minima = (0, 0),
                        start = (0.5, 0.5), 
                        precision = 200)
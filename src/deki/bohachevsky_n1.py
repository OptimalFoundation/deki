import torch
import math
from .visualizer import Visualizer

__all__ = ['bohachevsky_fn', 'BohachevskyN1']

def bohachevsky_fn(X):
  x, y = X
  t1 = torch.pow(x,2) + 2*torch.pow(y,2) 
  t2 = 0.3*torch.cos(3*math.pi*x) + 0.4*torch.cos(4*math.pi*y)

  return t1 - t2 + 0.7


BohachevskyN1 = Visualizer(bohachevsky_fn,
                        x_range = (-1, 1),
                        y_range = (-1, 1),
                        minima = (0, 0),
                        start = (0.5, 0.5), 
                        precision = 200)
import torch
from .visualizer import Visualizer

__all__ = ['bartels_fn', 'Bartels']

def bartels_fn(X):
  x, y = X
  res = torch.abs(torch.pow(x,2) + torch.pow(y,2) + x*y) + torch.abs(torch.sin(x)) + torch.abs(torch.cos(y))
  return res


Bartels = Visualizer(bartels_fn,
                        x_range = (-5, 5),
                        y_range = (-5, 5),
                        minima = (0, 0),
                        start = (1, 1), 
                        precision = 200)
import torch
from .visualizer import Visualizer

__all__ = ['alpinen2_fn', 'AlpineN2']

def alpinen2_fn(X):
  x, y = X
  x = torch.sqrt(x) * torch.sin(x)
  y = torch.sqrt(y) * torch.sin(y)
  return -(x*y)


AlpineN2 = Visualizer(alpinen2_fn,
                        x_range = (0, 10),
                        y_range = (0, 10),
                        minima = (7.917, 7.917),
                        start = (1, 1), 
                        precision = 200)
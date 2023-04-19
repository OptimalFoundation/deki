import torch
from .visualizer import Visualizer

__all__ = ['alpineN1_fn', 'AlpineN1']

def alpinen1_fn(X):
  x, y = X
  x = x * torch.sin(x) + 0.1 * x
  y = y * torch.sin(y) + 0.1 * y
  return 1/2 * (x+y)


AlpineN1 = Visualizer(alpinen1_fn,
                        x_range = (0, 10),
                        y_range = (0, 10),
                        minima = (0, 0),
                        start = (1, 1), 
                        precision = 200)
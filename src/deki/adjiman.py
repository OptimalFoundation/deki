import torch
from .visualizer import Visualizer

__all__ = ['adjiman_fn', 'Adjiman']

def adjiman_fn(X):
  x, y = X
  res = (torch.cos(x) * torch.sin(y)) - (x / (torch.pow(y, 2) + 1))
  return res

Adjiman = Visualizer(adjiman_fn,
                        x_range = (-1, 2),
                        y_range = (-1, 1),
                        minima = (0, 0),
                        start = (-0.25, 0.25), 
                        precision = 200)
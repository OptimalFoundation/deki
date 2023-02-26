import torch
from .visualizer import Visualiser

__all__ = ['sphere_fn', 'Sphere']

def sphere_fn(X):
  x, y = X
  return torch.pow(x, 2) + torch.pow(y, 2)

Sphere = Visualiser(sphere_fn,
                    x_range=(-2, 2),
                    y_range=(-2, 2), 
                    minima=(0, 0), 
                    start = (-1.5, 1), 
                    precision = 200  
                    )


import torch
from .visualizer import Visualizer

__all__ = ['baele_fn', 'Baele']

def baele_fn(X):
  x, y = X
  t1 = torch.pow((1.5 - x + x*y), 2)
  t2 = torch.pow((2.25 - x + x*torch.pow(y,2)), 2)
  t3 = torch.pow(2.625 - x + x*torch.pow(y,3), 2) 
  return t1 + t2 + t3


Baele = Visualizer(baele_fn,
                        x_range = (-4.5, 4.5),
                        y_range = (-4.5, 4.5),
                        minima = (3, 0.5),
                        start = (1, 1), 
                        precision = 200)
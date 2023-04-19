import torch
from .visualizer import Visualizer

__all__ = ['rosenbrock_fn', 'Rosenbrock']

def rosenbrock_fn(X, a=1, b=100):
  x, y = X
  t1 = torch.pow((a-x), 2)
  t2 = torch.mul(b, torch.pow(y - torch.pow(x, 2),2))
  return t1 + t2


Rosenbrock = Visualizer(rosenbrock_fn,
                        x_range = (-2, 2),
                        y_range = (-1, 3),
                        minima = (1, 1),
                        start = (-1.5, 1), 
                        precision = 200)


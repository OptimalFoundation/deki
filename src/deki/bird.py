import torch
import math
from .visualizer import Visualizer

__all__ = ['bird_fn', 'Bird']

def bird_fn(X):
  x, y = X
  t1 = torch.sin(x) * torch.exp(torch.pow((1 - torch.cos(y)), 2))
  t2 = torch.cos(y) * torch.exp(torch.pow((1 - torch.sin(x)), 2))
  t3 = torch.pow((x-y), 2) 
  return t1 + t2 + t3


Bird = Visualizer(bird_fn,
                        x_range = (-2*math.pi, 2*math.pi),
                        y_range = (-2*math.pi, 2*math.pi),
                        minima = (4.701, 3.152),
                        start = (1, 1), 
                        precision = 200)
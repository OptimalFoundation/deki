import torch
from .visualizer import Visualizer
__all__ = ['ackleyn2_fn','AckleyN2']


def ackleyn2_fn(X):
  x, y = X
  return -200 * torch.exp(-0.2 * torch.sqrt(torch.pow(x,2)+torch.pow(y,2)))



AckleyN2 = Visualizer(ackleyn2_fn,
                        x_range = (-2, 2),
                        y_range = (-1, 3),
                        minima = (0,0),
                        start = (-1.5, 2.5), 
                        precision = 200)

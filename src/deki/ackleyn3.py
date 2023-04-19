import torch
from .visualizer import Visualizer
__all__ = ['ackleyn3_fn','AckleyN3']


def ackleyn3_fn(X):
  x,y = X
  res = -200 * torch.exp(-0.2 * torch.sqrt(torch.pow(x,2)+torch.pow(y,2))) 
  res += 5 * torch.exp(torch.cos(3*x)+torch.sin(3*y))
  return res



AckleyN3 = Visualizer(ackleyn3_fn,
                        x_range = (-3.2, 3.2),
                        y_range = (-3.2, 3.2),
                        minima = (1, 1),
                        start = (-0.68, -0.36), 
                        precision = 200)
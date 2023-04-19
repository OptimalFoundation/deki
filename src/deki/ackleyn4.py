import torch
from .visualizer import Visualizer
__all__ = ['ackley', 'ackleyn2_fn', 'ackleyn3_fn', 'ackleyn4_fn', 'Ackley','AckleyN2', 'AckleyN3', 'AckleyN4']



def ackleyn4_fn(X):
  x, y = X
  res  = torch.mul(torch.exp(torch.tensor(-0.2)), torch.sqrt(torch.pow(x,2)+torch.pow(y,2))) + 3 * torch.cos(2 * x)+ torch.sin(2 * y)
  return res


AckleyN4 = Visualizer(ackleyn4_fn,
                        x_range = (-2, 2),
                        y_range = (-1, 3),
                        minima = (1, 1),
                        start = (-1.5, 2.5), 
                        precision = 200)
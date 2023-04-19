import torch
from .visualizer import Visualizer
__all__ = ['ackley_fn', 'Ackley']


def ackley_fn(X, a=20, b=0.2, c=2 * 22/7):
  x,y = X
  t1 = -a * torch.exp(-b * torch.sqrt((1/2 * (torch.pow(x,2) + torch.pow(y,2)))))
  t2 = torch.exp(1/2 * (torch.cos(c*x) + torch.cos(c*y))) + a + torch.exp(torch.tensor(1))
  return t1 - t2



Ackley = Visualizer(ackley_fn,
                        x_range = (-3.2, 3.2),
                        y_range = (-3.2, 3.2),
                        minima = (0, 0),
                        start = (-1.5, 2.5), 
                        precision = 200)







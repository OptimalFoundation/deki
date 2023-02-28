import os
from typing import Callable, Union, List, Tuple, Dict

import matplotlib.pyplot as plt
import numpy as np

import imageio

import torch
from torch import optim
from torch.optim import Optimizer

__all__ = ['Visualizer']


class Visualizer:

  def __init__ (self, 
                function : Callable[..., torch.Tensor],
                x_range : Union[torch.Tensor, np.ndarray, Tuple[int, int]], 
                y_range : Union[torch.Tensor, np.ndarray, Tuple[int, int]],
                start: Union[torch.Tensor, np.ndarray],
                minima: Union[torch.Tensor, np.ndarray], 
                precision : int = 200, 
                save_dir : str = './__deki__'):
    
    self.func = function
    
    self.start = self.__type_check__(start)
    self.minima = self.__type_check__(minima)
    self.x_range = self.__type_check__(x_range)
    self.y_range = self.__type_check__(y_range)

    self.precision = precision

    if save_dir.split('/')[-1] not in os.listdir():
      os.mkdir(save_dir.split('/')[-1])
      self.save_dir = save_dir
    else:
      self.save_dir = save_dir
  def __type_check__(self, var, expected_shape = (2,)):
    if type(var) == torch.Tensor:
      var = var.numpy()
    
    if type(var) == tuple:
      var = np.asarray(var, dtype=np.float64)

    if type(var) == np.ndarray:
      assert var.shape == expected_shape, f"variable shape does not match the expected"

    return var

  def _get_function_plot(self, traj=None, ax=None, label=None, title = None):
    if ax is None:
      plt.figure(figsize=(10,10))
      ax = plt.axes()

      if title != None:
        ax.title(title)
    
    x = torch.linspace(self.x_range[0], self.x_range[1], self.precision)
    y = torch.linspace(self.y_range[0], self.y_range[1], self.precision)
    X = torch.meshgrid(x, y, indexing='xy')


    Z = self.func(X)

    ax.contour(X[0], X[1], Z, levels=50, cmap='jet')
    
    ax.set_xlabel('x')
    ax.set_ylabel('y')

    if type(traj) == List or type(traj) == np.ndarray:
      path = ax.plot(traj[:, 0], traj[:, 1], label = label)
      color = ax.get_lines()[-1].get_color()
      ax.plot(traj[-1, 0], traj[-1, 1], color=color, marker="o")

    ax.plot(self.start[0], self.start[1], color="red", marker = "x")
    ax.plot(self.minima[0], self.minima[1], color="green", marker="x")


  def _set_start(self, start : Tuple[float, float]):
    self.start = start
  
  def _get_traj(self, optimizer: Optimizer, opt_state : Dict = {}, steps : int = 100):
    x = torch.tensor(self.start, requires_grad = True)
    opt = optimizer([x], **opt_state)

    traj = []

    for _ in range(steps):
      opt.zero_grad()
      z = self.func(x)
      z.backward()
      opt.step()

      traj.append(x.detach().numpy().copy())
    
    return np.asarray(traj)
  
  
  def plot(self,
           optimizers : Union[Optimizer, List[Optimizer], Tuple[Optimizer, Dict], List[Tuple[Optimizer, Dict]]],
           steps : int, 
           title : str = None,
           save : bool = True, 
           ):
    
    if type(optimizers) == list:
      self._plot_multiple(optimizers, steps, title)
    else:
      self._plot_single(optimizers, steps, title)

    plt.show()
    if save: 
      self._save_plot(title)

  def _save_plot (self, title : str = None):
    
    if title != None:
      assert type(title) == str, f"Title must be of str type, you gave {type(title)}"
      
      plt.savefig(self.save_dir +
                  '/' +
                  f"{title}.png")
    else:
      plt.savefig(self.save_dir +
                  '/' +
                  f"{len(os.listdir(self.save_dir)) + 1}.png")
  
  def _plot_single (self,
                    optimizer: Union[Optimizer, Tuple[Optimizer, Dict]],
                    steps:int, 
                    title : str = None) -> None:
      
    if type(optimizer) == tuple:
      traj = self._get_traj(optimizer[0], opt_state=optimizer[1], steps = steps)
      self._get_function_plot(traj=traj)
    else:
      traj = self._get_traj(optimizer, steps = steps)
      self._get_function_plot(traj=traj)

  def _plot_multiple (self,
                      optimizers: Union[List[Optimizer], List[Tuple[Optimizer, Dict]]],
                      steps : int, 
                      title : str = None) -> None:

      if type(optimizers[0]) == tuple: 
        plt.figure(figsize=(10, 10))
        ax = plt.axes()
        
        if title != None:
          ax.title(title)

        legend = []

        for i, (opt, opt_state) in enumerate(optimizers):
          traj = self._get_traj(opt, opt_state, steps=steps)

          self._get_function_plot(traj=traj,
                                  ax=ax,
                                  label=f"{opt.__name__}:{opt_state['lr']}")
        
        ax.legend()
      
      else:
        plt.figure(figsize=(10, 10))
        ax = plt.axes()
        for opt in optimizers:
          traj = self._get_traj(opt, {"lr" : 3E-4}, steps=steps)
          self._get_function_plot(traj=traj,
                                  ax=ax,
                                  label=f"{opt.__name__}")
          
  def gif(self,
          optimizers : Union[Optimizer, List[Optimizer],
                             Tuple[Optimizer, Dict],
                             List[Tuple[Optimizer, Dict]]],
          steps : int
          ):
    pass

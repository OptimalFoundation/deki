import numpy as np
import matplotlib.pyplot as plt
from torch.optim import Optimizer

from .visualizer import Visualizer
from .sphere import sphere_fn, Sphere
from .rosenbrock import rosenbrock_fn, Rosenbrock
from .ackley import ackley_fn, Ackley
from .ackleyn2 import ackleyn2_fn, AckleyN2
from .ackleyn3 import ackleyn3_fn, AckleyN3
from .ackleyn4 import ackleyn4_fn, AckleyN4
from .alpinen1 import alpinen1_fn, AlpineN1
from .alpinen2 import alpinen2_fn, AlpineN2
from .adjiman import adjiman_fn, Adjiman
from .bartels import bartels_fn, Bartels
from .baele import baele_fn, Baele
from .bird import bird_fn, Bird
from .bohachevsky_n1 import bohachevsky_fn, BohachevskyN1
from .bohachevsky_n2 import bohachevskyn2_fn, BohachevskyN2
from .bohachevsky_n3 import bohachevskyn3_fn, BohachevskyN3

__version__ = "0.0.1.dev1"

__all__ = ('Visualizer',
           'sphere_fn',
           'Sphere', 
           'rosenbrock_fn',
           'Rosenbrock',
           'ackley_fn',
           'Ackley',
           'ackleyn2_fn',
            'AckleyN2',
            'ackleyn3_fn',
            'AckleyN3',
            'ackleyn4_fn',
            'AckleyN4',
            'adjiman_fn',
            'alpinen1_fn', 
            'AlpineN1',
            'alpinen2_fn', 
            'AlpineN2',
            'Adjiman',
            'bartels_fn',
            'Bartels',
            'baele_fn',
            'Baele',
            'bird_fn',
            'Bird',
            'bohachevsky_fn',
            'BohachevskyN1',
            'bohachevskyn2_fn',
            'BohachevskyN2',
            'bohachevskyn3_fn',
            'BohachevskyN3',            
        )
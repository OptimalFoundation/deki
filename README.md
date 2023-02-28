# ✨ Deki ✨

Deki is a library is to create beautiful ✨interative✨ visuals to understand your optimizer better, because seeing is believing!

And the best part, it is extremely simple to use! Just a few steps and you'll find your visuals ready to be played with.

## Table of Contents

- [✨ Deki ✨](#✨-deki-✨)
- [Table of Contents](#table-of-contents)
- [Installation](#installation)
- [Usage](#usage)
    - [Custom Optimizer](#custom-optimizer)
    - [Custom Test Function](#custom-test-function)
- [Supported Functions](#supported-functions)
- [Acknowledgement](#acknowledgement)
- [Citation](#citation)


## Installation
```bash
$ pip install deki
```
## Usage

Deki works by taking in `torch.optim.Optimizer` classes and a configuration dictionary for the hyperparameters and other keyword arguments, like in the example below 🔻

```python
from torch.optim import SGD
from deki import Rosenbrock

optimiser = SGD
optimiser_state = {"lr":1E-3}
steps = 1000

Rosenbrock.plot((optimiser, optimiser_state), steps)
```


![Rosenbrock Plot with SGD](https://github.com/Dawn-Of-Eve/deki/raw/main/assets/rosenbrock_sgd.png)

### Custom Optimizer

Just like its mentioned above, any PyTorch optimizer class can be used as long as its required keyword args can be passed in a dict. In conclusion, **Deki can handle your custom optimizers, so build away!**

```python
from torch.optim import Optimizer
from deki import Rosenbrock

class MySpecialOptimizer(Optimizer):
    ...

optimizer = MySpecialOptimizer
optimiser_state = {"lr":1E-3}
steps = 1000

Rosenbrock.plot((optimiser, optimiser_state), steps)
```


### Custom Test Function

Not only this, Deki also manages to make its Visualizer general so you can implement your custom test functions super easily, just like in the following manner 🔻

```python
import torch
from torch.optim import SGD
from deki import Visualizer

# you can define any function as long as it takes a X
# wherein X will be torch Tensor tuple of x,y
def my_special_test_function (X : Tuple[torch.Tensor, torch.Tensor]):
    x, y = X

    # as an example we will return the sphere function
    # use only torch ops to maintain differentiability
    return torch.pow(x, 2) + torch.pow(y, 2)

# Define the object for the Visualizer class
# For supported functions, such objects come pre-defined in deki
visualizer = Visualizer( my_special_test_function, 
                         x_range = (-2,2),
                         y_range = (-2,2),
                         minima  = (0, 0),
                         start   = (-1.5, 1)
                        )

# Now call the plot function with the optimizer to get the plot
visualizer.plot((SGD, {"lr":1E-3}), steps=1000)
```
![Custom function plot with SGD](https://github.com/Dawn-Of-Eve/deki/raw/main/assets/sphere_sgd.png)


## Supported Functions

|  Function  	| Example 	|
|:----------:	|:-------:	|
| Rosenbrock 	|         	|
|   Sphere   	|         	|


## Acknowledgement

A special thanks to all the [contributors](https://github.com/Dawn-Of-Eve/deki/graphs/contributors) on this project for making it what it is today! ❤️

If you found this repository helpful, please leave a [star](https://github.com/Dawn-Of-Eve/deki/stargazers)! ⭐



## Citation 

If you are using this repository for research, please cite it in the following manner:

```bibtex
 @misc{minhas_2023,
  url={https://github.com/Dawn-Of-Eve/deki},
  title={Deki: Library for Visualizing Optimization Algorithms}, 
  publisher={Dawn Of Eve},
  author={Minhas, Bhavnick Singh},
  year={2023},
  month={Feb}} 
```

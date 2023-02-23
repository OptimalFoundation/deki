# :sparkles: Deki

Deki is a library is to create beautiful ✨interative✨ visuals to understand your optimizer better, because seeing is believing!

Just a few easy steps to get your own visuals:

```
from deki import Rosenbrock

optimiser = my_special_optimiser
optimiser_state = {"lr":0.01}
steps = 1000

Rosenbrock.plot((optimiser, optimiser_state), steps)

#output: get a interative Rosenbrock plot
```
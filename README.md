The repository provides utilities for integrating and sampling from pretrained neural flows.

By flows we mean the models defined in the following;

- [Flow Matching for Generative Modeling](https://arxiv.org/abs/2210.02747)
- [Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow](https://arxiv.org/abs/2209.03003)
- [Stochastic Interpolants: A Unifying Framework for Flows and Diffusions](https://arxiv.org/abs/2303.08797)

## Usage

A example use case is generating samples from a pretrained neural flow.

```python
import flow

# defined elsewhere
import pretrained_nn import model, params
import source_distribution

velocity_field = lambda z, t: model(params, z, t)
f = flow.Flow(velocity_field)

y = source_distribution.sample(n)
x = f.backward(y, n_steps=50)
```

# TODO

- support stochastic differential equations
- move to diffrax / similar to integrate the dynamical systems
<!-- - support entropy calculation -->
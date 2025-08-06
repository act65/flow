### Flow

A example use case is generating samples from a pretrained neural flow.

We support two kinds of dynamical system;

- ODEs (flows)
- SDEs (process aka diffusion models)

The key difference are;

- `flows`;
    - require only a velocity field
    - support `forward` / `backward` evolution (ie sampling), `push_forward` / `push_backward` evolution (allowing the evaluation of p(x))
- `processes`;
    - require a deterministic velocity field (aka the drift) and a score function (aka diffusion)
    - support only `forward` / `backward` evolution (ie sampling)

```python
import flow

# defined elsewhere
import pretrained_nn import model, params
import source_distribution

velocity_field = lambda z, t: model(params, z, t)
f = flow.Flow(velocity_field)

# sample from p_1
x0 = source_distribution.sample(n)
x1 = f.forward(x0, n_steps=50)

# evaluate the probability of x1
x1_observed = x1  # observations are generated somehow
x0_observed = f.backward(x1_observed)
p_x0_observed = source_distribution.prob(x0_observed)  # requires the ability to evaluate p_0(x). when p_0 is a guassian this is easy.
p_x1_observed, _ = f.push_forward(p_x0_observed, x0_observed)
```
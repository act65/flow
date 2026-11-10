# flow

Continuous normalizing flows in [JAX](https://github.com/google/jax) /
[Equinox](https://github.com/patrick-kidger/equinox): integrate a learnable
vector field forwards and backwards, and evaluate densities by pushing them
through the flow.

By *flows* we mean the models of

- [Flow Matching for Generative Modeling](https://arxiv.org/abs/2210.02747)
- [Flow Straight and Fast: Rectified Flow](https://arxiv.org/abs/2209.03003)
- [Stochastic Interpolants: A Unifying Framework for Flows and Diffusions](https://arxiv.org/abs/2303.08797)

using the notation of stochastic interpolants. The interpolants, couplings and
training targets live in the companion library
[sinterp](https://github.com/act65/sinterp); conditional and structured flows
built on both live in [structflow](https://github.com/act65/structflow).

| module | contents |
| :--- | :--- |
| `flow.flow` | `Flow` (ODE): `forward` / `backward` sampling, `push_forward` / `push_backward` densities, batched `b_*` versions |
| `flow.process` | `Process` (SDE): drift + score sampling |
| `flow.integrators` | Euler, RK4, Euler–Maruyama, and adaptive solvers via diffrax |
| `flow.net` | small network building blocks |
| `flow.utils` | divergence estimators (exact and Hutchinson) |

See [`flow/README.md`](flow/README.md) for a worked example.

## Install

Install JAX for your hardware first (`pip install "jax[cpu]"` or
`"jax[cuda12]"`), then

```bash
pip install "flow-jax @ git+https://github.com/act65/flow@v0.2.0"   # import flow
```

or, from a clone, `pip install -e ".[dev]"` and run `pytest`.

> Version 0.2.0 is a rewrite: earlier versions of this repository also
> contained `sinterp` and `bayes`, which now live in
> [sinterp](https://github.com/act65/sinterp) and
> [structflow](https://github.com/act65/structflow).

## License

MIT, see [LICENSE](LICENSE).

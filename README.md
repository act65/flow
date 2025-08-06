The repository provides utilities for working with neural flows aka stochastic interpolants.

By flows we mean the models defined in the following;

- [Flow Matching for Generative Modeling](https://arxiv.org/abs/2210.02747)
- [Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow](https://arxiv.org/abs/2209.03003)
- [Stochastic Interpolants: A Unifying Framework for Flows and Diffusions](https://arxiv.org/abs/2303.08797)

( we use notation / conventions from stochastic interpolants )

- sinterp: implements the core math of stochastic interpolants
- flow: handles the integration of the vector fields learnable via the sinterp framework
- bayes: uses sinterp/flow to construct distributions which support bayesian manipulation.
Our goal here is to use neural flows to power any / all computations we might want to do with probabilities / distributions. This includes;

- modelling a distribution (see `bayes/distribution`):
    - sampling $x \sim p(x)$
    - evaluating probabilities $p(x)$
- bayesian reasoning (`bayes/posterior`)
    - calculate a posterior: $p(\theta \mid D) = \frac{p(D \theta \theta)p(\theta)}{p(D)}$
    - maximum a posteriori
- calculating expectations of various quantities (TODO)
    - $\mathbb E_{p(x)}[x]$ the mean
    - $\mathbb E_{p(x)}[f(x)]$
    - $-\mathbb E_{p(x)}[\log p(x)]$ the entropy
- KL divergence (TODO)

Thoughts

- because flows provide access to $x\sim p(x)$ we can use MCMC estimates of expectations whose error will converge with $\mathcal O(\frac{1}{\sqrt(n)})$.
- the posterior can be calculated non-parametrically and integrated into the flow using $\nabla \log p(x_t | D) = \nabla \log p(D | x_t) + \log p(x_t)$. This is also known as 'guidance' and is easily derived from Bayes equation.
- MAP can be found (somewhat) efficiently, again because we have access to both $p(x)$ and samples $x\sim p(x)$ (we can sample starting points then follow gradients of $p(x)$)
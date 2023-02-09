---
jupytext:
  formats: md:myst,ipynb
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.14.4
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

# Poisson Gaussian Process Regression

We consider a non-parametric univariate Poisson regression model for count data. The log rate of the Poisson observation models is a Gaussian process $z(x)$ with squared exponential covariance function, i.e.,
$$\begin{align}
x_i &\sim \mathsf{Normal}\left(0, 1\right)\text{ for }i\in\left\{1,\ldots,n\right\}\\
z &\sim\mathsf{Gaussian Process}\left(x, \sigma, \ell\right)\\
y_i &\sim \mathsf{Poisson}\left(\exp(z_i)\right),
\end{align}$$
where $x$ are covariates, $\sigma$ and $\ell$ are the amplitude and correlation length of the Gaussian process, respectively, and $y$ are count observations.

Let's define the model, generate some data, and visualize it.

```{code-cell} ipython3
from matplotlib import pyplot as plt
import torch as th
import vbzero as vz


@vz.util.model(return_state=True)
def model():
    # Sample size.
    n = 50
    # Sample kernel parameters, covariates, and evaluate the covariance.
    sigma = vz.util.sample("sigma", th.distributions.Gamma(2, 2))
    length_scale = vz.util.sample("length_scale", th.distributions.Gamma(2, 2))
    x = vz.util.sample("x", th.distributions.Normal(0, 1), sample_shape=n)
    cov = sigma * (- (x[:, None] - x) ** 2 / (2 * length_scale ** 2)).exp() + 1e-3 * th.eye(n)
    # Sample the latent Gaussian process and count observations.
    z = vz.util.sample("z", th.distributions.MultivariateNormal(th.zeros(n), cov))
    vz.util.sample("y", th.distributions.Poisson(z.exp()))

th.manual_seed(0)
state = model()

fig, ax = plt.subplots()
ax.scatter(state["x"], state["y"], label=r"counts $y\sim\mathsf{Poisson}\left(\lambda\right)$")
idx = state["x"].argsort()
ax.plot(state["x"][idx], state["z"][idx].exp(), label=r"rate $\lambda=\exp\left(z(x)\right)$")
ax.set_xlabel("covariate x")
ax.legend()
```

Having declared the generative model, let's define a variational approximation for the parameters theta with random starting points. A {class}`vbzero.nn.ParameterizedDistribution` is an easy way to define a distribution whose parameters can be optimized. Calling the module returns a distribution, and we can draw samples from it.

```{code-cell} ipython3
n, = state["x"].size()
approximation = vz.nn.ParameterizedDistributionDict({
    "z": vz.nn.ParameterizedDistribution(th.distributions.Normal, loc=th.randn(n), 
                                         scale=1 * th.ones(n)),
    "length_scale": vz.nn.ParameterizedDistribution(th.distributions.LogNormal, loc=0., scale=1.),
    "sigma": vz.nn.ParameterizedDistribution(th.distributions.LogNormal, loc=0., scale=1.),
})
```

With the variational approximation in hand, we define a variational loss module which returns a stochastic estimate of the negative [evidence lower bound](https://en.wikipedia.org/wiki/Evidence_lower_bound) (ELBO) and the entropy of the distribution. However, we need to condition the model on the data before we can evaluate the loss.

```{code-cell} ipython3
import numpy as np
import os
from vbzero.util import train

conditioned = vz.util.condition(model, state["x", "y"])
loss = vz.nn.VariationalLoss(conditioned, approximation)
steps = 10 if "CI" in os.environ else 2_000


def run_epoch(loss, lr, steps):
    optim = th.optim.Adam(approximation.parameters(), lr=lr)
    losses = []
    for _ in range(steps):
        optim.zero_grad()
        loss_value, _ = loss()
        loss_value.backward()
        optim.step()
        losses.append(loss_value.item())
    print(f"epoch with lr={lr}; loss={np.mean(losses)}")
    return losses

losses = run_epoch(loss, 0.1, steps)
losses += run_epoch(loss, 0.02, steps)
losses += run_epoch(loss, 0.01, steps)

fig, ax = plt.subplots()
ax.plot(losses)
ax.set_xlabel("epoch")
ax.set_ylabel("step")
ax.set_yscale("log")
fig.tight_layout()
```

Finally, let's visualize the inferred kernel parameters and the latent Gaussian process rate.

```{code-cell} ipython3
dist = approximation()
sample = dist.sample([200])
fig, (ax1, ax2) = plt.subplots(1, 2)
ax1.scatter(sample["length_scale"], sample["sigma"], marker=".", alpha=0.5, label="posterior samples")
ax1.scatter(state["length_scale"], state["sigma"], edgecolor="w", label="ground truth")
ax1.set_xlabel(r"length scale $\ell$")
ax1.set_ylabel(r"amplitude $\sigma$")
ax1.legend(fontsize="small")

l, m, u = sample["z"].exp().quantile(th.as_tensor([0.025, 0.5, 0.975]), axis=0)
ax2.plot(state["x"][idx], m[idx], label="posterior\nsamples")
ax2.fill_between(state["x"][idx], l[idx], u[idx], alpha=0.25)
ax2.plot(state["x"][idx], state["z"][idx].exp(), ls="--", label="ground truth")
ax2.set_xlabel("covariate $x$")
ax2.set_ylabel("rate $\lambda$")
ax2.legend(fontsize="small")

fig.tight_layout()
```

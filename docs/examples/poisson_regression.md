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

# Poisson regression with vbzero

We consider a univariate Poisson regression model for count data with random effects, and we will estimate two parameters: the intercepts (one for each group) and slope of the Poisson rate on the log scale. Let's define the model, generate some data, and visualize it.

```{code-cell} ipython3
import matplotlib as mpl
from matplotlib import pyplot as plt
import torch as th
from vbzero.util import model, sample


@model(return_state=True)
def poisson_regression():
    # We can also define n and k as hyperparameters or pass them in as arguments. But let's define
    # them here for simplicity.
    n = 100
    k = 3
    z = sample("z", th.distributions.Categorical(logits=th.zeros(k)), sample_shape=n)
    x = sample("x", th.distributions.Normal(0, 1), sample_shape=n)
    intercepts = sample("intercepts", th.distributions.Normal(0, 3), sample_shape=k)
    slope = sample("slope", th.distributions.Normal(0, 1))
    log_rate = intercepts[z] + slope * x
    y = sample("y", th.distributions.Poisson(log_rate.exp()))


th.manual_seed(4)  # For reproducibility.
state = poisson_regression()

def plot_state(state):
    fig, ax = plt.subplots()
    cmap = mpl.cm.viridis
    ax.scatter(state["x"], state["y"], c=state["intercepts"][state["z"]], cmap=cmap)

    norm = mpl.colors.Normalize(state["intercepts"].min(), state["intercepts"].max())
    mappable = mpl.cm.ScalarMappable(norm, cmap)
    lin = th.linspace(state["x"].min(), state["x"].max(), 50)
    for intercept in state["intercepts"]:
        color = mappable.to_rgba(intercept)
        ax.plot(lin, (intercept + state["slope"] * lin).exp(), color=color, ls="--")
    ax.set_yscale("log")
    ax.set_xlabel("covariate $x$")
    ax.set_ylabel("counts $y$")
    fig.tight_layout()

plot_state(state)
```

Having declared the generative model, let's define a variational approximation for the parameters theta with random starting points. A {class}`vbzero.nn.ParameterizedDistribution` is an easy way to define a distribution whose parameters can be optimized. Calling the module returns a distribution, and we can draw samples from it.

```{code-cell} ipython3
from vbzero.nn import ParametrizedDistribution, ParameterizedDistributionDict

k, = state["intercepts"].size()
approximation = ParameterizedDistributionDict({
    "intercepts": ParametrizedDistribution(th.distributions.Normal, loc=th.randn(k),
                                           scale=th.ones(k)),
    "slope": ParametrizedDistribution(th.distributions.Normal, loc=th.randn([]), scale=1.0)
})
distribution = approximation()
distribution.sample()
```

With the variational approximation in hand, we define a variational loss module which returns a stochastic estimate of the negative [evidence lower bound](https://en.wikipedia.org/wiki/Evidence_lower_bound) (ELBO) and the entropy of the distribution. However, we need to condition the model on the data before we can evaluate the loss.

```{code-cell} ipython3
from vbzero.nn import VariationalLoss
from vbzero.util import condition


conditioned = condition(poisson_regression, x=state["x"], y=state["y"], z=state["z"])
loss = VariationalLoss(conditioned, approximation)
loss()
```

The loss can be optimized with torch like any other loss function. However, to make life easier, vbzero provides a utility function for optimizing ELBOs with automatic learning rate reduction and stopping criteria based on the entropy of the distribution.

```{code-cell} ipython3
import os
from vbzero.util import train

result = train(loss, rtol=1.0 if "CI" in os.environ else 0.01)

fig, ax = plt.subplots()
ax.plot(result["entropies"])
ax.set_xlabel("epoch")
ax.set_ylabel("entropy")
fig.tight_layout()
```

Having optimized the variational approximation, we can compare posterior samples with the parameter values used to generate the data.

```{code-cell} ipython3
distributions = approximation()
samples = distributions.sample(500)

fig, axes = plt.subplots(2, 2)
ax = axes[0, 0]
ax.hist(samples["slope"].numpy())
ax.axvline(state["slope"], color="k", ls="--")
ax.set_xlabel(r"slope $\theta$")
for i, ax in enumerate(axes.ravel()[1:]):
    ax.hist(samples["intercepts"][:, i].numpy())
    ax.axvline(state["intercepts"][i], color="k", ls="--")
    ax.set_xlabel(f"intercept $b_{i + 1}$")
fig.tight_layout()
```

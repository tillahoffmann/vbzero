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

We consider a univariate Poisson regression model for count data, and we will estimate two parameters: the intercept and slope of the Poisson rate on the log scale. Let's define the model, generate some data, and visualize it.

```{code-cell} ipython3
from matplotlib import pyplot as plt
import torch as th
from vbzero.util import model, sample


@model(return_state=True)
def poisson_regression():
    n = 100
    x = sample("x", th.distributions.Normal(0, 1), sample_shape=n)
    intercept, slope = sample("theta", th.distributions.Normal(0, 1), sample_shape=2)
    log_rate = intercept + slope * x
    y = sample("y", th.distributions.Poisson(log_rate.exp()), sample_shape=n)
    

th.manual_seed(4)  # For reproducibility.
state = poisson_regression()

def plot_state(state):
    fig, ax = plt.subplots()
    ax.scatter(state["x"], state["y"])
    ax.set_yscale("log")
    ax.set_xlabel("covariate $x$")
    ax.set_ylabel("counts $y$")
    fig.tight_layout()
    
plot_state(state)
```

Having declared the generative model, let's define a variational approximation for the parameters theta with random starting points. A {class}`vbzero.nn.ParameterizedDistribution` is an easy way to define a distribution whose parameters can be optimized. Calling the module returns a distribution, and we can draw samples from it.

```{code-cell} ipython3
from vbzero.nn import ParametrizedDistribution

approximation = ParametrizedDistribution(th.distributions.MultivariateNormal, loc=th.randn(2), 
                                         scale_tril=th.eye(2))
distribution = approximation()
distribution.sample()
```

With the variational approximation in hand, we define a variational loss module which returns a stochastic estimate of the negative [evidence lower bound](https://en.wikipedia.org/wiki/Evidence_lower_bound) (ELBO) and the entropy of the distribution. However, we need to condition the model on the data before we can evaluate the loss.

```{code-cell} ipython3
from vbzero.nn import VariationalLoss
from vbzero.util import condition


conditioned = condition(poisson_regression, x=state["x"], y=state["y"])
loss = VariationalLoss(conditioned, {"theta": approximation})
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
distribution = approximation()
fig, ax = plt.subplots()
ax.scatter(*distribution.sample([200]).detach().T, marker=".", alpha=0.5, label="posterior samples")
ax.scatter(*state["theta"], label="true value")
ax.set_xlabel(r"intercept $\theta_1$")
ax.set_ylabel(r"slope $\theta_2$")
ax.legend()
fig.tight_layout()
```

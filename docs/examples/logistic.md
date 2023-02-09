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

# Logistic Regression

We consider a univariate logistic regression model for binary data with random effects, and we will estimate two parameters: the intercepts $b$ (one for each of $k$ groups) and slope $\theta$ of the log odds. In particular,
$$\begin{align}
x_i &\sim \mathsf{Normal}\left(0, 1\right)\text{ for }i\in\left\{1,\ldots,n\right\}\\
z_i &\sim \mathsf{Categorical}\left(\left\{k^{-1},\ldots\right\}\right)\\
y_i &\sim \mathsf{Bernoulli}\left(\text{expit}\left[b_{z_i} + \theta x_i\right]\right),
\end{align}$$
where $x$ are covariates, $z$ are group labels, and $y$ are binary outcomes.

Let's define the model, generate some data, and visualize it.

```{code-cell} ipython3
import matplotlib as mpl
from matplotlib import pyplot as plt
import numpy as np
import torch as th
from vbzero.util import model, sample


@model(return_state=True)
def logistic_regression():
    """
    Logistic regression model with random effects.
    """
    # Number of observations and number of groups.
    n = 100
    k = 5
    # Covariates and group assignments.
    x = sample("x", th.distributions.Normal(0, 1), sample_shape=n)
    z = sample("z", th.distributions.Categorical(logits=th.zeros(k)), sample_shape=n)
    # Model parameters of interest.
    intercepts = sample("intercepts", th.distributions.Normal(0, 3), sample_shape=k)
    slope = sample("slope", th.distributions.Normal(0, 1))
    # The ... notation ensures we can pass in batches of samples for posterior predictive
    # replication.
    logits = intercepts[..., z] + slope[..., None] * x
    y = sample("y", th.distributions.Bernoulli(logits=logits))


th.manual_seed(5)  # For reproducibility.
state = logistic_regression()

# Visualize the samples (we add some jitter to visually distinguish members of different groups).
fig, ax = plt.subplots()
lin = th.linspace(state["x"].min(), state["x"].max(), 50)
k, = state["intercepts"].size()
for i, intercept in enumerate(state["intercepts"]):
    line, = ax.plot(lin, th.special.expit(intercept + state["slope"] * lin), ls="--")
    fltr = state["z"] == i
    ax.scatter(state["x"][fltr], state["y"][fltr] + 0.02 * (i - k // 2), color=line.get_color(), marker=".")
ax.set_xlabel("covariate $x$")
ax.set_ylabel(r"probability $p\left(y=1\mid x\right)$")
fig.tight_layout()
```

Having declared the generative model, let's define a variational approximation for the parameters theta with random starting points. A {class}`vbzero.nn.ParameterizedDistribution` is an easy way to define a distribution whose parameters can be optimized. Calling the module returns a distribution, and we can draw samples from it.

```{code-cell} ipython3
from vbzero.nn import ParameterizedDistribution, ParameterizedDistributionDict

k, = state["intercepts"].size()
approximation = ParameterizedDistributionDict({
    "intercepts": ParameterizedDistribution(th.distributions.Normal, loc=th.randn(k),
                                           scale=th.ones(k)),
    "slope": ParameterizedDistribution(th.distributions.Normal, loc=th.randn([]), scale=1.0)
})
distribution = approximation()
distribution.sample()
```

With the variational approximation in hand, we define a variational loss module which returns a stochastic estimate of the negative [evidence lower bound](https://en.wikipedia.org/wiki/Evidence_lower_bound) (ELBO) and the entropy of the distribution. However, we need to condition the model on the data before we can evaluate the loss.

```{code-cell} ipython3
from vbzero.nn import VariationalLoss
from vbzero.util import condition


conditioned = condition(logistic_regression, state["x", "y", "z"])
loss = VariationalLoss(conditioned, approximation)
loss()
```

The loss can be optimized with torch like any other loss function. However, to make life easier, vbzero provides a utility function for optimizing ELBOs with automatic learning rate reduction and stopping criteria based on the entropy of the distribution.

```{code-cell} ipython3
import os
from vbzero.util import train

steps = 10 if "CI" in os.environ else 1000
optim = th.optim.Adam(approximation.parameters(), lr=0.01)
losses = []
for _ in range(steps):
    optim.zero_grad()
    loss_value, _ = loss()
    loss_value.backward()
    optim.step()
    losses.append(loss_value.item())


fig, ax = plt.subplots()
ax.plot(losses)
ax.set_xlabel("step")
ax.set_ylabel("loss")
fig.tight_layout()
```

Having optimized the variational approximation, we can compare posterior samples with the parameter values used to generate the data.

```{code-cell} ipython3
distributions = approximation()
samples = distributions.sample(500)

fig, (ax1, ax2) = plt.subplots(1, 2)
ax1.hist(samples["slope"], density=True)
ax1.axvline(state["slope"], color="k", ls="--")
ax1.set_xlabel(r"slope $\theta$")
ax1.set_ylabel(r"posterior density $p\left(\theta\mid x, y, z\right)$")

lu = state["intercepts"].min(), state["intercepts"].max()
ax2.plot(lu, lu, color="k", ls="--")
ax2.errorbar(state["intercepts"], samples["intercepts"].mean(axis=0),
             1.96 * samples["intercepts"].std(axis=0), ls="none", color="silver")
ax2.scatter(state["intercepts"], samples["intercepts"].mean(axis=0),
            c=[f"C{i}" for i in range(k)], edgecolor="w", zorder=9)
```

Let's check our model using posterior predictive replication. We generate a confusion matrix for observed and replicated binary outcomes, one for each group. We observe a strong diagonal, indicating that our model has indeed learned something informative.

```{code-cell} ipython3
fig, axes = plt.subplots(2, 3, sharex=True, sharey=True)
axes = axes.ravel()

replicates = condition(logistic_regression, state["x", "z"], samples)()
a, b = np.broadcast_arrays(state["y"], replicates["y"])

for i, ax in enumerate(axes[:-1]):
    fltr = state["z"] == i
    confusion, *_ = np.histogram2d(a[:, fltr].ravel(), b[:, fltr].ravel(), bins=2)
    ax.imshow(confusion, origin="lower")
    ax.set_title(f"group {i + 1}")

confusion, *_ = np.histogram2d(a.ravel(), b.ravel(), bins=2)
ax = axes[-1]
ax.imshow(confusion, origin="lower")
ax.set_xticks([0, 1])
ax.set_yticks([0, 1])
ax.set_title("total")
fig.tight_layout()
```

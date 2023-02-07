🦾 vbzero
=========

.. image:: https://github.com/tillahoffmann/vbzero/actions/workflows/main.yml/badge.svg
  :target: https://github.com/tillahoffmann/vbzero/actions/workflows/main.yml

.. toctree::
  :hidden:

  docs/interface/interface
  docs/examples/poisson_regression

vbzero is a minimal stochastic variational inference framework for `torch <https://pytorch.org>`__ with an interface similar to `pyro <https://pyro.ai>`__.

Models are declared as python functions using :func:`vbzero.util.sample` statements. For example, the following snippet encodes the standard biased coin example.

.. doctest::

  >>> import torch as th
  >>> from vbzero.util import condition, LogProb, model, sample, State

  >>> @model
  ... def biased_coin():
  ...     proba = sample("proba", th.distributions.Beta(1, 1))
  ...     x = sample("x", th.distributions.Bernoulli(proba), sample_shape=10)
  ...     return proba, x

  >>> th.manual_seed(1)  # For reproducibility.
  <torch...>
  >>> biased_coin()
  (tensor(0.6003), tensor([1., 1., 0., ...]))


If provided, state information is encapsulated in a :class:`vbzero.util.State`. For example, we can access all variables as follows.

.. doctest::

  >>> th.manual_seed(1)  # For reproducibility.
  <torch...>
  >>> with State() as state:
  ...     biased_coin()
  (tensor(0.6003), tensor([1., 1., 0., ...]))
  >>> state
  {'proba': tensor(0.6003), 'x': tensor([1., 1., 0., ...])}


The :class:`vbzero.util.LogProb` context can be used to evaluate the likelihood of a sample under the model.

.. doctest::

  >>> conditioned = condition(biased_coin, state)
  >>> with LogProb() as log_prob:
  ...     conditioned()
  (tensor(0.6003), tensor([1., 1., 0., ...]))
  >>> log_prob
  {'proba': tensor(0.), 'x': tensor([-0.5103, -0.5103, -0.9171, ...])}

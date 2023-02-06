🦾 vbzero
=========

.. image:: https://github.com/tillahoffmann/vbzero/actions/workflows/main.yml/badge.svg
  :target: https://github.com/tillahoffmann/vbzero/actions/workflows/main.yml

vbzero is a minimal stochastic variational inference framework for `torch <https://pytorch.org>`__ with an interface similar to `pyro <https://pyro.ai>`__.

Models are declared as python functions using :func:`vbzero.util.sample` statements. For example, the following snippet encodes the standard biased coin example.

.. doctest::

  >>> import torch as th
  >>> from vbzero.util import sample, StochasticContext

  >>> def model():
  ...     proba = sample("proba", th.distributions.Beta(1, 1))
  ...     x = sample("x", th.distributions.Bernoulli(proba), sample_shape=10)
  ...     return proba, x

  >>> th.manual_seed(1)  # For reproducibility.
  <torch...>
  >>> model()
  (tensor(0.6003), tensor([1., 1., 0., ...]))


If provided, state information is encapsulated in a :class:`vbzero.util.StochasticContext`. For example, we can access all variables as follows.

.. doctest::

  >>> th.manual_seed(1)  # For reproducibility.
  <torch...>
  >>> with StochasticContext() as context:
  ...     model()
  (tensor(0.6003), tensor([1., 1., 0., ...]))
  >>> draw = context.values
  >>> draw
  {'proba': tensor(0.6003), 'x': tensor([1., 1., 0., ...])}


Contexts can also be used to evaluate the likelihood of a sample under the model.

.. doctest::

  >>> with StochasticContext(draw, mode="log_prob") as context:
  ...     model()
  (tensor(0.6003), tensor([1., 1., 0., ...]))
  >>> context.log_probs
  {'proba': tensor(0.), 'x': tensor([-0.5103, -0.5103, -0.9171, ...])}

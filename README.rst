🦾 vbzero
=========

.. image:: https://github.com/tillahoffmann/vbzero/actions/workflows/main.yml/badge.svg
  :target: https://github.com/tillahoffmann/vbzero/actions/workflows/main.yml

.. toctree::
  :hidden:

  docs/examples/examples
  docs/interface/interface

vbzero is a minimal stochastic variational inference framework for `torch <https://pytorch.org>`__ with an interface similar to `pyro <https://pyro.ai>`__.

Models are declared as python functions using :func:`vbzero.util.sample` statements. For example, the following snippet encodes the standard biased coin example.

.. doctest::

  >>> import torch as th
  >>> from vbzero.util import model, sample

  >>> @model
  ... def biased_coin():
  ...     proba = sample("proba", th.distributions.Beta(1, 1))
  ...     x = sample("x", th.distributions.Bernoulli(proba), sample_shape=10)
  ...     return proba, x

  >>> th.manual_seed(1)  # For reproducibility.
  <torch...>
  >>> biased_coin()
  (tensor(0.6003), tensor([1., 1., 0., ...]))

State Management
----------------

If provided, state information is encapsulated in a :class:`vbzero.util.State`. For example, we can access all variables as follows.

.. doctest::

  >>> from vbzero.util import State

  >>> th.manual_seed(1)  # For reproducibility.
  <torch...>
  >>> with State() as state:
  ...     biased_coin()
  (tensor(0.6003), tensor([1., 1., 0., ...]))
  >>> state
  {'proba': tensor(0.6003), 'x': tensor([1., 1., 0., ...])}

This allows different datasets and models to be handled within the same process. If a :class:`vbzero.util.State` context is not active, a state will be created implicitly. It can be retrieved by calling :meth:`vbzero.util.State.get_instance` within the model, but all state will be discarded after the model invocation unless it is created explicitly as above.

The :class:`vbzero.util.LogProb` context can be used to evaluate the likelihood of a sample under the model.

.. doctest::

  >>> from vbzero.util import LogProb

  >>> with state, LogProb() as log_prob:
  ...     biased_coin()
  (tensor(0.6003), tensor([1., 1., 0., ...]))
  >>> log_prob
  {'proba': tensor(0.), 'x': tensor([-0.5103, -0.5103, -0.9171, ...])}

Including :code:`state` in the :code:`with` statement ensures that all variables are defined and the likelihood can be evaluated. We consider counterfactuals by modifying the state directly or using the :func:`vbzero.util.condition` statement.

.. doctest::

  >>> from vbzero.util import condition

  >>> conditioned = condition(biased_coin, proba=th.as_tensor(0.5))
  >>> with state, LogProb() as log_prob:
  ...     conditioned()
  (tensor(0.5000), tensor([1., 1., 0., ...]))
  >>> log_prob
  {'proba': tensor(0.), 'x': tensor([-0.6931, -0.6931, -0.6931, ...])}
  >>> state
  {'proba': tensor(0.5000), 'x': tensor([1., 1., 0., ...])}

.. note::

  The state is modified by invoking the :code:`conditioned` model. Use :meth:`vbzero.util.State.copy` to create a shallow copy and prevent it from being modified. In general, we recommend not sharing state across model invocations.

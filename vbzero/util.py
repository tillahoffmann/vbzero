"""
util
====
"""

from collections.abc import Mapping, MutableMapping
import functools as ft
from numbers import Integral
import numpy as np
import torch as th
from torch.distributions import Distribution
from torch.nn import Module
from torch.optim import Adam, Optimizer
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
from typing import Any, Callable, Iterable, Optional, Union, Type


def normalize_shape(shape: Optional[Union[Integral, tuple, th.Size]]) -> th.Size:
    """
    Normalize a sample shape akin to numpy.

    Args:
        shape: Possibly unnormalized shape, e.g., an integer sample size, or :code:`None` to
            indicate single samples.

    Returns:
        Normalized shape as a tuple or :class:`torch.Size`.
    """
    if shape is None:
        return ()
    if isinstance(shape, Integral):
        return (shape,)
    return shape


class SingletonContextMixin:
    """
    Baseclass for singleton contexts. Inheriting classes must override :meth:`sample` which handles
    drawing of random variables for the context. The class-level :attr:`ORDER` attribute determines
    the default application order of :meth:`sample` statements in relation to other contexts.

    Args:
        order: Application order of the context (defaults to :attr:`ORDER`).
    """
    INSTANCE: "SingletonContextMixin" = None
    ORDER: Integral = 0

    def __init__(self, order: Optional[Integral] = None) -> None:
        self.order = self.ORDER if order is None else order

    def __enter__(self):
        cls = self.__class__
        # "Reentrant" context manager.
        if cls.is_active() and cls.INSTANCE is not self:
            raise RuntimeError(f"a different {cls} context is already active")
        cls.INSTANCE = self
        return self

    def __exit__(self, *args):
        self.__class__.INSTANCE = None

    @classmethod
    def get_instance(cls, *args, strict: bool = True, **kwargs) -> "SingletonContextMixin":
        """
        Get the active context instance or create a new one if :code:`strict` is false-y.

        Args:
            *args: Positional arguments passed to the constructor.
            strict: Enforce that a context is active instead of returning a new instance.
            **kwargs: Keyword arguments passed to the constructor.
        """
        if cls.is_active():
            return cls.INSTANCE
        if strict:
            raise RuntimeError(f"{cls} context is not active")
        return cls(*args, **kwargs)

    @classmethod
    def is_active(cls) -> bool:
        """
        Return whether the context is active.
        """
        return cls.INSTANCE is not None

    def _evaluate_distribution(self, dist_cls: Union[Distribution, Type[Distribution]], *args,
                               sample_shape: Optional[th.Size] = None, **kwargs) -> Distribution:
        """
        Helper function to obtain a distribution instance for deferred evaluation.
        See :func:`sample` for details.
        """
        distribution = dist_cls if isinstance(dist_cls, Distribution) else dist_cls(*args, **kwargs)
        # Try to automatically infer the batch shape if not given.
        sample_shape = distribution.batch_shape if sample_shape is None else \
            normalize_shape(sample_shape)
        return distribution.expand(sample_shape)

    def sample(self, name: str, dist_cls: Union[Distribution, Type[Distribution]], *args,
               sample_shape: Optional[th.Size] = None, **kwargs) -> Any:
        raise NotImplementedError


class State(SingletonContextMixin, dict):
    """
    Dictionary-like context for managing the state of a model invocation. Values can only be set
    once to ensure consistent state for each model invocation.
    """
    # We want State sampling statements to be executed at an intermediate level because they modify
    # the state. We may want to use other contexts before or after the state is modified.
    ORDER = 100

    def __init__(self, *args, order: Optional[Integral] = None, **kwargs) -> None:
        super().__init__(order)
        dict.__init__(self, *args, **kwargs)

    def __getitem__(self, name: Union[str, Iterable[str]]) -> th.Tensor:
        if isinstance(name, str):
            return dict.__getitem__(self, name)
        return {key: dict.__getitem__(self, key) for key in name}

    def __setitem__(self, key: str, value: th.Tensor) -> None:
        if key in self:
            raise KeyError(f"key {key} has already been set")
        return super().__setitem__(key, value)

    def __delitem__(self, key: str) -> None:
        raise KeyError(f"keys cannot be removed; attempted {key}")

    def update(self, value: Mapping, **kwargs) -> None:
        return MutableMapping.update(self, value, **kwargs)

    def sample(self, name: str, dist_cls: Union[Distribution, Type[Distribution]], *args,
               sample_shape: Optional[th.Size] = None, **kwargs) -> Any:
        if (x := self.get(name)) is not None:
            return x
        dist = self._evaluate_distribution(dist_cls, *args, **kwargs, sample_shape=sample_shape)
        self[name] = dist.sample()


class LogProb(SingletonContextMixin, dict):
    """
    Context for evaluating and storing log probabilities.
    """
    # We want to evaluate log probabilities before any variables might be sampled by the State and
    # give it a low ORDER.
    ORDER = 0

    def __init__(self, *args, order: Optional[Integral] = None, **kwargs) -> None:
        super().__init__(order)
        dict.__init__(self, *args, **kwargs)

    def sample(self, name: str, dist_cls: Union[Distribution, Type[Distribution]], *args,
               sample_shape: Optional[th.Size] = None, **kwargs) -> Any:
        if name in self:
            raise RuntimeError(f"log probability has already been evaluated for variable {name}")
        # We need all variables to be present to evaluate the log probability.
        if name not in State.INSTANCE:
            raise ValueError(f"cannot evaluate log probability because variable {name} is missing")
        x = State.INSTANCE[name]
        dist = self._evaluate_distribution(dist_cls, *args, **kwargs, sample_shape=sample_shape)
        expected_shape = dist.batch_shape + dist.event_shape
        if expected_shape != x.shape:
            raise ValueError(f"expected shape {expected_shape} for variable {name} but got "
                             f"{x.shape}")
        self[name] = dist.log_prob(x)


def sample(name: str, dist_cls: Union[Distribution, Type[Distribution]], *args,
           sample_shape: Optional[th.Size] = None, **kwargs) -> Any:
    """
    Sample a value from a distribution.

    Args:
        name: Name of the variable to sample.
        dist_cls: Distribution or callable that returns a distribution given :code:`*args` and
            :code:`**kwargs`.
        *args: Positional arguments passed to :code:`dist_cls` if it is not a distribution.
        **kwargs: Keyword arguments passed to :code:`dist_cls` if it is not a distribution.
        sample_shape: Shape of the sample to draw.

    Returns:
        Value of the sampled variable.
    """
    if name == "order":
        raise ValueError("`order` is a reserved name; choose another")
    # State is special. We ensure it's there every time.
    state = State.get_instance(strict=True)
    # First validate the state for all active contexts and then apply them.
    contexts = [cls.INSTANCE for cls in SingletonContextMixin.__subclasses__() if cls.is_active()]
    for cls in sorted(contexts, key=lambda context: context.order):
        cls.INSTANCE.sample(name, dist_cls, *args, **kwargs, sample_shape=sample_shape)
    # Return the variable value.
    return state[name]


def hyperparam(name: str, *names: str) -> Any:
    """
    Get a hyperparameter by name.

    Args:
        name: Name of the hyperparameter.

    Returns:
        Value of the hyperparameter.
    """
    state = State.get_instance()
    if names:
        values = [name]
        values.extend(state[name] for name in names)
        return names
    return state[name]


def condition(func: Callable, *values: Iterable[dict], **kwvalues: dict) -> Callable:
    """
    Condition a model with the given values.

    Args:
        *values: Mappings of values to condition on.
        **kwargs: Mapping of values to condition on as keyword arguments.

    Returns:
        Model conditioned on the provided values.
    """
    @ft.wraps(func)
    def _wrapper(*args, **kwargs) -> Any:
        with State.get_instance(strict=False) as state:
            for value in values:
                state.update(value)
            state.update(kwvalues)
            return func(*args, **kwargs)
    return _wrapper


def maybe_aggregate(value: dict[str, th.Tensor], aggregate: bool) -> Any:
    """
    Maybe aggregate the values of a dictionary.

    Args:
        value: Mapping whose values are tensors.
        aggregate: Aggregate the tensors.

    Returns:
        Aggregated tensors if :code:`aggregate` is truth-y or the input value.
    """
    if aggregate:
        return sum(x.sum() for x in value.values())
    return value


def train(loss: Module, loss_args: Optional[tuple] = None, loss_kwargs: Optional[dict] = None,
          optim: Optional[Optimizer] = None, scheduler: Optional[Any] = None,
          num_steps_per_epoch: int = 1_000, num_epochs: Optional[int] = None,
          progress: Union[bool, tqdm] = True, atol: float = 0, rtol=0.001) -> dict:
    """
    Minimize a variational loss with sensible defaults.

    Args:
        loss: Variational loss to optimize.
        optim: Optimizer (defaults to an :class:`torch.optim.Adam` optimizer).
        scheduler: Learning rate scheduler (defaults to a
            :class:`torch.optim.scheduler.ReduceLROnPlateau` scheduler).
        num_steps_per_epoch: Optimization steps per epoch.
    """
    if optim is None:
        optim = Adam(loss.parameters())
    if scheduler is None and scheduler != "none":
        scheduler = ReduceLROnPlateau(optim, verbose=True)
    if progress is True:
        progress = tqdm(total=num_epochs)
    loss_args = loss_args or []
    loss_kwargs = loss_kwargs or {}

    values = {}
    previous_entropy = None
    while True:
        # Run one epoch.
        current_loss = 0
        current_entropy = 0
        for _ in range(num_steps_per_epoch):
            optim.zero_grad()
            loss_value: th.Tensor
            entropy: th.Tensor
            loss_value, entropy = loss(*loss_args, **loss_kwargs)
            loss_value.backward()
            optim.step()

            current_loss += loss_value.item()
            current_entropy += entropy.item()

        current_loss /= num_steps_per_epoch
        values.setdefault("losses", []).append(current_loss)
        current_entropy /= num_steps_per_epoch
        values.setdefault("entropies", []).append(current_entropy)

        if progress is not None:
            progress.update()
            progress.set_description(f"loss = {loss_value:.3e}")

        # Determine whether we've reached the maximum number of epochs.
        if num_epochs and len(values["losses"]) == num_epochs:
            break

        # Evaluate convergence based on entropy of the distribution.
        if previous_entropy is not None \
                and np.isclose(previous_entropy, current_entropy, rtol=rtol, atol=atol):
            break
        previous_entropy = current_entropy

        # Update the learning rate.
        scheduler.step(current_loss)

    return values


def model(func: Optional[Callable] = None, *, return_state: bool = False) -> Callable:
    """
    Handle state for a callable model.

    Args:
        return_state: Return the state of the decorated model on invocation.

    Returns:
        Model with state handling.
    """
    if func:  # Directly wrap the callable.
        @ft.wraps(func)
        def _wrapper(*args, **kwargs) -> Any:
            with State.get_instance(strict=False) as state:
                result = func(*args, **kwargs)
            if return_state:
                return state if result is None else (result, state)
            return result
        return _wrapper
    else:  # Apply keyword arguments.
        return ft.partial(model, return_state=return_state)

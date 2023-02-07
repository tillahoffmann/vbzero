from collections.abc import MutableMapping
import functools as ft
from numbers import Integral
import numpy as np
import torch as th
from torch.distributions import Distribution
from torch.nn import Module
from torch.optim import Adam, Optimizer
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
from typing import Any, Callable, Optional, Union, Type


def normalize_shape(shape: Optional[th.Size]) -> th.Size:
    """
    Normalize a sample shape akin to numpy.
    """
    if shape is None:
        return ()
    if isinstance(shape, Integral):
        return (shape,)
    return shape


class SingletonContextMixin:
    """
    Baseclass for singleton contexts.
    """
    INSTANCE = None

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
    def get_instance(cls, *args, **kwargs):
        return cls.INSTANCE if cls.is_active() else cls(*args, **kwargs)

    @classmethod
    def is_active(cls):
        return cls.INSTANCE is not None

    def _evaluate_distribution(self, dist_cls: Union[Distribution, Type[Distribution]], *args,
                               sample_shape: Optional[th.Size] = None, **kwargs) -> Distribution:
        distribution = dist_cls if isinstance(dist_cls, Distribution) else dist_cls(*args, **kwargs)
        return distribution.expand(normalize_shape(sample_shape))

    def sample(self, name: str, dist_cls: Union[Distribution, Type[Distribution]], *args,
               sample_shape: Optional[th.Size] = None, **kwargs) -> Any:
        raise NotImplementedError


class State(SingletonContextMixin, dict):
    """
    Dictionary-like context for managing the state of a model invocation. Values can only be set
    once to ensure consistent state for each model invocation.
    """
    # Use `MutableMapping`'s implementation in terms of __setitem__ which we have overridden.
    update = MutableMapping.update

    def __setitem__(self, key: str, value: th.Tensor) -> None:
        if key in self:
            raise KeyError(f"key {key} has already been set")
        return super().__setitem__(key, value)

    def __delitem__(self, key: str) -> None:
        raise KeyError(f"keys cannot be removed; attempted {key}")

    def sample(self, name: str, dist_cls: Union[Distribution, Type[Distribution]], *args,
               sample_shape: Optional[th.Size] = None, **kwargs) -> Any:
        if (x := self.get(name)) is not None:
            return x
        dist = self._evaluate_distribution(dist_cls, *args, **kwargs, sample_shape=sample_shape)
        self[name] = dist.sample()


class LogProb(SingletonContextMixin, dict):
    """
    Context for storing log probabilities.
    """
    def sample(self, name: str, dist_cls: Union[Distribution, Type[Distribution]], *args,
               sample_shape: Optional[th.Size] = None, **kwargs) -> Any:
        if name in LogProb.INSTANCE:
            raise RuntimeError(f"log probability has already been evaluated for variable {name}")
        if (x := State.INSTANCE.get(name)) is None:
            raise ValueError(f"cannot evaluate log probability because variable {name} is missing")
        dist = self._evaluate_distribution(dist_cls, *args, **kwargs, sample_shape=sample_shape)
        expected_shape = dist.batch_shape + dist.event_shape
        if expected_shape != x.shape:
            raise ValueError(f"expected shape {expected_shape} for variable {name} but got "
                             f"{x.shape}")
        self[name] = dist.log_prob(x)


def hyperparam(name: str) -> Any:
    """
    Get a hyperparameter.
    """
    if not State.is_active():
        raise RuntimeError("state context is not active; decorate your model with @model or create "
                           "an explicit state")
    return State.INSTANCE[name]


def sample(name: str, dist_cls: Union[Distribution, Type[Distribution]], *args,
           sample_shape: Optional[th.Size] = None, **kwargs) -> Any:
    """
    Sample a value from a distribution or evaluate its log probability under the distribution.

    Args:
        name: Name of the variable to sample.
        dist_cls: Distribution or callable that returns a distribution given :code:`*args` and
            :code:`**kwargs`.
        *args: Positional arguments passed to :code:`dist_cls` if it is not a distribution.
        **kwargs: Keyword arguments passed to :code:`dist_cls` if it is not a distribution.
        sample_shape: Shape of the sample to draw.
        context: Stochastic context for sampling or log probability evaluation. If :code:`context`
            is not given, random variables are drawn without reference to any state.
    """
    # We always need to have state.
    if not State.is_active():
        raise RuntimeError("state context is not active; decorate your model with @model or create "
                           "an explicit state")
    # Apply all the contexts.
    for cls in SingletonContextMixin.__subclasses__():
        if cls.is_active():
            cls.INSTANCE.sample(name, dist_cls, *args, sample_shape=sample_shape, **kwargs)
    # Return the variable value.
    return State.INSTANCE[name]


def condition(func: Callable, values: Optional[dict] = None, **kwvalues) -> Callable:
    """
    Condition a model with the given values.
    """
    @ft.wraps(func)
    def _wrapper(*args, **kwargs) -> Any:
        with State.get_instance() as state:
            if values:
                state.update(values)
            state.update(kwvalues)
            return func(*args, **kwargs)
    return _wrapper


def maybe_aggregate(value: dict[str, th.Tensor], aggregate: bool) -> Any:
    """
    Aggregate the values of a dictionary.
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
        return_state: Return the state of the decorated model.

    Returns: Model with state handling.
    """
    if func:  # Directly wrap the callable.
        @ft.wraps(func)
        def _wrapper(*args, **kwargs) -> Any:
            with State.get_instance() as state:
                result = func(*args, **kwargs)
            if return_state:
                return state if result is None else (result, state)
            return result
        return _wrapper
    else:  # Apply keyword arguments.
        return ft.partial(model, return_state=return_state)

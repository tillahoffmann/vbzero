"""
util
====
"""

from __future__ import annotations
import functools as ft
from numbers import Integral
import numpy as np
import torch as th
from torch.distributions import Distribution
from torch.nn import Module
from torch.optim import Adam, Optimizer
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
from typing import Any, Callable, Iterable, List, Optional, Union, Tuple, Type


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
    return th.Size(shape)


class State(dict):
    STACK: List[State] = []

    def __enter__(self) -> State:
        self.STACK.append(self)
        return self

    def __exit__(self, *args) -> None:
        if (other := self.STACK.pop()) is not self:
            raise RuntimeError(f"incosistent stack; expected {self} but got {other}")

    @classmethod
    def get_instance(cls, strict: bool = True) -> State:
        if cls.STACK:
            return cls.STACK[-1]
        if strict:
            raise RuntimeError(f"no active {cls}")
        return cls()

    def __getitem__(self, key: Union[Tuple[str], str]) -> Any:
        if isinstance(key, str):
            return super().__getitem__(key)
        # Cannot use super() in comprehension (https://stackoverflow.com/a/31895448/1150961).
        return {x: dict.__getitem__(self, x) for x in key}

    def __or__(self, other: State) -> State:
        new = self.copy()
        new.update(other)
        return new

    def copy(self) -> State:
        new = State()
        new.update(self)
        return new


class TraceMixin:
    INSTANCE: Optional[TraceMixin] = None

    def __enter__(self) -> State:
        if TraceMixin.INSTANCE is not None:
            raise RuntimeError(f"cannot activate {self} because {TraceMixin.INSTANCE} is active")
        TraceMixin.INSTANCE = self
        return TraceMixin.INSTANCE

    def __exit__(self, *args) -> None:
        if TraceMixin.INSTANCE is not self:
            raise RuntimeError(f"inconsistent context; expected {self} but got "
                               f"{TraceMixin.INSTANCE}")
        TraceMixin.INSTANCE = None

    @staticmethod
    def _evaluate_distribution(dist_cls: Union[Distribution, Type[Distribution]], *args,
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

    def __call__(self, state: State, name: str, dist_cls: Union[Distribution, Type[Distribution]],
                 *args, sample_shape: Optional[th.Size] = None, **kwargs) -> None:
        raise NotImplementedError


class Sample(TraceMixin):
    """
    Sample random variables.
    """
    def __call__(self, state: State, name: str, dist_cls: Union[Distribution, Type[Distribution]],
                 *args, sample_shape: Optional[th.Size] = None, **kwargs) -> None:
        if (x := state.get(name)) is not None:
            return x
        dist = self._evaluate_distribution(dist_cls, *args, **kwargs, sample_shape=sample_shape)
        state[name] = dist.sample()


class LogProb(TraceMixin, dict):
    """
    Evaluate and store log probabilities.
    """
    def __call__(self, state: State, name: str, dist_cls: Union[Distribution, Type[Distribution]],
                 *args, sample_shape: Optional[th.Size] = None, **kwargs) -> None:
        if name in self:
            raise RuntimeError(f"log probability has already been evaluated for variable {name}")
        # We need all variables to be present to evaluate the log probability.
        if (x := state.get(name)) is None:
            raise ValueError(f"cannot evaluate log probability because variable {name} is missing")
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
    trace = Sample() if TraceMixin.INSTANCE is None else TraceMixin.INSTANCE
    state = State.get_instance()
    trace(state, name, dist_cls, *args, **kwargs, sample_shape=sample_shape)
    return state[name]


def hyperparam(name: str, *names: str) -> Any:
    """
    Get hyperparameters by name.

    Args:
        name: Name of the hyperparameter.

    Returns:
        Value of the hyperparameter.
    """
    state = State.get_instance()
    if names:
        return tuple(state[x] for x in (name, *names))
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
        # We make a copy of the state so we don't modify the original context.
        with State.get_instance(strict=False).copy() as state:
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

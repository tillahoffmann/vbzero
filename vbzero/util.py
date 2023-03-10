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
from typing import Any, Callable, Dict, Iterable, Optional, Union, Tuple, Type


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


class SingletonContextMixin:
    """
    Mixin to manage singletons, one for each unique :attr:`SINGLETON_KEY`. Inheriting classes must
    override :attr:`SINGLETON_KEY` to declare the group they belong to.
    """
    INSTANCES: Dict[str, SingletonContextMixin] = {}
    SINGLETON_KEY = None

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.counter = 0

    def __enter__(self) -> SingletonContextMixin:
        if self.SINGLETON_KEY is None:
            raise RuntimeError(f"inheriting class {self.__class__} must override SINGLETON_KEY")
        other = self.INSTANCES.setdefault(self.SINGLETON_KEY, self)
        if other is not self:
            raise RuntimeError(f"singleton {other} with key {self.SINGLETON_KEY} is already active")
        self.counter += 1
        return self

    def __exit__(self, *args) -> None:
        self.counter -= 1
        if self.counter == 0 and (other := self.INSTANCES.pop(self.SINGLETON_KEY)) is not self:
            raise ValueError(f"expected singleton {self} but got {other}")

    @classmethod
    def get_instance(cls) -> Optional[SingletonContextMixin]:
        """
        Get the active singleton context if available.
        """
        if (instance := cls.INSTANCES.get(cls.SINGLETON_KEY)) is not None:
            return instance


class State(SingletonContextMixin, dict):
    """
    State for model invocations.
    """
    SINGLETON_KEY = "state"

    def __or__(self, other: State) -> State:
        new = self.copy()
        new.update(other)
        return new

    def copy(self) -> State:
        new = State()
        new.update(self)
        return new

    def __getitem__(self, key: Union[str, Tuple[str]]) -> Union[th.Tensor, Dict[str, th.Tensor]]:
        if isinstance(key, str):
            return super().__getitem__(key)
        return {x: dict.__getitem__(self, x) for x in key}


class TraceMixin(SingletonContextMixin):
    """
    Mixin for tracing the invocation of models, e.g., to :class:`Sample` or evaluate
    :class:`LogProb`.
    """
    SINGLETON_KEY = "trace"

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

    def sample(self, state: State, name: str, dist_cls: Union[Distribution, Type[Distribution]],
               *args, sample_shape: Optional[th.Size] = None, **kwargs) -> None:
        raise NotImplementedError

    def record(self, state: State, name: str, value: Any) -> None:
        return value


class Sample(TraceMixin):
    """
    Sample random variables.
    """
    def sample(self, state: State, name: str, dist_cls: Union[Distribution, Type[Distribution]],
               *args, sample_shape: Optional[th.Size] = None, **kwargs) -> None:
        if (x := state.get(name)) is not None:
            return x
        dist = self._evaluate_distribution(dist_cls, *args, **kwargs, sample_shape=sample_shape)
        state[name] = x = dist.sample()
        return x

    def record(self, state: State, name: str, value: Any) -> None:
        if name in state:
            raise ValueError(f"{name} is already set")
        state[name] = value
        return value


class LogProb(TraceMixin, dict):
    """
    Evaluate and store log probabilities.
    """
    def sample(self, state: State, name: str, dist_cls: Union[Distribution, Type[Distribution]],
               *args, sample_shape: Optional[th.Size] = None, **kwargs) -> None:
        if name in self:
            raise RuntimeError(f"log probability has already been evaluated for variable {name}")
        # We need all variables to be present to evaluate the log probability.
        x: th.Tensor
        if (x := state.get(name)) is None:
            raise ValueError(f"cannot evaluate log probability because variable {name} is missing")
        dist = self._evaluate_distribution(dist_cls, *args, **kwargs, sample_shape=sample_shape)
        expected_shape = dist.batch_shape + dist.event_shape
        if expected_shape != x.shape:
            raise ValueError(f"expected shape {expected_shape} for variable {name} but got "
                             f"{x.shape}")
        self[name] = dist.log_prob(x)
        return x

    def __exit__(self, exception_type: Optional[Type[Exception]], *args) -> None:
        super().__exit__(*args)
        # Only raise this exception if everything else worked out fine.
        if not self and not exception_type:
            raise ValueError("no log probs evaluated; did you invoke the model?")


def sample(name: str, dist_cls: Union[Distribution, Type[Distribution]], *args,
           sample_shape: Optional[th.Size] = None, **kwargs) -> th.Tensor:
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
    trace = Sample() if (x := TraceMixin.get_instance()) is None else x
    if (state := State.get_instance()) is None:
        raise RuntimeError("no active state")
    return trace.sample(state, name, dist_cls, *args, **kwargs, sample_shape=sample_shape)


def record(name: str, value: th.Tensor) -> th.Tensor:
    """
    Record a value that is not a random variable.
    """
    trace = Sample() if (x := TraceMixin.get_instance()) is None else x
    if (state := State.get_instance()) is None:
        raise RuntimeError("no active state")
    return trace.record(state, name, value)


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
        # Get the state or create a new one and update it with the given values.
        state = State() if (x := State.get_instance()) is None else x
        for value in values:
            state.update(value)
        state.update(kwvalues)
        with state:
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
            state = State() if (x := State.get_instance()) is None else x
            with state:
                result = func(*args, **kwargs)
            if return_state:
                return state if result is None else (result, state)
            return result
        return _wrapper
    else:  # Apply keyword arguments.
        return ft.partial(model, return_state=return_state)

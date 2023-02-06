from enum import Enum
from numbers import Integral
import numpy as np
import torch as th
from torch.distributions import Distribution
from torch.nn import Module
from torch.optim import Adam, Optimizer
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
from typing import Any, Optional, Union, Type


def normalize_shape(shape: Optional[th.Size]) -> th.Size:
    """
    Normalize a sample shape akin to numpy.
    """
    if shape is None:
        return ()
    if isinstance(shape, Integral):
        return (shape,)
    return shape


class StochasticContextMode(Enum):
    SAMPLE = "sample"
    LOG_PROB = "log_prob"


class StochasticContext:
    """
    Context for storing variables and evaluating log probabilities.
    """
    DEFAULT_CONTEXT: Optional["StochasticContext"] = None
    INSTANCE: Optional["StochasticContext"] = None

    def __init__(self, values: Optional[dict[str, th.Tensor]] = None,
                 mode: StochasticContextMode = StochasticContextMode.SAMPLE) -> None:
        self.mode = StochasticContextMode(mode)
        self.log_probs: dict[str, th.Tensor] = {}
        self.values = values or {}

    def __enter__(self) -> "StochasticContext":
        if StochasticContext.INSTANCE:
            raise RuntimeError("only one log prob context can be active")
        StochasticContext.INSTANCE = self
        return self

    def __exit__(*args) -> None:
        StochasticContext.INSTANCE = None

    def log_prob(self, aggregate: bool = False) -> None:
        """
        Get the log probability for the stochastic context.
        """
        if not self.log_probs:
            raise RuntimeError("log probs have not yet been evaluated")
        return maybe_aggregate(self.log_probs, aggregate)


def sample(name: str, dist_cls: Union[Distribution, Type[Distribution]], *args,
           sample_shape: Optional[th.Size] = None, context: Optional[StochasticContext] = None,
           **kwargs) -> Any:
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
    context = context or StochasticContext.INSTANCE
    sample_shape = normalize_shape(sample_shape)
    if context:
        mode = context.mode
        x = context.values.get(name)
    else:
        mode = StochasticContextMode.SAMPLE
        x = None
    if mode == StochasticContextMode.LOG_PROB:  # We want to evaluate log probabilities.
        if name in context.log_probs:
            raise RuntimeError(f"log probability has already been evaluated for variable {name}")
        if x is None:
            raise ValueError(f"cannot evaluate log probability because variable {name} is missing")
        dist = dist_cls if isinstance(dist_cls, Distribution) else dist_cls(*args, **kwargs)
        expected_shape = sample_shape + dist.batch_shape + dist.event_shape
        if expected_shape != x.shape:
            raise ValueError(f"expected shape {expected_shape} for variable {name} but got "
                             f"{x.shape}")
        context.log_probs[name] = dist.log_prob(x)
    elif mode == StochasticContextMode.SAMPLE:  # We want to sample from the model.
        if x is None:
            dist = dist_cls if isinstance(dist_cls, Distribution) else dist_cls(*args, **kwargs)
            # We use sample rather than rsample here to generate samples from the model. This will
            # stop any gradients so we cannot differentiate through any forward passes.
            x = dist.sample(sample_shape)
            if context:
                context.values[name] = x
    else:
        raise ValueError(mode)
    return x


def maybe_aggregate(value: dict[str, th.Tensor], aggregate: bool) -> Any:
    """
    Aggregate the values of a dictionary.
    """
    if aggregate:
        return sum(x.sum() for x in value.values())
    return value


def train(loss: Module, optim: Optional[Optimizer] = None, scheduler: Optional[Any] = None,
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
            loss_value, entropy = loss(return_entropy=True)
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

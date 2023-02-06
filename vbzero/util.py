from numbers import Integral
import numpy as np
import torch as th
from torch.distributions import Distribution
from torch.nn import Module
from torch.optim import Adam, Optimizer
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
from typing import Any, Optional, Union


def normalize_shape(shape: Optional[th.Size]) -> th.Size:
    """
    Normalize a sample shape akin to numpy.
    """
    if shape is None:
        return ()
    if isinstance(shape, Integral):
        return (shape,)
    return shape


class LogProbContext:
    """
    Context for evaluating log probabilities.
    """
    INSTANCE: Optional["LogProbContext"] = None

    def __init__(self) -> None:
        self.log_probs: dict[str, th.Tensor] = {}

    def __enter__(self) -> "LogProbContext":
        if LogProbContext.INSTANCE:
            raise RuntimeError("only one log prob context can be active")
        LogProbContext.INSTANCE = LogProbContext()
        return LogProbContext.INSTANCE

    def __exit__(*args) -> None:
        LogProbContext.INSTANCE = None

    def log_prob(self, aggregate: bool = False) -> None:
        return maybe_aggregate(self.log_probs, aggregate)


def sample(name: str, value: dict, dist_cls: Union[Distribution, type[Distribution]], *args,
           sample_shape: Optional[th.Size] = None, **kwargs) -> Any:
    """
    Sample a value if it does not already exist in the dictionary or evaluate log probabilities if a
    :class:`.LogProbContext` is active.
    """
    x = value.get(name)
    if instance := LogProbContext.INSTANCE:  # We want to evaluate log probabilities.
        if name in instance.log_probs:
            raise RuntimeError(f"log probability has already been evaluated for variable {name}")
        if x is None:
            raise ValueError(f"cannot evaluate log probability because variable {name} is missing")
        dist = dist_cls if isinstance(dist_cls, Distribution) else dist_cls(*args, **kwargs)
        instance.log_probs[name] = dist.log_prob(x)
    else:  # We want to sample from the model.
        if x is None:
            dist = dist_cls if isinstance(dist_cls, Distribution) else dist_cls(*args, **kwargs)
            x = dist.rsample(normalize_shape(sample_shape))
            value[name] = x
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

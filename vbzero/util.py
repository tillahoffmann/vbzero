from numbers import Integral
import torch as th
from torch.distributions import Distribution
from typing import Any, Optional


def normalize_shape(shape: Optional[th.Size]) -> th.Size:
    """
    Normalize a sample shape akin to numpy.
    """
    if shape is None:
        return ()
    if isinstance(shape, Integral):
        return (shape,)
    return shape


def maybe_sample(name: str, value: dict, cls: type[Distribution], *args,
                 sample_shape: Optional[th.Size] = None, **kwargs) -> Any:
    """
    Sample a value if it does not already exist in the dictionary.
    """
    try:
        return value[name]
    except KeyError:
        dist = cls(*args, **kwargs)
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

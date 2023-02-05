import torch as th
from torch.distributions import Distribution
from typing import Any, Optional


class Model(Distribution):
    """
    Base class for probabilistic models.
    """
    def rsample(self, sample_shape: Optional[th.Size] = None, value: Optional[dict] = None) -> dict:
        raise NotImplementedError

    def log_prob(self, value: dict, aggregate: bool = False) -> Any:
        raise NotImplementedError

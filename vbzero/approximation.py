import torch as th
from torch.distributions import Distribution
from typing import Any, Optional
from .util import maybe_aggregate, normalize_shape


class Approximation(Distribution):
    def entropy(self, aggregate: bool = False) -> Any:
        """
        Evaluate the entropy of the approximation.

        Args:
            aggregate: Sum the entropies of all variables.

        Returns: Entropy tensor if :code:`aggregate` else a dictionary of entropy tensors keyed by
            variable name.
        """
        raise NotImplementedError

    def log_prob(self, value: dict, aggregate: bool = False) -> Any:
        """
        Evaluate the log probability of all variables.

        Args:
            value: Dictionary of values keyed by variable name.
            aggregate: Sum the log probabilities of all variables.

        Returns: Log probability tensor if :code:`aggregate` else a dictionary of log probability
            tensors keyed by variable name.
        """
        raise NotImplementedError

    def rsample(self, sample_shape: Optional[th.Size] = None, value: Optional[dict] = None) -> dict:
        """
        Draw a sample from the approximate distribution.

        Args:
            sample_shape: Sample batch shape.
            value: Dictionary of variables to treat as fixed, e.g., hyperparameters or pinned
                variables for debugging.

        Returns: Dictionary of samples keyed by variable name.
        """
        raise NotImplementedError


class DistributionDict(Distribution):
    """
    Mean-field approximation.

    Args:
        distribution: Dictionary of mean field distributions keyed by variable name.
        strict: Enforce that all variables are given and that there are no extra variables.
    """
    def __init__(self, distributions: dict[str, Distribution], strict: bool = True) -> None:
        self.distributions = distributions
        self.strict = strict

    def _maybe_enforce_strict(self, value: dict[str, Any]) -> None:
        if not self.strict:
            return  # Nothing to be done here.
        if missing := set(self.distributions) - set(value):
            raise ValueError(f"encountered missing variables: {', '.join(missing)}; "
                             f"expected {', '.join(self.distributions)}")
        if extra := set(value) - set(self.distributions):
            raise ValueError(f"encountered extra variables: {', '.join(extra)}; "
                             f"expected {', '.join(self.distributions)}")

    def rsample(self, sample_shape: Optional[th.Size] = None, value: Optional[dict] = None) -> dict:
        value = value.copy() if value else {}
        sample_shape = normalize_shape(sample_shape)
        for name, distribution in self.distributions.items():
            if name not in value:
                value[name] = distribution.rsample(sample_shape)
        return value

    def log_prob(self, value: dict, aggregate: bool = False) -> Any:
        self._maybe_enforce_strict(value)
        lp = {name: self.distributions[name].log_prob(x) for name, x in value.items()}
        return maybe_aggregate(lp, aggregate)

    def entropy(self, aggregate: bool = False) -> Any:
        ent = {name: distribution.entropy() for name, distribution in self.distributions.items()}
        return maybe_aggregate(ent, aggregate)

    def __getitem__(self, name: str) -> Distribution:
        return self.distributions[name]

import torch as th
from torch.distributions import Distribution
from typing import Any, Optional
from .util import maybe_aggregate, normalize_shape


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

    @property
    def mean(self) -> dict:
        return {name: distribution.mean for name, distribution in self.distributions.items()}

    @property
    def mode(self):
        return {name: distribution.mode for name, distribution in self.distributions.items()}

    def __getitem__(self, name: str) -> Distribution:
        return self.distributions[name]

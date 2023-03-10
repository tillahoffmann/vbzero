"""
nn
==
"""

import torch as th
from torch.distributions import Distribution, Transform, TransformedDistribution, transform_to
from torch.distributions.constraints import Constraint
from torch.nn import Module, ModuleDict, Parameter, ParameterDict
from typing import Any, Callable, Mapping, Optional, Tuple, Type
from .approximation import DistributionDict
from .util import condition, maybe_aggregate, LogProb


class ParameterizedDistribution(Module):
    """
    Parameterized distribution with initial conditions. Parameters are transformed to an
    unconstrained space for optimization, and the distribution is constructed by transforming back
    to the constrained space in the forward pass.

    Args:
        cls: Distribution to create.
        const: Parameter names that should be treated as constants, i.e., do not require gradients.
        transforms: Transformations to apply to the base distribution.
        **kwargs: Initial conditions of the distribution.
    """
    def __init__(self, cls: Type[Distribution], *, const: Optional[set[str]] = None,
                 transforms: Optional[list[Transform]] = None, **kwargs: dict[str, Any]):
        super().__init__()
        const = const or set()
        self.cls = cls
        self.transforms = transforms
        unconstrained = {}
        self.parameter_transforms: dict[str, Transform] = {}
        self.const = {}
        for key, value in kwargs.items():
            constraint: Constraint = cls.arg_constraints.get(key)
            # Treat parameters as constants if they are explicitly marked as constant or do not have
            # a constraint. Transform trainable parameters to an unconstrained space and store the
            # transform to get back to the constrained space.
            if constraint is None or key in const:
                self.const[key] = value
            else:
                # Check that the constraint is satisfied before transforming.
                value = th.as_tensor(value)
                if not constraint.check(value).all():
                    raise ValueError(f"parameter {key} does not satisfy {constraint}: {value}")
                transform: Transform = transform_to(constraint)
                unconstrained[key] = Parameter(transform.inv(value))
                self.parameter_transforms[key] = transform
        self.unconstrained = ParameterDict(unconstrained)

    def forward(self) -> Distribution:
        """
        Evaluate a trainable distribution.
        """
        # Transform unconstrained parameters to the constrained space and add constant parameters.
        kwargs = {key: self.parameter_transforms[key](value) for key, value in
                  self.unconstrained.items()}
        kwargs.update(self.const)
        # Create the base distribution and apply transforms if given.
        distribution = self.cls(**kwargs)
        if self.transforms:
            return TransformedDistribution(distribution, self.transforms)
        return distribution


class ParameterizedDistributionDict(ModuleDict):
    """
    Dictionary of parameterized distributions.
    """
    def forward(self) -> DistributionDict:
        """
        Evaluate a trainable dictionary of distributions.
        """
        return DistributionDict({
            name: parameterized_distribution() for name, parameterized_distribution in self.items()
        })


class VariationalLoss(Module):
    """
    Variational loss function.

    Args:
        model: Model declaration as a function.
        approximation: Model to evaluate the variational approximation.
    """
    def __init__(self, model: Callable, approximation: Module) -> None:
        super().__init__()
        self.model = model
        if isinstance(approximation, Mapping) and not isinstance(approximation, Module):
            approximation = ParameterizedDistributionDict(approximation)
        self.approximation = approximation

    def forward(self, *args, **kwargs) -> Tuple[th.Tensor, th.Tensor]:
        """
        Evaluate the evidence lower bound and entropy of the variational distribution.
        """
        # Evaluate a stochastic estimate of the expected log joint and add the entropy.
        dist: Distribution = self.approximation()
        sample = dist.rsample()
        entropy = dist.entropy(aggregate=True)
        with LogProb() as log_prob:
            condition(self.model, sample)(*args, **kwargs)
        elbo = maybe_aggregate(log_prob, True) + entropy
        return - elbo, entropy

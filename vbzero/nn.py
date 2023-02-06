import torch as th
from torch.distributions import Distribution, Transform, TransformedDistribution, transform_to
from torch.distributions.constraints import Constraint
from torch.nn import Module, ModuleDict, Parameter, ParameterDict
from typing import Any, Callable, Optional
from .approximation import DistributionDict
from .model import Model
from .util import LogProbContext


class ParametrizedDistribution(Module):
    """
    Parametrized distribution with initial conditions. Parameters are transformed to an
    unconstrained space for optimization, and the distribution is constructed by transforming back
    to the constrained space in the forward pass.

    Args:
        cls: Distribution to create.
        const: Parameter names that should be treated as constants, i.e., do not require gradients.
        transforms: Transformations to apply to the base distribution.
        **kwargs: Initial conditions of the distribution.
    """
    def __init__(self, cls: type[Distribution], *, const: Optional[set[str]] = None,
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
                transform: Transform = transform_to(constraint)
                unconstrained[key] = Parameter(transform.inv(th.as_tensor(value)))
                self.parameter_transforms[key] = transform
        self.unconstrained = ParameterDict(unconstrained)

    def forward(self) -> type[Distribution]:
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
    def forward(self) -> dict[str, Distribution]:
        return DistributionDict({
            name: parameterized_distribution() for name, parameterized_distribution in self.items()
        })


class VariationalLoss(Module):
    def __init__(self, model: Model, approximation: Module,
                 value: Optional[dict[str, th.Tensor]] = None) -> None:
        super().__init__()
        self.model = model
        self.approximation = approximation
        self.value = value

    def forward(self, return_entropy: bool = False) -> th.Tensor:
        # Evaluate a stochastic estimate of the expected log joint and add the entropy.
        dist: Distribution = self.approximation()
        sample = dist.rsample(value=self.value)
        entropy = dist.entropy(aggregate=True)
        with LogProbContext() as context:
            if isinstance(self.model, Distribution):
                self.model.rsample(value=sample)
            elif isinstance(self.model, Callable):
                self.model(value=sample)
            else:
                raise TypeError("model must be a torch distribution or callable")
        elbo = context.log_prob(aggregate=True) + entropy
        if return_entropy:
            return - elbo, entropy
        return - elbo

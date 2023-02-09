import pytest
import torch as th
from torch.distributions import ExpTransform, LKJCholesky, Normal
from vbzero import nn, util


def test_parameterized_distribution() -> None:
    module = nn.ParameterizedDistribution(Normal, loc=0., scale=5.)
    assert set(module.unconstrained) == {"loc", "scale"}
    dist: Normal = module()
    assert dist.loc == 0
    assert dist.scale == 5


def test_parameterized_distribution_invalid() -> None:
    with pytest.raises(ValueError, match="does not satisfy"):
        nn.ParameterizedDistribution(Normal, loc=0., scale=-1.)


def test_parameterized_distribution_const() -> None:
    module = nn.ParameterizedDistribution(Normal, loc=0., scale=5., const="scale")
    assert set(module.unconstrained) == {"loc"}


def test_parameterized_distribution_parameter_without_constraint() -> None:
    module = nn.ParameterizedDistribution(LKJCholesky, dim=4, concentration=2)
    assert set(module.unconstrained) == {"concentration"}


def test_parameterized_distribution_with_transform() -> None:
    module = nn.ParameterizedDistribution(Normal, loc=0., scale=1., transforms=ExpTransform())
    x: th.Tensor = module().rsample([100])
    assert (x > 0).all()


def test_distribution_dict() -> None:
    module = nn.ParameterizedDistributionDict({
        "x": nn.ParameterizedDistribution(Normal, loc=0., scale=5.),
    })
    assert isinstance(module()["x"], Normal)


def test_variational_loss() -> None:
    @util.model
    def model():
        util.sample("x", Normal(0, 1))

    loss = nn.VariationalLoss(model, {
        "x": nn.ParameterizedDistribution(Normal, loc=0., scale=1.)
    })
    elbo, entropy = loss()
    assert elbo.shape == ()
    assert entropy.shape == ()

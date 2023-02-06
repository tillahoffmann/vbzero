import pytest
import torch as th
from typing import Callable
import vbzero as vz


def coin_model_eager() -> None:
    """
    Simple stochastic model for coin flips with eager distribution evaluation.
    """
    proba = vz.util.sample("proba", th.distributions.Beta(1, 1))
    x = vz.util.sample("x", th.distributions.Bernoulli(proba), sample_shape=10)
    return proba, x


def coin_model_lazy() -> None:
    """
    Simple stochastic model for coin flips with deferred distribution evaluation.
    """
    proba = vz.util.sample("proba", th.distributions.Beta, concentration0=1,
                           concentration1=1)
    vz.util.sample("x", th.distributions.Bernoulli, probs=proba, sample_shape=10)


@pytest.fixture(params=[coin_model_eager, coin_model_lazy])
def coin_model(request: pytest.FixtureRequest) -> Callable:
    return request.param


def test_sample_and_log_prob(coin_model: Callable) -> dict:
    with vz.util.StochasticContext() as context:
        coin_model()
    sample = context.values
    assert 0 <= sample["proba"] <= 1
    assert sample["x"].shape == (10,)

    with vz.util.StochasticContext(sample, mode="log_prob") as context:
        coin_model()


def test_sample_fixed(coin_model: Callable) -> dict:
    with vz.util.StochasticContext({"proba": 0.2}) as context:
        coin_model()
    assert context.values["proba"] == 0.2
    assert context.values["x"].shape == (10,)


def test_integration(coin_model: Callable) -> None:
    # Create Bernoulli random variables.
    x = th.zeros(10)
    x[:3] = 1

    # Parameterize the variational approximation for optimization.
    distributions = vz.nn.ParameterizedDistributionDict({
        "proba": vz.nn.ParametrizedDistribution(th.distributions.Beta, concentration0=1,
                                                concentration1=1),
    })

    # Create the loss and optimizer.
    variational = vz.nn.VariationalLoss(coin_model, distributions, value={"x": x})
    optim = th.optim.Adam(distributions.parameters(), lr=0.1)

    vz.util.train(variational, optim, rtol=0.01)

    expected = (x.sum() + 1) / (x.numel() + 2)
    actual = distributions()["proba"].mean

    assert (expected - actual).abs() < 0.1

import torch as th
import vbzero as vz


def coin_model() -> None:
    """
    Simple stochastic model for coin flips with eager distribution evaluation.
    """
    n = vz.util.hyperparam("n")
    proba = vz.util.sample("proba", th.distributions.Beta(1, 1))
    x = vz.util.sample("x", th.distributions.Bernoulli(proba), sample_shape=n)
    return proba, x


def test_integration() -> None:
    # Create Bernoulli random variables.
    x = th.zeros(10)
    x[:3] = 1

    # Parameterize the variational approximation for optimization.
    distributions = vz.nn.ParameterizedDistributionDict({
        "proba": vz.nn.ParameterizedDistribution(th.distributions.Beta, concentration0=1,
                                                 concentration1=1),
    })

    # Create the loss and optimizer.
    conditioned = vz.util.condition(coin_model, x=x, n=x.numel())
    variational = vz.nn.VariationalLoss(conditioned, distributions)
    optim = th.optim.Adam(distributions.parameters(), lr=0.1)

    vz.util.train(variational, optim=optim, rtol=0.01)

    expected = (x.sum() + 1) / (x.numel() + 2)
    actual = distributions()["proba"].mean

    assert (expected - actual).abs() < 0.1

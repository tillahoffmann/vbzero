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


def test_prior_and_conditioned_model() -> None:
    @vz.util.model(return_state=True)
    def model():
        # Hyperparameters.
        n = 50
        sigma = 1
        length_scale = 0.1
        kappa = 0.1
        # Sample from the model.
        x = vz.util.sample("x", th.distributions.Normal(0, 1), sample_shape=n)
        cov = sigma * (- (x[:, None] - x) ** 2 / (2 * length_scale ** 2)).exp() + 1e-3 * th.eye(n)
        w = vz.util.sample("w", th.distributions.Normal(0, 1), sample_shape=n)
        z = vz.util.record("z", th.linalg.cholesky(cov) @ w)
        vz.util.sample("y", th.distributions.Normal(z, kappa))

    # Check that conditioning works here.
    state = model()
    assert set(state) == set("wxyz")
    conditioned = vz.util.condition(model, w=state["w"], x=state["x"])
    prediction = conditioned()
    th.testing.assert_close(state["w"], prediction["w"])
    th.testing.assert_close(state["x"], prediction["x"])
    th.testing.assert_close(state["z"], prediction["z"])

    # Evaluate an estimate of the log likelihood.
    n, = state["x"].size()
    approximation = vz.nn.ParameterizedDistributionDict({
        "w": vz.nn.ParameterizedDistribution(th.distributions.Normal, loc=th.randn(n),
                                             scale=th.ones(n)),
    })
    dist: vz.nn.DistributionDict = approximation()
    sample = dist.rsample()
    with vz.util.LogProb() as log_prob:
        vz.util.condition(model, state["x", "y"], sample)()

    # Check that gradients exist.
    loss: th.Tensor = vz.util.maybe_aggregate(log_prob, True)
    loss.backward()
    for parameter in approximation.parameters():
        # Check gradients exist for all parameters and reset them.
        assert parameter.grad is not None
        parameter.grad = None

    # Check that gradients exist for the entropy.
    dist: vz.nn.DistributionDict = approximation()
    loss = dist.entropy(aggregate=True)
    loss.backward()
    # Location does not contribute to the entropy so should have no gradient.
    assert approximation["w"].unconstrained.loc.grad is None
    assert approximation["w"].unconstrained.scale.grad is not None

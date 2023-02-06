import torch as th
from typing import Optional
import vbzero as vz


def test_integration() -> None:
    # Create Bernoulli random variables.
    x = th.zeros(10)
    x[:3] = 1

    # Declare a model.
    class CoinModel(vz.model.Model):
        arg_constraints = {}

        def rsample(self, sample_shape: Optional[th.Size] = None, value: Optional[dict] = None) \
                -> dict:
            value = value.copy() if value else {}
            proba = vz.util.sample("proba", value, th.distributions.Beta, concentration0=1,
                                   concentration1=1, sample_shape=sample_shape)
            vz.util.sample("x", value, th.distributions.Bernoulli, probs=proba,
                           sample_shape=sample_shape)
            return value
    model = CoinModel()

    # Parameterize the variational approximation for optimization.
    distributions = vz.nn.ParameterizedDistributionDict({
        "proba": vz.nn.ParametrizedDistribution(th.distributions.Beta, concentration0=1,
                                                concentration1=1),
    })

    # Create the loss and optimizer.
    variational = vz.nn.VariationalLoss(model, distributions, value={"x": x})
    optim = th.optim.Adam(distributions.parameters(), lr=0.01)

    vz.util.train(variational, optim, rtol=0.01)

    expected = (x.sum() + 1) / (x.numel() + 2)
    actual = distributions()["proba"].mean

    assert (expected - actual).abs() < 0.1

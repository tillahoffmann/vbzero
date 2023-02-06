import torch as th
from typing import Any
import vbzero as vz


def test_integration() -> None:
    # Create Bernoulli random variables.
    x = th.zeros(10)
    x[:3] = 1

    # Declare a model.
    class CoinModel(vz.model.Model):
        arg_constraints = {}

        def log_prob(self, value: dict, aggregate: bool = False) -> Any:
            return vz.util.maybe_aggregate({
                "proba": th.distributions.Beta(1, 1).log_prob(value["proba"]),
                "x": th.distributions.Bernoulli(value["proba"]).log_prob(value["x"]).sum(),
            }, aggregate)
    model = CoinModel()

    # Parameterize the variational approximation for optimization.
    distributions = vz.nn.ParameterizedDistributionDict({
        "proba": vz.nn.ParametrizedDistribution(th.distributions.Beta, concentration0=1,
                                                concentration1=1),
    })

    # Create the loss and optimizer.
    variational = vz.nn.VariationalLoss(model, distributions, value={"x": x})
    optim = th.optim.Adam(distributions.parameters(), lr=0.01)

    for _ in range(1000):
        optim.zero_grad()
        loss_value = variational()
        loss_value.backward()
        optim.step()

    expected = (x.sum() + 1) / (x.numel() + 2)
    actual = distributions()["proba"].mean

    assert (expected - actual).abs() < 0.1

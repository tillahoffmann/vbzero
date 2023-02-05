import torch as th
from typing import Any
import vbzero as vz


def test_integration() -> None:
    class CoinModel(vz.model.Model):
        arg_constraints = {}

        def log_prob(self, value: dict, aggregate: bool = False) -> Any:
            return vz.util.maybe_aggregate({
                "proba": th.distributions.Beta(1, 1).log_prob(value["proba"]),
                "x": th.distributions.Bernoulli(value["proba"]).log_prob(value["x"]).sum(),
            }, aggregate)

    model = CoinModel()

    distributions = vz.nn.ParameterizedDistributionDict({
        "proba": vz.nn.ParametrizedDistribution(th.distributions.Beta, concentration0=1,
                                                concentration1=1),
    })

    loss = vz.nn.VariationalLoss()
    optim = th.optim.Adam(distributions.parameters(), lr=0.01)
    x = th.zeros(10)
    x[:3] = 1

    for _ in range(1000):
        optim.zero_grad()
        approximation = vz.approximation.MeanField(distributions())
        loss_value = loss(model, approximation, value={"x": x})
        loss_value.backward()
        optim.step()

    expected = (x.sum() + 1) / (x.numel() + 2)
    actual = distributions()["proba"].mean

    assert (expected - actual).abs() < 0.1

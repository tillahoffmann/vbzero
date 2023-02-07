import numpy as np
import pytest
import torch as th
from vbzero import util


def model(x) -> None:
    y = util.sample("y", th.distributions.Normal(x, 1))
    z = util.sample("z", th.distributions.Normal(y, 1))
    return {
        "x1p": x + 1,
        "y": y,
        "z": z,
    }


def test_model_decorator() -> None:
    # Check that the model fails without state.
    with pytest.raises(RuntimeError, match="context is not active"):
        model(5)

    # Use explicit state.
    with util.State() as state:
        result = model(5)
    assert result["x1p"] == 6
    assert 2 <= result["y"] <= 8 and state["y"] is result["y"]

    # Use implicit state without returning it.
    wrapped_model = util.model(model)
    assert wrapped_model(7)["x1p"] == 8

    # Use implicit state but ensure we can still use explicit state.
    with util.State() as state:
        wrapped_model(6)
    assert 3 <= state["y"] <= 9

    # Use implicit state and return it with the decorator.
    wrapped_model = util.model(return_state=True)(model)
    result, state = wrapped_model(9)
    assert result["x1p"] == 10
    assert 6 <= result["y"] <= 12 and state["y"] is result["y"]

    # Ensure the implicit state is the explicit state.
    with util.State() as state:
        _, other = wrapped_model(11)
        assert state is other


def test_given_state() -> None:
    with util.State(y=42):
        result = model(-10)
    assert result["y"] == 42


def test_condition() -> None:
    conditioned = util.condition(model, y=37)
    result = conditioned(-19)
    assert result["y"] == 37
    assert 34 <= result["z"] <= 40

    conditioned2 = util.condition(conditioned, z=19)
    assert conditioned2(-19) == {"x1p": -18, "y": 37, "z": 19}

    with pytest.raises(KeyError, match="key y has already been set"):
        util.condition(conditioned, y=18)(0)

    with pytest.raises(KeyError, match="key z has already been set"):
        util.condition(model, {"z": 17}, z=19)()


def test_log_prob_context() -> None:
    x = 17
    y = th.as_tensor(19.)
    z = th.as_tensor(21.)
    with util.LogProb() as log_prob:
        util.condition(model, y=y, z=z)(x)
    np.testing.assert_allclose(log_prob["y"], th.distributions.Normal(x, 1).log_prob(y))
    np.testing.assert_allclose(log_prob["y"], th.distributions.Normal(y, 1).log_prob(z))

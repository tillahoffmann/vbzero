import numpy as np
import pytest
import torch as th
from typing import Any
from vbzero import util


def model(x=5) -> None:
    y = util.sample("y", th.distributions.Normal(x, 1))
    z = util.sample("z", th.distributions.Normal(y, 1))
    return {
        "x1p": x + 1,
        "y": y,
        "z": z,
    }


def test_model_decorator() -> None:
    # Check that the model fails without state.
    with pytest.raises(RuntimeError, match="no active <class '.*?State'>"):
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
    with util.State(y=42, z=10) as state:
        result = model(-10)
    assert result["y"] == 42
    assert state["y", "z"] == {"y": 42, "z": 10}


@pytest.mark.xfail(reason="write-once state not yet implemented")
def test_state_once() -> None:
    state = util.State(a=1)
    with pytest.raises(KeyError):
        del state["a"]
    with pytest.raises(KeyError):
        state.pop("a")
    with pytest.raises(KeyError):
        state["a"] = 5
    assert state["a"] == 1


def test_condition() -> None:
    conditioned = util.condition(model, y=37)
    result = conditioned(-19)
    assert result["y"] == 37
    assert 34 <= result["z"] <= 40

    conditioned2 = util.condition(conditioned, z=19)
    assert conditioned2(-19) == {"x1p": -18, "y": 37, "z": 19}


@pytest.mark.xfail(reason="write-once state not yet implemented")
def test_condition_write_once() -> None:
    conditioned = util.condition(model, y=37)

    with pytest.raises(KeyError, match="key y has already been set"):
        util.condition(conditioned, y=18)(0)

    with pytest.raises(KeyError, match="key z has already been set"):
        util.condition(model, {"z": 17}, z=19)()


def test_log_prob_context() -> None:
    x = 17
    y = th.as_tensor(19.)
    z = th.as_tensor(21.)
    with util.LogProb() as log_prob:
        assert util.TraceMixin.INSTANCE is log_prob
        util.condition(model, y=y, z=z)(x)
    np.testing.assert_allclose(log_prob["y"], th.distributions.Normal(x, 1).log_prob(y))
    np.testing.assert_allclose(log_prob["y"], th.distributions.Normal(y, 1).log_prob(z))


def test_log_prob_context_invalid() -> None:
    x = 17
    # Check that re-evaluation fails.
    conditioned = util.condition(model, y=th.as_tensor(19.), z=th.as_tensor(21.))
    with util.LogProb() as log_prob:
        conditioned(x)
    with pytest.raises(RuntimeError, match="has already been evaluated"), log_prob:
        conditioned(x)

    # Check that missing values raise errors.
    with pytest.raises(ValueError, match="variable y is missing"), \
            util.State.get_instance(strict=False), util.LogProb():
        util.model(model)(x)

    # Wrong shape.
    with pytest.raises(ValueError, match="expected shape"), util.LogProb():
        util.condition(model, y=th.randn(17, 18))(x)


@pytest.mark.parametrize("arg, expected", [
    ((), th.Size([])),
    (None, th.Size([])),
    (15, th.Size([15])),
    ((7, 8), th.Size([7, 8])),
    ([9, 11], th.Size([9, 11])),
])
def test_normalize_shape(arg: Any, expected: th.Size) -> None:
    assert util.normalize_shape(arg) == expected


def test_hyperparam() -> None:
    with util.State(x=1, y=2):
        assert util.hyperparam("x", "y") == (1, 2)
        assert util.hyperparam("x") == 1


def test_maybe_aggregate() -> None:
    values = {"x": th.arange(5), "y": th.as_tensor(5)}
    assert util.maybe_aggregate(values, False) is values
    assert util.maybe_aggregate(values, True) == 15


def test_context():
    class TestContext(util.TraceMixin, dict):
        def __call__(self, state: util.State, name: str, *args, **kwargs) -> None:
            value = state.get(name)
            self[name] = value
            if value is None:
                state[name] = self._evaluate_distribution(*args, **kwargs).sample()

    # If the state context is at the top of the stack, we expect variables to not yet be available
    # because we run `sample` statements from the inside out.
    with util.State() as state, TestContext() as context:
        model()
    assert context == {"y": None, "z": None}
    with state, TestContext() as context:
        model()
    assert context == state


def test_context_reentry() -> None:
    assert not util.TraceMixin.INSTANCE
    with util.Sample() as sample:
        assert util.TraceMixin.INSTANCE is sample
        with pytest.raises(RuntimeError), sample:
            pass
        assert util.TraceMixin.INSTANCE is sample
    assert not util.TraceMixin.INSTANCE


def test_context_uniqueness() -> None:
    with util.Sample(), pytest.raises(RuntimeError, match="cannot activate"), util.Sample():
        pass


def test_state_creation() -> None:
    state = util.State(x=5)
    other = state.copy()
    assert isinstance(other, util.State)
    assert other == state
    assert other is not state

    other = state | {"y": 3}
    assert isinstance(other, util.State)
    assert other is not state

    state |= {"y": 3}
    assert state == {"x": 5, "y": 3}

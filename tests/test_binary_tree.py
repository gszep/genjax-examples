from functools import partial

import jax
import jax.numpy as jnp
from genjax import Arguments, Const, flip, gen, normal, ChoiceMap  # type: ignore
from jax import jit, vmap
from jaxtyping import Array, Float, PRNGKeyArray
from tests.test_mcmc import run_inference

MAX_DEPTH = 5
global depth
depth = 0


@gen
def leaf(node_id: Const[int], *args: Float[Array, "..."]):
    y = normal(50.0, 1.0) @ f"normal:{node_id.unwrap()}"
    return y


@gen
def branch(node_id: Const[int], *args: Float[Array, "..."]):

    x = model(Const(2 * node_id.unwrap()), *args) @ f"branch:left:{node_id.unwrap()}"
    y = (
        model(Const(2 * node_id.unwrap() + 1), *args)
        @ f"branch:right:{node_id.unwrap()}"
    )

    return x + y


@gen
def model(node_id: Const[int], *args: Float[Array, "..."]):
    model_args = (node_id, *args)
    branch_prob = args[0]

    global depth  # hacky way to avoid infinite recursion
    if depth >= MAX_DEPTH:
        return leaf(*model_args) @ f"leaf:{node_id.unwrap()}"

    else:
        depth += 1
        is_branch = flip(branch_prob) @ f"is_branch:{node_id.unwrap()}"
        return (
            branch.or_else(leaf)(is_branch, model_args, model_args)
            @ f"branch:{node_id.unwrap()}"
        )


@jit
@partial(vmap, in_axes=(0, None))
def simulate(key: PRNGKeyArray, args: Arguments):
    return model.simulate(key, args)


def test_binary_tree_inference():
    key = jax.random.PRNGKey(42)
    # simulate(
    #     keys,
    #     (
    #         Const(1),  # initial node_id
    #         jnp.array(0.3),  # branch_prob
    #     ),
    # )

    obs = ChoiceMap.d({"y": 5.0})
    model_args = (
        Const(1),  # initial node_id
        jnp.array(0.3),  # branch_prob
    )

    num_samples = 1
    key, subkey = jax.random.split(key)
    trace, mh_chain = run_inference(model, model_args, obs, subkey, num_samples)


def test_binary_tree():
    keys = jax.random.split(jax.random.PRNGKey(42), 200)
    trace = simulate(
        keys,
        (
            Const(1),  # initial node_id
            jnp.array(0.3),  # branch_prob
        ),
    )

    assert trace.retval.shape == (200,)

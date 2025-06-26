import jax
import jax.numpy as jnp
from jax import jit, vmap  # type: ignore
from functools import partial
from jax.scipy.special import gammaln  # Added for log gamma function

from genjax import gen, beta, bernoulli, categorical, gamma, scan, Trace, ChoiceMap, Diff  # type: ignore
from genjax import ChoiceMapBuilder as C
from jaxtyping import Array, Float, Integer, PRNGKeyArray  # type: ignore
from dataclasses import dataclass, field
import numpy as np

# --- Constants for numerical stability ---
LOG_ETA_CLIP_MIN = -20.0
GAMMA_SAMPLE_FLOOR = 1e-9
ALPHA_SBP_FLOOR = (
    1.0  # MODIFIED FOR DEBUGGING: Ensure SBP alpha is >= 1.0
    # to prevent beta(1, alpha) from having a pole (infinite density) at v_draw=1
    # if alpha < 1. This helps avoid 'inf' scores from such beta choices.
)
BERNOULLI_PARAM_EPSILON = 1e-7
# New constant for alpha grid sampling
ALPHA_GRID_SIZE = 100
ALPHA_GRID_MIN = 0.01
ALPHA_GRID_MAX = 10.0

# --- data ---
test_dataset_raw = [
    [True, False, False, False, False, True],
    [True, False, False, False, False, True],
    [True, False, False, False, False, True],
    [True, False, False, False, False, True],
    [True, False, False, False, False, True],
    [True, False, False, False, False, True],
    [True, False, False, False, False, True],
    [False, False, True, True, True, False],
    [False, False, True, True, True, False],
    [False, False, True, True, True, False],
    [False, False, True, True, True, False],
    [False, False, True, True, True, False],
]


# --- SBP module---
@jax.tree_util.register_dataclass
@dataclass
class StickBreakingResult:
    weights: Array


@jax.tree_util.register_dataclass
@dataclass
class _SBPLoopState:
    weights: Array
    remaining_stick: Float[Array, ""]
    alpha: Float[Array, ""]


@gen
def _sbp_step(
    state: _SBPLoopState, k_idx: int
) -> tuple[_SBPLoopState, Float[Array, ""]]:
    # state.alpha is guaranteed to be >= ALPHA_SBP_FLOOR (currently 1.0) due to sbp_instance_model.
    # If state.alpha were < 1.0, beta(1.0, state.alpha) would have infinite density at v_draw = 1.0.
    # Sampling v_draw = 1.0 in such a case could lead to its score being 'inf',
    # propagating to the overall trace score.
    v_draw = (
        beta(jnp.array(1.0, dtype=jnp.float32), state.alpha)
        @ f"v_draw_for_step_{k_idx}"
    )
    epsilon_clip = 1e-9
    v_draw_safe = jnp.clip(v_draw, epsilon_clip, 1.0 - epsilon_clip)

    current_weight_value = v_draw_safe * state.remaining_stick

    updated_weights = state.weights.at[k_idx].set(current_weight_value)
    new_remaining_stick = state.remaining_stick * (
        jnp.array(1.0, dtype=jnp.float32) - v_draw_safe
    )
    next_state = _SBPLoopState(
        weights=updated_weights, remaining_stick=new_remaining_stick, alpha=state.alpha
    )
    return next_state, current_weight_value


def create_sbp_model(num_max_components: int):
    if not isinstance(num_max_components, int):
        raise TypeError(
            f"create_sbp_model expects a Python int for num_max_components, but got {type(num_max_components)} with value {num_max_components}"
        )
    if num_max_components <= 1:
        pass

    scanned_body = scan(n=max(0, num_max_components - 1))(_sbp_step)

    @gen
    def sbp_instance_model(
        alpha_param: Float[Array, ""],
    ) -> StickBreakingResult:
        safe_alpha_param = jnp.maximum(
            jnp.asarray(alpha_param, dtype=jnp.float32), ALPHA_SBP_FLOOR
        )

        initial_state = _SBPLoopState(
            weights=jnp.zeros(num_max_components, dtype=jnp.float32),
            remaining_stick=jnp.array(1.0, dtype=jnp.float32),
            alpha=safe_alpha_param,
        )

        if num_max_components == 1:
            final_weights_array = jnp.array([1.0], dtype=jnp.float32)
        else:
            indices_for_scan = jnp.arange(num_max_components - 1, dtype=jnp.int32)
            final_loop_state, _ = (
                scanned_body(initial_state, indices_for_scan) @ "sbp_internal_scan_loop"
            )
            final_weights_array = final_loop_state.weights.at[
                num_max_components - 1
            ].set(final_loop_state.remaining_stick)
        return StickBreakingResult(weights=final_weights_array)

    return sbp_instance_model


# --- CrossCat data structure ---
@jax.tree_util.register_dataclass
@dataclass
class CrossCatHyperparamsConfig:
    bern_alpha: float = 1.1
    bern_beta: float = 1.1
    alpha_D_prior_shape: float = 1.0
    alpha_D_prior_rate: float = 1.0
    alpha_v_prior_shape: float = 1.0
    alpha_v_prior_rate: float = 1.0


@jax.tree_util.register_dataclass
@dataclass
class CrossCatLatents:
    alpha_D: Float[Array, ""]
    alpha_v_all_views: Float[Array, "max_views"]
    view_weights: Float[Array, "max_views"]
    column_view_assignments: Integer[Array, "num_cols"]
    local_col_indices: Integer[Array, "num_cols"]
    num_cols_per_view: Integer[Array, "max_views"]
    category_weights_all_views: Float[Array, "max_views max_categories"]
    row_category_assignments_all_views: Integer[Array, "max_views num_rows"]
    bernoulli_params: Float[Array, "max_views max_categories max_cols_per_view"]


@jax.tree_util.register_dataclass
@dataclass
class CrossCatResult:
    data: Integer[Array, "num_rows num_cols"]
    latents: CrossCatLatents


# --- CrossCat generative model ---
def create_crosscat_model(
    num_rows_static: int,
    num_cols_static: int,
    max_views_static: int,
    max_categories_static: int,
    max_cols_per_view_static: int,
):
    sbp_for_views = create_sbp_model(max_views_static)
    sbp_for_categories = create_sbp_model(max_categories_static)

    @gen
    def crosscat_instance_model(
        hp_alpha_D_shape_in: Float[Array, ""],
        hp_alpha_D_rate_in: Float[Array, ""],
        hp_alpha_v_shape_in: Float[Array, ""],
        hp_alpha_v_rate_in: Float[Array, ""],
        hp_bern_alpha_in: Float[Array, ""],
        hp_bern_beta_in: Float[Array, ""],
    ) -> CrossCatResult:
        num_rows = num_rows_static
        num_cols = num_cols_static
        max_views = max_views_static
        max_categories = max_categories_static
        max_cols_per_view = max_cols_per_view_static

        _hp_alpha_D_shape = jnp.asarray(
            jnp.maximum(hp_alpha_D_shape_in, 1e-7), dtype=jnp.float32
        )
        _hp_alpha_D_rate = jnp.asarray(
            jnp.maximum(hp_alpha_D_rate_in, 1e-7), dtype=jnp.float32
        )
        _hp_alpha_v_shape = jnp.asarray(
            jnp.maximum(hp_alpha_v_shape_in, 1e-7), dtype=jnp.float32
        )
        _hp_alpha_v_rate = jnp.asarray(
            jnp.maximum(hp_alpha_v_rate_in, 1e-7), dtype=jnp.float32
        )

        _hp_bern_alpha = jnp.asarray(
            jnp.maximum(hp_bern_alpha_in, 1.0000001), dtype=jnp.float32
        )
        _hp_bern_beta = jnp.asarray(
            jnp.maximum(hp_bern_beta_in, 1.0000001), dtype=jnp.float32
        )

        alpha_D_val = gamma(_hp_alpha_D_shape, _hp_alpha_D_rate) @ "alpha_D"

        view_sbr: StickBreakingResult = (
            sbp_for_views(alpha_D_val) @ "view_sbp_generation"
        )
        view_weights = view_sbr.weights

        _col_assignments_list = []
        for j_idx in range(num_cols):
            assignment = categorical(view_weights) @ f"column_view_assignment_{j_idx}"
            _col_assignments_list.append(assignment)
        column_view_assignments = (
            jnp.stack(_col_assignments_list)
            if num_cols > 0
            else jnp.array([], dtype=jnp.int32)
        )
        column_view_assignments = jnp.reshape(column_view_assignments, (num_cols,))

        _local_col_indices = jnp.full(num_cols, -1, dtype=jnp.int32)
        _num_cols_per_view = jnp.bincount(
            column_view_assignments.astype(jnp.int32), length=max_views
        ).astype(jnp.int32)

        _current_local_idx_counters = jnp.zeros(max_views, dtype=jnp.int32)
        for j_global in range(num_cols):
            assigned_v = column_view_assignments[j_global]
            _local_col_indices = _local_col_indices.at[j_global].set(
                _current_local_idx_counters[assigned_v]
            )
            _current_local_idx_counters = _current_local_idx_counters.at[
                assigned_v
            ].add(1)

        _bernoulli_params = jnp.zeros((max_views, max_categories, max_cols_per_view))
        _row_category_assignments = jnp.zeros((max_views, num_rows), dtype=jnp.int32)
        _category_weights_all = jnp.zeros((max_views, max_categories))
        _alpha_v_all_views_orig = jnp.zeros(max_views)

        for v_idx in range(max_views):
            alpha_v_val = (
                gamma(_hp_alpha_v_shape, _hp_alpha_v_rate) @ f"alpha_v_{v_idx}"
            )
            _alpha_v_all_views_orig = _alpha_v_all_views_orig.at[v_idx].set(alpha_v_val)

            cat_sbr_loop: StickBreakingResult = (
                sbp_for_categories(alpha_v_val) @ f"category_sbp_view_{v_idx}"
            )
            cat_weights_v = cat_sbr_loop.weights
            _category_weights_all = _category_weights_all.at[v_idx, :].set(
                cat_weights_v
            )

            _row_assignments_for_view_v_list = []
            for r_idx in range(num_rows):
                assignment = (
                    categorical(cat_weights_v)
                    @ f"row_cat_assignment_view_{v_idx}_row_{r_idx}"
                )
                _row_assignments_for_view_v_list.append(assignment)
            if num_rows > 0:
                _row_category_assignments = _row_category_assignments.at[v_idx, :].set(
                    jnp.reshape(
                        jnp.stack(_row_assignments_for_view_v_list), (num_rows,)
                    )
                )

            for c_idx in range(max_categories):
                for local_j_idx in range(max_cols_per_view):
                    p_val_beta = (
                        beta(_hp_bern_alpha, _hp_bern_beta)
                        @ f"p_v{v_idx}_c{c_idx}_lc{local_j_idx}"
                    )
                    _bernoulli_params = _bernoulli_params.at[
                        v_idx, c_idx, local_j_idx
                    ].set(p_val_beta)

        _data = jnp.zeros((num_rows, num_cols), dtype=jnp.int32)
        for r_idx in range(num_rows):
            for j_global_col in range(num_cols):
                assigned_view = column_view_assignments[j_global_col]
                assigned_category_for_row_in_view = _row_category_assignments[
                    assigned_view, r_idx
                ]
                local_col_idx_in_view = _local_col_indices[j_global_col]
                param_for_cell = _bernoulli_params[
                    assigned_view,
                    assigned_category_for_row_in_view,
                    local_col_idx_in_view,
                ]

                param_for_cell_safe = jnp.clip(
                    param_for_cell,
                    BERNOULLI_PARAM_EPSILON,
                    1.0 - BERNOULLI_PARAM_EPSILON,
                )
                val = bernoulli(param_for_cell_safe) @ f"data_r{r_idx}_c{j_global_col}"
                _data = _data.at[r_idx, j_global_col].set(val)

        latents = CrossCatLatents(
            alpha_D=alpha_D_val,
            alpha_v_all_views=_alpha_v_all_views_orig,
            view_weights=view_weights,
            column_view_assignments=column_view_assignments,
            local_col_indices=_local_col_indices,
            num_cols_per_view=_num_cols_per_view,
            category_weights_all_views=_category_weights_all,
            row_category_assignments_all_views=_row_category_assignments,
            bernoulli_params=_bernoulli_params,
        )
        return CrossCatResult(data=_data, latents=latents)

    return crosscat_instance_model


# --- New Alpha Sampling Function Factory (Grid-based) ---
def create_alpha_sampler_from_grid(address_suffix: str):
    @gen
    def _sample_alpha_internal(
        k_int: Integer[Array, ""],
        N_float: Float[Array, ""],
        alpha_prior_shape: Float[Array, ""],
        alpha_prior_rate: Float[Array, ""],
        alpha_grid_points: Float[Array, "grid_size"],
    ) -> Float[Array, ""]:
        log_prior_alpha_grid = gamma.logpdf(
            alpha_grid_points, alpha_prior_shape, alpha_prior_rate
        )

        safe_alpha_grid_points = jnp.maximum(alpha_grid_points, 1e-35)
        safe_alpha_plus_N = jnp.maximum(alpha_grid_points + N_float, 1e-35)

        log_crp_likelihood_grid = (
            k_int * jnp.log(safe_alpha_grid_points)
            + gammaln(safe_alpha_grid_points)
            - gammaln(safe_alpha_plus_N)
        )

        log_posterior_unnormalized_grid = log_prior_alpha_grid + log_crp_likelihood_grid
        log_posterior_unnormalized_grid_stable = (
            log_posterior_unnormalized_grid - jnp.max(log_posterior_unnormalized_grid)
        )
        probs = jax.nn.softmax(log_posterior_unnormalized_grid_stable)

        probs = jnp.where(jnp.isnan(probs), jnp.zeros_like(probs), probs)
        probs_sum = jnp.sum(probs)
        probs = jnp.where(
            probs_sum < 1e-9, jnp.ones_like(probs) / probs.shape[0], probs / probs_sum
        )
        probs = jnp.clip(probs, 0.0, 1.0)

        chosen_idx = categorical(probs) @ f"alpha_{address_suffix}_grid_choice"
        new_alpha_val = alpha_grid_points[chosen_idx]
        sampled_alpha_safe = jnp.maximum(new_alpha_val, GAMMA_SAMPLE_FLOOR)
        return sampled_alpha_safe

    return _sample_alpha_internal


# --- Gibbs Kernel for alpha_D (Updated) ---
@partial(
    jit,
    static_argnames=[
        "max_views_static_val",
        "num_total_columns",
        "alpha_D_prior_shape_static",
        "alpha_D_prior_rate_static",
    ],
)
def sample_alpha_D_gibbs_kernel(
    key: PRNGKeyArray,
    current_trace: Trace,
    num_total_columns: int,
    alpha_D_prior_shape_static: float,
    alpha_D_prior_rate_static: float,
    max_views_static_val: int,
    alpha_grid_dynamic: Float[Array, "grid_size"],
):
    column_view_assignments = current_trace.get_retval().latents.column_view_assignments
    bincount_views = jnp.bincount(
        column_view_assignments.astype(jnp.int32), length=max_views_static_val
    )
    k_int = jnp.sum(bincount_views > 0)
    N_float = jnp.array(num_total_columns, dtype=jnp.float32)

    sampler_gf_key, update_key = jax.random.split(key, 2)

    alpha_D_sampler = create_alpha_sampler_from_grid("D")

    sampler_args = (
        k_int,
        N_float,
        jnp.asarray(alpha_D_prior_shape_static, dtype=jnp.float32),
        jnp.asarray(alpha_D_prior_rate_static, dtype=jnp.float32),
        alpha_grid_dynamic,
    )

    sampler_trace = alpha_D_sampler.simulate(sampler_gf_key, sampler_args)
    new_alpha_D_sampled_val = sampler_trace.get_retval()

    new_choice_map = C.d({"alpha_D": new_alpha_D_sampled_val})
    model_args_from_trace = current_trace.get_args()
    argdiffs_for_model = Diff.no_change(model_args_from_trace)

    updated_trace, weight, _, _ = current_trace.update(
        update_key, new_choice_map, argdiffs_for_model
    )
    return updated_trace


# --- Gibbs Kernel for alpha_v (Updated) ---
@partial(
    jit,
    static_argnames=[
        "view_idx",
        "num_rows_static_val",
        # "max_categories_static_val", # This was the incorrect name in the error
        "max_categories_val",  # Corrected to match the function signature if it's static
        # However, max_categories_val is usually dynamic based on the model's configuration
        # If it is intended to be static (fixed for JIT), it should be passed as such from test_crosscat_mcmc
        # For now, assuming it's dynamic, so remove from static_argnames if it's not truly static across JIT calls.
        # Based on the NameError and its suggestion, it seems this was intended to be the *value* from the main function.
        # If the *value* can change, it CANNOT be a static_argname.
        # The error was about 'max_categories_static_val' NOT being an arg of the kernel.
        # The kernel arg is 'max_categories_val'. This one should be listed if static.
        "alpha_v_prior_shape_static",
        "alpha_v_prior_rate_static",
    ],
)
def sample_alpha_v_kernel(
    key: PRNGKeyArray,
    current_trace: Trace,
    view_idx: int,
    num_rows_static_val: int,
    max_categories_val: int,  # This is the actual argument name
    alpha_v_prior_shape_static: float,
    alpha_v_prior_rate_static: float,
    alpha_grid_dynamic: Float[Array, "grid_size"],
):
    # If max_categories_val should be static for JIT compilation (i.e., its value doesn't change
    # across different calls to a *specific compiled version* of this kernel for a given view_idx),
    # then it should be listed in static_argnames. The NameError implied 'max_categories_static_val'
    # was in static_argnames but not an argument, the actual argument 'max_categories_val' was used.
    # For safety and common usage, values like max_categories usually come from model configuration
    # and could be static if the configuration is fixed for the JIT.
    # Let's assume it's intended to be static as per the error context.
    # If not, it should be removed from static_argnames above.
    # For this fix, I will ensure 'max_categories_val' is in static_argnames if it should be static.

    alpha_v_key_str = f"alpha_v_{view_idx}"
    row_cat_assignments_for_view = (
        current_trace.get_retval().latents.row_category_assignments_all_views[
            view_idx, :
        ]
    )

    if num_rows_static_val > 0:
        bincount_cat = jnp.bincount(
            row_cat_assignments_for_view.astype(jnp.int32),
            length=max_categories_val,
        )
        kv_int = jnp.sum(bincount_cat > 0)
    else:
        kv_int = jnp.array(0, dtype=jnp.int32)
    Nv_float = jnp.array(num_rows_static_val, dtype=jnp.float32)

    sampler_gf_key, update_key = jax.random.split(key, 2)

    alpha_v_sampler = create_alpha_sampler_from_grid(f"v_{view_idx}")

    sampler_args = (
        kv_int,
        Nv_float,
        jnp.asarray(alpha_v_prior_shape_static, dtype=jnp.float32),
        jnp.asarray(alpha_v_prior_rate_static, dtype=jnp.float32),
        alpha_grid_dynamic,
    )

    sampler_trace = alpha_v_sampler.simulate(sampler_gf_key, sampler_args)
    new_alpha_v_sampled_val = sampler_trace.get_retval()

    new_choice_map = C.d({alpha_v_key_str: new_alpha_v_sampled_val})
    model_args_from_trace = current_trace.get_args()
    argdiffs_for_model = Diff.no_change(model_args_from_trace)

    updated_trace, weight, _, _ = current_trace.update(
        update_key, new_choice_map, argdiffs_for_model
    )
    return updated_trace


# --- Main ---
def test_crosscat_mcmc():
    observed_data_matrix = jnp.array(test_dataset_raw, dtype=jnp.int32)
    num_rows_val, num_cols_val = observed_data_matrix.shape

    max_views_static_val = max(1, min(4, num_cols_val) if num_cols_val > 0 else 1)
    max_categories_val_for_model = max(
        1, min(8, num_rows_val) if num_rows_val > 0 else 1
    )  # Renamed for clarity
    max_cols_per_view_val = max(1, num_cols_val if num_cols_val > 0 else 1)

    alpha_grid = jnp.linspace(
        ALPHA_GRID_MIN, ALPHA_GRID_MAX, ALPHA_GRID_SIZE, dtype=jnp.float32
    )

    print(
        f"Gibbs Sampling Test (alpha_D and alpha_v with Grid Sampling): Using observed_data with shape: rows={num_rows_val}, cols={num_cols_val}"
    )
    print(
        f"Max views static: {max_views_static_val}, Max categories static: {max_categories_val_for_model}"
    )
    print(
        f"Alpha grid size: {ALPHA_GRID_SIZE}, range: [{ALPHA_GRID_MIN}, {ALPHA_GRID_MAX}]"
    )
    print("Observed Data (first 5 rows):")
    print(observed_data_matrix[:5, :])
    print("-" * 30)

    obs_dict = {}
    for r in range(num_rows_val):
        for c_col in range(num_cols_val):
            obs_dict[f"data_r{r}_c{c_col}"] = observed_data_matrix[r, c_col]
    observations_choicemap = C.d(obs_dict)

    hyperparams_config = CrossCatHyperparamsConfig()
    print(
        f"Effective bern_alpha for Beta distribution: {hyperparams_config.bern_alpha}"
    )
    print(f"Effective bern_beta for Beta distribution: {hyperparams_config.bern_beta}")
    print(f"Effective ALPHA_SBP_FLOOR for SBP: {ALPHA_SBP_FLOOR}")

    key = jax.random.PRNGKey(123)

    my_crosscat_model = create_crosscat_model(
        num_rows_static=num_rows_val,
        num_cols_static=num_cols_val,
        max_views_static=max_views_static_val,
        max_categories_static=max_categories_val_for_model,  # Use the renamed variable
        max_cols_per_view_static=max_cols_per_view_val,
    )

    key_init_model, key_gibbs_alpha_d, key_gibbs_alpha_v_loop = jax.random.split(key, 3)

    model_args_tuple_for_init = (
        jnp.array(hyperparams_config.alpha_D_prior_shape, dtype=jnp.float32),
        jnp.array(hyperparams_config.alpha_D_prior_rate, dtype=jnp.float32),
        jnp.array(hyperparams_config.alpha_v_prior_shape, dtype=jnp.float32),
        jnp.array(hyperparams_config.alpha_v_prior_rate, dtype=jnp.float32),
        jnp.array(hyperparams_config.bern_alpha, dtype=jnp.float32),
        jnp.array(hyperparams_config.bern_beta, dtype=jnp.float32),
    )

    initial_trace, init_weight = my_crosscat_model.importance(
        key_init_model, observations_choicemap, model_args_tuple_for_init
    )
    print(
        f"Initial trace generated. Weight: {init_weight}, Score: {initial_trace.get_score()}"
    )
    current_trace = initial_trace

    if initial_trace.get_score() != -jnp.inf and not jnp.isnan(
        initial_trace.get_score()
    ):
        initial_alpha_D = current_trace.get_choices()["alpha_D"]
        print(f"  Initial sampled alpha_D: {initial_alpha_D}")
        for v_idx_print in range(max_views_static_val):
            alpha_v_key_to_print = f"alpha_v_{v_idx_print}"
            if alpha_v_key_to_print in current_trace.get_choices():
                print(
                    f"  Initial sampled {alpha_v_key_to_print}: {current_trace.get_choices()[alpha_v_key_to_print]}"
                )

        if num_cols_val > 0:
            print("\nTesting sample_alpha_D_gibbs_kernel (one step):")
            updated_trace_after_alpha_d_gibbs = sample_alpha_D_gibbs_kernel(
                key_gibbs_alpha_d,
                current_trace,
                num_cols_val,
                hyperparams_config.alpha_D_prior_shape,
                hyperparams_config.alpha_D_prior_rate,
                max_views_static_val,
                alpha_grid,
            )
            new_alpha_D = updated_trace_after_alpha_d_gibbs.get_choices()["alpha_D"]
            print(f"  alpha_D after one Gibbs step: {new_alpha_D}")
            print(
                f"  Score of trace after Gibbs update for alpha_D: {updated_trace_after_alpha_d_gibbs.get_score()}"
            )
            current_trace = updated_trace_after_alpha_d_gibbs
        else:
            print("\nSkipping sample_alpha_D_gibbs_kernel as num_cols_val is 0.")

        if max_views_static_val > 0 and num_rows_val > 0:
            key_gibbs_alpha_v_list = jax.random.split(
                key_gibbs_alpha_v_loop, max_views_static_val
            )
            for v_idx_update in range(max_views_static_val):
                alpha_v_key_str = f"alpha_v_{v_idx_update}"
                if alpha_v_key_str in current_trace.get_choices():
                    print(
                        f"\nTesting sample_alpha_v_kernel for view {v_idx_update} (one step):"
                    )
                    alpha_v_before_update = current_trace.get_choices()[alpha_v_key_str]
                    print(f"  {alpha_v_key_str} before update: {alpha_v_before_update}")

                    # Ensure max_categories_val_for_model is passed to the kernel
                    # if it's intended to be static for that kernel compilation.
                    # The kernel argument is named max_categories_val.
                    current_trace = sample_alpha_v_kernel(
                        key_gibbs_alpha_v_list[v_idx_update],
                        current_trace,
                        v_idx_update,
                        num_rows_val,
                        max_categories_val_for_model,  # Pass the correct variable
                        hyperparams_config.alpha_v_prior_shape,
                        hyperparams_config.alpha_v_prior_rate,
                        alpha_grid,
                    )
                    new_alpha_v = current_trace.get_choices()[alpha_v_key_str]
                    print(f"  {alpha_v_key_str} after one Gibbs step: {new_alpha_v}")
                    print(
                        f"  Score of trace after Gibbs update for {alpha_v_key_str}: {current_trace.get_score()}"
                    )
                else:
                    print(
                        f"\nSkipping sample_alpha_v_kernel for view {v_idx_update} as {alpha_v_key_str} not in choices."
                    )
        else:
            print(
                "\nSkipping sample_alpha_v_kernel as max_views_static_val or num_rows_val is 0."
            )
    else:
        print(
            "Skipping Gibbs kernel test due to invalid initial trace (score is -inf or NaN)."
        )
    print("=" * 30)


if __name__ == "__main__":
    test_crosscat_mcmc()

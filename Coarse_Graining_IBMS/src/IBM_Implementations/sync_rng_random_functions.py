"""A small set of helper functions to implement SyncRNG (e.g. testable against R) analogues of numpy.random functions"""

from SyncRNG import SyncRNG
import numpy as np


def sync_rng_choice(
    arr: np.ndarray,
    s: SyncRNG,
    size: tuple | int = 1,
    weights: np.ndarray | None = None,
    replace: bool = True,
):
    """Choose elements from an array, with or without replacement. Optional weights parameter as needed."""

    if replace:
        if weights is not None:
            raise NotImplementedError(
                "Sampling with replacement with weights isn't implemented yet"
            )

        # Preserve dtype of input array to match R behavior
        out_arr = np.empty(size, dtype=arr.dtype).ravel()
        flat_input_arr = arr.ravel()
        n_to_generate = out_arr.shape[0]
        for i in range(n_to_generate):
            idx = s.randi() % flat_input_arr.shape[0]
            out_arr[i] = flat_input_arr[idx]
        return out_arr.reshape(size)

    if isinstance(size, tuple):
        raise NotImplementedError(
            "sampling without replacement from multidimensional arrays isn't implemented yet"
        )

    if weights is not None:
        weights = weights / weights.sum()  # normalize

        chosen = []
        remaining_arr = list(arr)  # mutable copy of original array
        remaining_weights = list(weights)  # mutable copy of weights

        for _ in range(size):
            # Build cumulative probabilities (prefix sum starting from 0)
            cumulative_probabilities = np.array([
                np.sum(remaining_weights[:i]) for i in range(len(remaining_weights))
            ])
            # Find last index where cumulative < rand (inverse CDF sampling)
            idx = np.argwhere(cumulative_probabilities < s.rand())[-1][0]  # Extract scalar from 2D array
            chosen.append(remaining_arr[idx])

            # Remove chosen element and its weight by position
            del remaining_arr[idx]
            del remaining_weights[idx]

            # Renormalize remaining weights
            if len(remaining_weights) > 0:
                total = sum(remaining_weights)
                remaining_weights = [w / total for w in remaining_weights]

        out_arr = np.array(chosen)
    else:
        out_arr = s.shuffle(arr)[:size]

    return out_arr


def sync_rng_binomial(n: int, p: float, s: SyncRNG) -> int:
    """Returns a binomial distributed random number using the crudest direct simulation (e.g. n bernoulli trials)"""

    n_successes = 0

    for _ in range(n):
        n_successes += s.rand() < p

    return n_successes


def sync_rng_uniform(
    lower_bounds: np.ndarray | int,
    upper_bounds: np.ndarray | int,
    size: tuple | int,
    s: SyncRNG,
) -> np.ndarray:
    """Generate random variates within a range"""

    n_variates_to_generate = np.prod(np.array(size))

    out_array = np.empty(size).ravel()

    for i in range(n_variates_to_generate):
        out_array[i] = s.rand()

    out_array = out_array.reshape(size)

    widths = upper_bounds - lower_bounds

    out_array = lower_bounds + widths * out_array

    return out_array

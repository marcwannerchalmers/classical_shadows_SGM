"""Batch-compatible mean and median-of-means estimators."""

from flax import struct
from jax import Array
from jax import numpy as jnp
from jax.lax import cond
import numpy as np


@struct.dataclass
class MeanState:
    """Streaming state for arithmetic-mean prefix estimates."""

    total: Array
    count: Array
    values: Array


@struct.dataclass
class MedianOfMeansState:
    """Streaming state for median-of-means prefix estimates."""

    total: Array
    count: Array
    prefix_values: Array


@struct.dataclass
class Estimator:
    """Arithmetic-mean estimator with offline and streaming interfaces."""

    def __call__(
        self, x: Array, N: Array = jnp.array(0, dtype=jnp.int32)
    ) -> Array:
        """Average at most the first `N` samples on the final axis.

        Args:
            x: Values whose final axis is the sample axis.
            N: Prefix length. Nonpositive or oversized values use every sample.

        Returns:
            The arithmetic mean over the selected prefix.
        """

        len_x = x.shape[-1]
        N = cond(
            (N <= 0) | (len_x < N),
            lambda: jnp.asarray(len_x, dtype=N.dtype),
            lambda: N,
        )
        mask = jnp.arange(len_x, dtype=jnp.int32) < N
        return jnp.sum(x * mask, axis=-1, dtype=jnp.float32) / jnp.asarray(
            N, dtype=jnp.float32
        )

    def validate_Ns(self, Ns: Array) -> None:
        """Validate requested prefix sample counts.

        Args:
            Ns: One-dimensional positive prefix lengths.

        Raises:
            ValueError: If any prefix length is nonpositive.
        """

        if np.any(np.asarray(Ns) <= 0):
            raise ValueError("Requested sample counts must be positive")

    def init_state(self, value_shape: tuple[int, ...], Ns: Array) -> MeanState:
        """Initialize streaming accumulator state.

        Args:
            value_shape: Shape of one value before the sample axis.
            Ns: Prefix lengths whose estimates must be retained.

        Returns:
            Zero-filled accumulator state.
        """

        return MeanState(
            total=jnp.zeros(value_shape, dtype=jnp.float32),
            count=jnp.array(0, dtype=jnp.int32),
            values=jnp.zeros(value_shape + (Ns.shape[0],), dtype=jnp.float32),
        )

    def update(self, state: MeanState, x_batch: Array, Ns: Array) -> MeanState:
        """Consume one contiguous sample batch.

        Args:
            state: Accumulator returned by `init_state` or a prior update.
            x_batch: Values with the sample axis last.
            Ns: Requested global prefix lengths.

        Returns:
            Updated accumulator including any prefixes reached in this batch.
        """

        batch_prefix = jnp.cumsum(x_batch, axis=-1, dtype=jnp.float32)
        offsets = jnp.clip(Ns - state.count - 1, 0, x_batch.shape[-1] - 1)
        prefix_sums = state.total[..., None] + jnp.take(
            batch_prefix, offsets, axis=-1
        )
        reached = (Ns > state.count) & (
            Ns <= state.count + x_batch.shape[-1]
        )
        shape = (1,) * state.total.ndim + (Ns.shape[0],)
        values = jnp.where(
            reached.reshape(shape),
            prefix_sums / Ns.astype(jnp.float32).reshape(shape),
            state.values,
        )
        return MeanState(
            total=state.total + jnp.sum(x_batch, axis=-1, dtype=jnp.float32),
            count=state.count + x_batch.shape[-1],
            values=values,
        )

    def finalize(self, state: MeanState, Ns: Array) -> Array:
        """Return the retained estimates for every requested prefix.

        Args:
            state: Completed streaming accumulator.
            Ns: Requested prefix lengths, retained for interface consistency.

        Returns:
            Estimates with the prefix index on the final axis.
        """

        del Ns
        return state.values


@struct.dataclass
class MedianOfMeans(Estimator):
    """Median-of-means estimator using `k` nearly equal sample buckets.

    Attributes:
        k: Number of buckets for every requested prefix.
    """

    k: int = struct.field(pytree_node=False)

    def validate_Ns(self, Ns: Array) -> None:
        super().validate_Ns(Ns)
        if np.any(np.asarray(Ns) < self.k):
            raise ValueError("MedianOfMeans requires every requested N >= k")

    def _boundaries(self, Ns: Array) -> Array:
        Ns = jnp.asarray(Ns, dtype=jnp.int32)
        bucket = jnp.arange(self.k + 1, dtype=jnp.int32)
        quotient, remainder = Ns[:, None] // self.k, Ns[:, None] % self.k
        return bucket * quotient + jnp.minimum(bucket, remainder)

    def __call__(
        self, x: Array, N: Array = jnp.array(0, dtype=jnp.int32)
    ) -> Array:
        len_x = x.shape[-1]
        N = cond(
            (N <= 0) | (len_x < N),
            lambda: jnp.asarray(len_x, dtype=N.dtype),
            lambda: N,
        )
        boundaries = self._boundaries(jnp.asarray([N]))[0]
        prefix = jnp.concatenate(
            [jnp.zeros(x.shape[:-1] + (1,), x.dtype), jnp.cumsum(x, axis=-1)],
            axis=-1,
        )
        bucket_sums = jnp.diff(jnp.take(prefix, boundaries, axis=-1), axis=-1)
        bucket_sizes = jnp.diff(boundaries).astype(jnp.float32)
        return jnp.median(bucket_sums / bucket_sizes, axis=-1)

    def init_state(
        self, value_shape: tuple[int, ...], Ns: Array
    ) -> MedianOfMeansState:
        return MedianOfMeansState(
            total=jnp.zeros(value_shape, dtype=jnp.float32),
            count=jnp.array(0, dtype=jnp.int32),
            prefix_values=jnp.zeros(
                value_shape + (Ns.shape[0], self.k + 1), dtype=jnp.float32
            ),
        )

    def update(
        self, state: MedianOfMeansState, x_batch: Array, Ns: Array
    ) -> MedianOfMeansState:
        boundaries = self._boundaries(Ns)
        batch_prefix = jnp.cumsum(x_batch, axis=-1, dtype=jnp.float32)
        offsets = jnp.clip(
            boundaries - state.count - 1, 0, x_batch.shape[-1] - 1
        )
        prefix_sums = state.total[..., None, None] + jnp.take(
            batch_prefix, offsets.reshape(-1), axis=-1
        ).reshape(state.prefix_values.shape)
        reached = (boundaries > state.count) & (
            boundaries <= state.count + x_batch.shape[-1]
        )
        shape = (1,) * state.total.ndim + reached.shape
        return MedianOfMeansState(
            total=state.total + jnp.sum(x_batch, axis=-1, dtype=jnp.float32),
            count=state.count + x_batch.shape[-1],
            prefix_values=jnp.where(
                reached.reshape(shape), prefix_sums, state.prefix_values
            ),
        )

    def finalize(self, state: MedianOfMeansState, Ns: Array) -> Array:
        boundaries = self._boundaries(Ns)
        bucket_sums = jnp.diff(state.prefix_values, axis=-1)
        bucket_sizes = jnp.diff(boundaries, axis=-1).astype(jnp.float32)
        shape = (1,) * state.total.ndim + bucket_sizes.shape
        return jnp.median(bucket_sums / bucket_sizes.reshape(shape), axis=-1)

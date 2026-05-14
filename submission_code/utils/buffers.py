"""Replay buffers for online continual learning.

RandomReplayBuffer:  FIFO circular buffer, uniform sampling.
PriorityReplayBuffer: Fixed-capacity buffer with priority scores, proportional sampling.

Both implementations use pre-allocated NumPy arrays for O(1) indexing and
vectorized batch assembly — significantly faster than list-based approaches
at capacity > 200.
"""
import numpy as np


class RandomReplayBuffer:
    """Circular FIFO buffer with uniform random sampling.

    Uses pre-allocated arrays: push and sample are both O(batch_size).
    """

    def __init__(self, capacity: int = 500):
        self.capacity  = capacity
        self._inputs:  np.ndarray | None = None
        self._targets: np.ndarray | None = None
        self._pos  = 0
        self._size = 0

    def __len__(self) -> int:
        return self._size

    def clear(self) -> None:
        """Drop all stored samples."""
        self._size = 0
        self._pos = 0

    def push(self, inp: np.ndarray, tgt: np.ndarray, **kwargs) -> None:
        """Add a sample (overwrites oldest when full)."""
        inp = np.asarray(inp, dtype=np.float32)
        tgt = np.asarray(tgt, dtype=np.float32)
        if self._inputs is None:
            self._inputs  = np.empty((self.capacity, inp.shape[0]), dtype=np.float32)
            self._targets = np.empty((self.capacity, tgt.shape[0]), dtype=np.float32)
        self._inputs[self._pos]  = inp
        self._targets[self._pos] = tgt
        self._pos  = (self._pos + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    @property
    def inputs(self) -> np.ndarray | None:
        return self._inputs

    @property
    def targets(self) -> np.ndarray | None:
        return self._targets

    def sample(self, batch_size: int):
        """Return uniformly random (inputs, targets) arrays of shape (B, D)."""
        batch_size = min(batch_size, self._size)
        idx = np.random.choice(self._size, size=batch_size, replace=False)
        return self._inputs[idx], self._targets[idx]


class PriorityReplayBuffer:
    """Fixed-capacity buffer with priority-proportional sampling.

    When full, new samples replace the lowest-priority sample only if the
    incoming priority is higher (informativeness-gated insertion).

    Pre-allocated arrays and vectorised argmin/choice for efficiency.
    """

    def __init__(self, capacity: int = 500, sampling_alpha: float = 1.0):
        self.capacity   = capacity
        self._inputs:   np.ndarray | None = None
        self._targets:  np.ndarray | None = None
        self._priorities = np.full(capacity, -np.inf, dtype=np.float64)
        self._size  = 0
        self._min_idx = 0  # cached index of minimum priority
        # Softened priority sampling. alpha=1 reproduces the original
        # priority-proportional behaviour byte-for-byte (the alpha branch
        # below is only entered when alpha != 1.0). alpha in (0, 1) makes
        # the sampler more uniform; alpha > 1 makes it more peaked.
        self.sampling_alpha = float(sampling_alpha)

    def __len__(self) -> int:
        return self._size

    def clear(self) -> None:
        """Drop all stored samples (used by ACE on entry to
        preserve-mode after adaptation, to discard stale shift-era
        data so the next adaptation phase starts from clean buffer)."""
        self._size = 0
        self._priorities[:] = -np.inf
        self._min_idx = 0

    def push(self, inp: np.ndarray, tgt: np.ndarray, priority: float = 1.0) -> None:
        """Add sample. If full, replace the lowest-priority slot."""
        inp = np.asarray(inp, dtype=np.float32)
        tgt = np.asarray(tgt, dtype=np.float32)
        if self._inputs is None:
            self._inputs  = np.empty((self.capacity, inp.shape[0]), dtype=np.float32)
            self._targets = np.empty((self.capacity, tgt.shape[0]), dtype=np.float32)

        if self._size < self.capacity:
            idx = self._size
            self._size += 1
        else:
            # Replace lowest priority only if new sample is more informative
            if priority <= self._priorities[self._min_idx]:
                return
            idx = self._min_idx

        self._inputs[idx]   = inp
        self._targets[idx]  = tgt
        self._priorities[idx] = priority
        # Update cached min index (O(n) but n≤500, negligible)
        self._min_idx = int(np.argmin(self._priorities[:self._size]))

    def sample(self, batch_size: int):
        """Return (inputs, targets) sampled proportional to priority."""
        n          = self._size
        batch_size = min(batch_size, n)
        p = self._priorities[:n].copy()
        p -= p.min() - 1e-9          # shift so all weights > 0
        if self.sampling_alpha != 1.0:
            # Softened (or sharpened) priority sampling. Power must come
            # AFTER the shift so it operates on positive numbers; otherwise
            # priorities that round to negative would become NaN under
            # fractional powers.
            p = np.power(p, self.sampling_alpha)
        p /= p.sum()
        idx = np.random.choice(n, size=batch_size, replace=False, p=p)
        return self._inputs[idx], self._targets[idx]

    @property
    def inputs(self) -> np.ndarray | None:
        return self._inputs

    @property
    def targets(self) -> np.ndarray | None:
        return self._targets

    @property
    def priorities(self) -> np.ndarray:
        return self._priorities[:self._size]

    def update_priority(self, idx: int, priority: float) -> None:
        """Update priority of an existing slot (for PER-style updates)."""
        if 0 <= idx < self._size:
            self._priorities[idx] = priority
            self._min_idx = int(np.argmin(self._priorities[:self._size]))

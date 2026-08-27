"""The ledger of managed tensors (Layer 2).

The registry answers "what is resident, how big is it, when was it last used,
and what would it cost to get it back" for every tensor VOptimizer manages. It
stores metadata only: it never moves a tensor and never scores one.

Entries hold a weak reference to the tensor, so registering a tensor cannot
keep it alive. Metadata outlives the tensor on purpose: once the offload
manager copies a tensor to CPU or NVMe and frees the GPU copy, the ledger still
has to know the entry exists, how big it was and what it would cost to bring
back. Removal is therefore always explicit, via :meth:`TensorRegistry.unregister`
or :meth:`TensorRegistry.prune`.
"""

from __future__ import annotations

import itertools
import time
import weakref
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable

import torch


class Location(str, Enum):
    GPU = "gpu"
    CPU = "cpu"
    NVME = "nvme"


class TensorKind(str, Enum):
    """What the tensor is for. Determines which manager may act on it."""

    WEIGHT = "weight"
    ACTIVATION = "activation"
    OPTIMIZER_STATE = "optimizer_state"
    KV_BLOCK = "kv_block"
    OTHER = "other"


@dataclass
class TensorMeta:
    """Everything the decision layer needs to reason about one tensor."""

    key: str
    size_bytes: int
    kind: TensorKind
    location: Location
    dtype: torch.dtype | None = None
    shape: tuple[int, ...] = ()
    owner: str | None = None
    recompute_cost_s: float = 0.0
    pinned: bool = False
    created_at: float = 0.0
    last_access: float = 0.0
    access_count: int = 0

    @property
    def size_gb(self) -> float:
        return self.size_bytes / float(1024**3)

    def age(self, now: float) -> float:
        """Seconds since last access."""
        return max(0.0, now - self.last_access)


class TensorRegistry:
    """Thread-unsafe ledger keyed by a caller-supplied or generated key.

    Args:
        time_fn: Injectable clock, so age-based queries are deterministic in
            tests.
    """

    def __init__(self, time_fn: Callable[[], float] = time.monotonic) -> None:
        self._time_fn = time_fn
        self._entries: dict[str, TensorMeta] = {}
        self._refs: dict[str, weakref.ref[torch.Tensor]] = {}
        self._counter = itertools.count()

    def __len__(self) -> int:
        return len(self._entries)

    def __contains__(self, key: object) -> bool:
        return key in self._entries

    def __iter__(self) -> Iterator[TensorMeta]:
        return iter(list(self._entries.values()))

    def register(
        self,
        tensor: torch.Tensor,
        kind: TensorKind = TensorKind.OTHER,
        key: str | None = None,
        owner: str | None = None,
        recompute_cost_s: float = 0.0,
        pinned: bool = False,
    ) -> TensorMeta:
        """Add a tensor to the ledger, replacing any entry with the same key."""
        key = key or f"{kind.value}:{next(self._counter)}"
        now = self._time_fn()
        meta = TensorMeta(
            key=key,
            size_bytes=tensor.numel() * tensor.element_size(),
            kind=kind,
            location=self._location_of(tensor),
            dtype=tensor.dtype,
            shape=tuple(tensor.shape),
            owner=owner,
            recompute_cost_s=recompute_cost_s,
            pinned=pinned,
            created_at=now,
            last_access=now,
            access_count=0,
        )
        self._entries[key] = meta
        self.attach(key, tensor)
        return meta

    def attach(self, key: str, tensor: torch.Tensor | None) -> None:
        """Point an existing entry at the tensor that currently holds its data.

        The action layer calls this after materializing a tensor on a new tier,
        so the ledger hands out the live object rather than a stale one.
        """
        if tensor is None:
            self._refs.pop(key, None)
            return
        try:
            self._refs[key] = weakref.ref(tensor)
        except TypeError:
            # Some tensor subclasses are not weak-referenceable.
            self._refs.pop(key, None)

    def unregister(self, key: str) -> TensorMeta | None:
        self._refs.pop(key, None)
        return self._entries.pop(key, None)

    def get(self, key: str) -> TensorMeta | None:
        return self._entries.get(key)

    def tensor(self, key: str) -> torch.Tensor | None:
        """The live tensor for ``key``, or None if it was collected or is
        currently held outside GPU memory by a manager."""
        ref = self._refs.get(key)
        return ref() if ref is not None else None

    def touch(self, key: str) -> TensorMeta | None:
        """Record an access. Drives the age and frequency terms of scoring."""
        meta = self._entries.get(key)
        if meta is None:
            return None
        meta.last_access = self._time_fn()
        meta.access_count += 1
        return meta

    def set_location(self, key: str, location: Location) -> TensorMeta | None:
        """Called by the action layer after a tensor finishes moving tiers."""
        meta = self._entries.get(key)
        if meta is None:
            return None
        meta.location = location
        return meta

    def entries(
        self,
        location: Location | None = None,
        kind: TensorKind | None = None,
        include_pinned: bool = True,
    ) -> list[TensorMeta]:
        return [
            meta
            for meta in self._entries.values()
            if (location is None or meta.location is location)
            and (kind is None or meta.kind is kind)
            and (include_pinned or not meta.pinned)
        ]

    def total_bytes(self, location: Location | None = None, kind: TensorKind | None = None) -> int:
        return sum(meta.size_bytes for meta in self.entries(location, kind))

    def evictable(self, location: Location = Location.GPU) -> list[TensorMeta]:
        """Entries a manager is allowed to move out of ``location``."""
        return self.entries(location=location, include_pinned=False)

    def oldest(
        self, count: int, location: Location = Location.GPU, include_pinned: bool = False
    ) -> list[TensorMeta]:
        """The ``count`` least recently accessed entries, oldest first."""
        candidates = self.entries(location=location, include_pinned=include_pinned)
        candidates.sort(key=lambda meta: meta.last_access)
        return candidates[:count]

    def clear(self) -> None:
        self._entries.clear()
        self._refs.clear()

    def prune(self) -> list[str]:
        """Drop GPU entries whose tensor was collected. Returns removed keys.

        Entries living on CPU or NVMe are left alone: their data is owned by a
        manager, not by a live ``torch.Tensor``.
        """
        dead = [
            key
            for key, meta in self._entries.items()
            if meta.location is Location.GPU and self.tensor(key) is None
        ]
        for key in dead:
            self.unregister(key)
        return dead

    def summary(self) -> dict[str, int]:
        """Bytes per location, for logging and tests."""
        return {location.value: self.total_bytes(location) for location in Location}

    def register_all(
        self, tensors: Iterable[tuple[str, torch.Tensor]], kind: TensorKind
    ) -> list[TensorMeta]:
        return [self.register(tensor, kind=kind, key=key) for key, tensor in tensors]

    @staticmethod
    def _location_of(tensor: torch.Tensor) -> Location:
        return Location.GPU if tensor.device.type == "cuda" else Location.CPU


@dataclass
class RegistrySnapshot:
    """Immutable view of the ledger, safe to hand to the decision layer."""

    entries: tuple[TensorMeta, ...] = field(default_factory=tuple)
    taken_at: float = 0.0

    @classmethod
    def of(cls, registry: TensorRegistry, now: float | None = None) -> RegistrySnapshot:
        return cls(
            entries=tuple(registry),
            taken_at=now if now is not None else time.monotonic(),
        )

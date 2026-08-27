import gc

import torch

from voptimizer import Location, RegistrySnapshot, TensorKind, TensorRegistry


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def test_register_captures_size_and_shape():
    registry = TensorRegistry()
    meta = registry.register(torch.zeros(4, 8, dtype=torch.float32), kind=TensorKind.WEIGHT)

    assert meta.size_bytes == 4 * 8 * 4
    assert meta.shape == (4, 8)
    assert meta.dtype is torch.float32
    assert meta.location is Location.CPU
    assert meta.key in registry
    assert len(registry) == 1


def test_generated_keys_are_unique():
    registry = TensorRegistry()
    keys = {registry.register(torch.zeros(1), kind=TensorKind.KV_BLOCK).key for _ in range(5)}
    assert len(keys) == 5


def test_registering_same_key_replaces_entry():
    registry = TensorRegistry()
    registry.register(torch.zeros(10), key="layer.0")
    registry.register(torch.zeros(20), key="layer.0")

    assert len(registry) == 1
    assert registry.get("layer.0").shape == (20,)


def test_touch_updates_age_and_frequency():
    clock = FakeClock()
    registry = TensorRegistry(time_fn=clock)
    meta = registry.register(torch.zeros(4), key="a")

    clock.advance(5.0)
    assert meta.age(clock.now) == 5.0

    registry.touch("a")
    assert meta.access_count == 1
    assert meta.age(clock.now) == 0.0
    assert registry.touch("missing") is None


def test_set_location_moves_bytes_between_tiers():
    registry = TensorRegistry()
    registry.register(torch.zeros(256), key="w")
    assert registry.total_bytes(Location.CPU) == 1024

    registry.set_location("w", Location.NVME)
    assert registry.total_bytes(Location.CPU) == 0
    assert registry.total_bytes(Location.NVME) == 1024
    assert registry.summary() == {"gpu": 0, "cpu": 0, "nvme": 1024}
    assert registry.set_location("missing", Location.GPU) is None


def test_entries_filter_by_kind_and_pinning():
    registry = TensorRegistry()
    registry.register(torch.zeros(1), kind=TensorKind.WEIGHT, key="w", pinned=True)
    registry.register(torch.zeros(1), kind=TensorKind.ACTIVATION, key="a")

    assert [m.key for m in registry.entries(kind=TensorKind.WEIGHT)] == ["w"]
    assert [m.key for m in registry.entries(include_pinned=False)] == ["a"]
    assert registry.evictable(Location.CPU) == registry.entries(kind=TensorKind.ACTIVATION)


def test_oldest_returns_least_recently_accessed_first():
    clock = FakeClock()
    registry = TensorRegistry(time_fn=clock)
    for key in ("a", "b", "c"):
        registry.register(torch.zeros(1), key=key)
        clock.advance(1.0)

    registry.touch("a")
    assert [m.key for m in registry.oldest(2, location=Location.CPU)] == ["b", "c"]


def test_pinned_entries_are_never_evictable():
    registry = TensorRegistry()
    registry.register(torch.zeros(1), key="pinned", pinned=True)
    assert registry.oldest(5, location=Location.CPU) == []
    assert len(registry.oldest(5, location=Location.CPU, include_pinned=True)) == 1


def test_registry_does_not_keep_the_tensor_alive():
    registry = TensorRegistry()
    tensor = torch.zeros(64)
    registry.register(tensor, key="temp")
    assert registry.tensor("temp") is tensor

    del tensor
    gc.collect()

    assert registry.tensor("temp") is None
    # Metadata survives: an offloaded tensor has no live GPU object either.
    assert registry.get("temp").size_bytes == 256


def test_prune_drops_collected_gpu_entries_only():
    registry = TensorRegistry()
    registry.register(torch.zeros(4), key="gpu-gone")
    registry.set_location("gpu-gone", Location.GPU)
    registry.register(torch.zeros(4), key="offloaded")
    registry.set_location("offloaded", Location.NVME)
    gc.collect()

    assert registry.prune() == ["gpu-gone"]
    assert "offloaded" in registry


def test_attach_rebinds_an_entry_to_a_new_tensor():
    registry = TensorRegistry()
    registry.register(torch.zeros(4), key="w")
    reloaded = torch.ones(4)
    registry.attach("w", reloaded)
    assert registry.tensor("w") is reloaded

    registry.attach("w", None)
    assert registry.tensor("w") is None


def test_unregister_and_clear():
    registry = TensorRegistry()
    registry.register(torch.zeros(1), key="a")
    assert registry.unregister("a").key == "a"
    assert registry.unregister("a") is None

    registry.register(torch.zeros(1), key="b")
    registry.clear()
    assert len(registry) == 0


def test_register_all_and_snapshot():
    registry = TensorRegistry()
    tensors = [("layer.0", torch.zeros(8)), ("layer.1", torch.zeros(8))]
    registry.register_all(tensors, kind=TensorKind.WEIGHT)

    snapshot = RegistrySnapshot.of(registry, now=1.0)
    assert {m.key for m in snapshot.entries} == {"layer.0", "layer.1"}
    assert snapshot.taken_at == 1.0

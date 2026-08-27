import pytest

from voptimizer import PressureThresholds, VOptimizerConfig


def test_defaults_and_derived_values():
    config = VOptimizerConfig(target_vram_gb=16.0)
    assert config.target_vram_bytes == 16 * 1024**3
    assert config.nvme_offload_enabled is False
    assert config.thresholds.moderate == 0.70


def test_nvme_enabled_when_path_set(tmp_path):
    config = VOptimizerConfig(target_vram_gb=8.0, nvme_offload_path=str(tmp_path))
    assert config.nvme_offload_enabled is True


@pytest.mark.parametrize(
    "kwargs",
    [
        {"target_vram_gb": 0},
        {"target_vram_gb": -1.0},
        {"target_vram_gb": 8.0, "latency_tolerance": "extreme"},
        {"target_vram_gb": 8.0, "throughput_priority": "extreme"},
        {"target_vram_gb": 8.0, "warmup_steps": -1},
        {"target_vram_gb": 8.0, "prefetch_window": -1},
        {"target_vram_gb": 8.0, "poll_interval_s": -0.1},
        {"target_vram_gb": 8.0, "device_index": -1},
    ],
)
def test_invalid_config_rejected(kwargs):
    with pytest.raises(ValueError):
        VOptimizerConfig(**kwargs)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"moderate": 0.9, "high": 0.85},
        {"critical": 1.5},
        {"moderate": 0.0},
        {"hysteresis": -0.1},
        {"hysteresis": 0.8},
    ],
)
def test_invalid_thresholds_rejected(kwargs):
    with pytest.raises(ValueError):
        PressureThresholds(**kwargs)

# VOptimizer

> **Treat GPU VRAM the way an operating system treats RAM — as a managed hierarchy, not a fixed allocation.**

VOptimizer is a runtime memory management system for large AI models. It runs on top of PyTorch and unifies existing memory techniques — offloading, activation checkpointing, KV cache paging, layer streaming — under a single adaptive control plane. No model weight changes. No accuracy loss.

---

## The Problem

Large models fail on limited hardware not because of compute, but because GPU memory is exhausted by multiple competing consumers:

| Memory Consumer | Example Scale |
|---|---|
| Model weights (20B, fp16) | ~40 GB |
| KV cache (32k context) | ~20–40 GB |
| Activations (large batch) | ~10–30 GB |
| Optimizer states (Adam) | ~2× weight size |
| Intermediate buffers | ~1–5 GB |

Existing tools solve parts of this in isolation:

- **DeepSpeed ZeRO** — partitions optimizer states
- **PagedAttention (vLLM)** — pages KV cache blocks
- **Activation checkpointing** — recomputes activations
- **CPU offloading** — moves tensors out of VRAM

None of them coordinate. VOptimizer coordinates all of them through one runtime layer.

---

## How It Works

```
User sets:   target_vram = 16GB, latency_tolerance = medium
                        │
                        ▼
            ┌─────────────────────┐
            │   VRAMMonitor       │  ← watches GPU pressure continuously
            └─────────┬───────────┘
                      │ pressure level
                      ▼
            ┌─────────────────────┐
            │   PolicyEngine      │  ← decides WHAT to do
            └──┬──────┬──────┬───┘
               │      │      │
               ▼      ▼      ▼
        Offload   Checkpoint   KVCache
        Manager   Manager      Scheduler
               │      │      │
               ▼      ▼      ▼
         GPU ↔ CPU ↔ NVMe    Paged blocks
```

The system operates at four pressure levels:

| Pressure | VRAM Used | Action |
|---|---|---|
| `normal` | < 70% | Nothing — full speed |
| `moderate` | 70–85% | Offload optimizer states, age-evict old KV blocks |
| `high` | 85–95% | Enable activation checkpointing, stream layers |
| `critical` | > 95% | Score all tensors, evict top-N by priority score |

---

## Architecture

VOptimizer is structured in four strict layers. Each layer has exactly one responsibility and never crosses into another layer's domain.

### Layer 1 — User Interface Layer
The developer-facing API. You configure targets here and never touch internals.

- `config.py` — `VOptimizerConfig` dataclass: `target_vram_gb`, `latency_tolerance`, `throughput_priority`
- `voptimizer.py` — `VOptimizer` main class: `.wrap(model)` is the single entry point

### Layer 2 — Observation Layer
Watches everything passively. Feeds data upward. Never makes decisions.

- `monitor.py` — `VRAMMonitor`: polls `torch.cuda.memory_allocated()`, maps to pressure level
- `registry.py` — `TensorRegistry`: ledger of every managed tensor with size, location, age, recompute cost
- `planner.py` — `ExecutionPlanner`: traces layer execution order via one dummy forward pass at startup

### Layer 3 — Decision Layer
The brain. Takes observations, applies policy, emits action commands. Never moves tensors itself.

- `policy_engine.py` — `PolicyEngine`: maps pressure level → strategy selection
- `tuner.py` — `WeightTuner`: self-calibrates eviction score weights during a warmup phase, then locks

### Layer 4 — Action Layer
Executes decisions. Each manager owns one concern only.

- `offload_manager.py` — `OffloadManager`: moves tensors GPU ↔ CPU ↔ NVMe via async CUDA streams
- `checkpoint_manager.py` — `CheckpointManager`: wraps layer segments with `torch.utils.checkpoint`
- `kv_scheduler.py` — `KVCacheScheduler`: manages paged KV blocks, evicts by age under pressure

### Integration Layer
The only layer that touches PyTorch hooks. Isolates all hook complexity in one place.

- `hooks.py` — `ModuleHookManager`: attaches `forward_pre_hook` and `forward_hook` to all leaf modules; triggers prefetch before each layer, triggers policy cycle after

---

## Eviction Scoring

When pressure is critical, every managed tensor gets a score. Higher score = evict first.

```python
score = (size_gb       × weight_size)       # bigger = more valuable to free
      + (time_since_access × weight_age)    # older = safer to evict
      - (recompute_cost × weight_recompute) # expensive to recompute = keep
      - (access_frequency × weight_freq)    # frequently used = keep
```

Weights start at calibrated defaults and self-adjust during a configurable warmup phase based on observed latency and memory freed per action. After warmup, weights lock in for the rest of the run.

---

## Project Structure

```
voptimizer/
├── __init__.py              # exports VOptimizer, VOptimizerConfig
├── config.py                # VOptimizerConfig dataclass
├── voptimizer.py            # main API — wires all subsystems together
│
├── monitor.py               # VRAMMonitor — hardware watcher
├── registry.py              # TensorRegistry, TensorMeta
├── planner.py               # ExecutionPlanner — layer order tracer
│
├── policy_engine.py         # PolicyEngine — core decision brain
├── tuner.py                 # WeightTuner — adaptive score calibration
│
├── offload_manager.py       # OffloadManager — GPU ↔ CPU ↔ NVMe
├── checkpoint_manager.py    # CheckpointManager — activation recompute
└── kv_scheduler.py          # KVCacheScheduler — paged KV blocks

tests/
├── test_monitor.py          # mock pressure injection tests
├── test_engine.py           # pressure level → action tests
├── test_offload.py          # GPU ↔ CPU movement tests
└── test_integration.py      # full wrap() smoke tests
```

---

## Setup

### Prerequisites

- Python 3.9+
- PyTorch 2.0+ (with CUDA support for GPU usage)
- Git

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/voptimizer.git
cd voptimizer
```

### 2. Create a virtual environment

```bash
python -m venv venv

# Linux / macOS
source venv/bin/activate

# Windows
venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

> **Note:** `requirements.txt` will be populated as modules are built. Core dependencies will be `torch`, `numpy`, and `psutil`. No external ML frameworks are required.

### 4. Verify installation

```bash
python -c "import voptimizer; print('VOptimizer ready')"
```

### 5. Run tests

```bash
pytest tests/ -v
```

---

## Usage (Target API)

```python
from voptimizer import VOptimizer, VOptimizerConfig

# 1. Define your memory target
config = VOptimizerConfig(
    target_vram_gb=16.0,
    latency_tolerance="medium",   # "low" | "medium" | "high"
    throughput_priority="high",
    warmup_steps=20,
    prefetch_window=2,
)

# 2. Wrap your model — that's it
optimizer = VOptimizer(config)
model = optimizer.wrap(model)

# 3. Run inference or training as normal
# VOptimizer manages memory in the background
output = model(input_ids)
```

The model runs identically to before. VOptimizer intercepts execution transparently via PyTorch hooks.

---

## Development Status

This is an active research prototype. The build order follows strict dependency sequencing:

- [ ] Phase 1 — Foundation: `config.py`, `monitor.py`, `registry.py`
- [ ] Phase 2 — Integration: `planner.py`, `hooks.py`
- [ ] Phase 3 — Decision: `policy_engine.py`, `tuner.py`
- [ ] Phase 4 — Action: `offload_manager.py`, `kv_scheduler.py`, `checkpoint_manager.py`
- [ ] Phase 5 — Assembly: `voptimizer.py`, integration tests, benchmarks

---

## Key Design Principles

**Separation of concerns is non-negotiable.** Observation layers never decide. Decision layers never move tensors. Action layers never read pressure. This is what makes the system debuggable at scale.

**No new primitives.** VOptimizer does not invent new memory techniques. It coordinates existing, proven ones — offloading, checkpointing, paging, streaming — under one policy layer.

**Adaptive, not static.** Eviction strategies adjust during warmup based on observed hardware behavior. The system tunes itself to your specific GPU and workload, then locks in the learned behavior.

**Testable by design.** `VRAMMonitor` accepts a `mock_pressure` parameter. Every pressure scenario is reproducible in unit tests without real GPU memory pressure.

---

## Motivation

As model sizes grow, memory — not compute — is the primary deployment bottleneck. A 20B parameter model requires ~40GB in fp16 just for weights, before any activations or KV cache. Consumer and edge hardware tops out at 8–24GB.

The gap is not bridged by better GPUs alone. It requires treating model execution as a memory-scheduled runtime — exactly what operating systems have done with RAM for decades.

VOptimizer applies that principle to GPU VRAM.

---

## Contributing

This project is in early development. Architecture decisions are being finalized before code modules are merged. If you want to contribute:

1. Read the architecture overview above fully before opening a PR
2. One file per component — no exceptions
3. Action layer modules must not import from the Decision layer directly
4. Every new module requires a corresponding test file

---

## License

MIT License. See `LICENSE` for details.

---

## Author

Built by Koushik — exploring the intersection of systems engineering and AI inference optimization.

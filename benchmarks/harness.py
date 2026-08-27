"""Benchmark harness: peak VRAM and throughput for a model runner.

The only claim worth making about VOptimizer is a measured one: does a given
memory strategy run a model that would otherwise OOM, and what does it cost in
tokens/sec? This harness produces those two numbers for any callable, so every
strategy is compared against the same baseline on the same hardware.

Run the built-in smoke benchmark:

    python -m benchmarks.harness --layers 8 --hidden 1024 --steps 20
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from typing import Callable

import torch

_BYTES_PER_MB = float(1024**2)

Runner = Callable[[], int]
"""A single benchmark step. Returns the number of tokens it processed."""


@dataclass
class BenchmarkResult:
    name: str
    steps: int
    tokens: int
    wall_s: float
    step_times_s: list[float] = field(default_factory=list)
    peak_allocated_mb: float = 0.0
    peak_reserved_mb: float = 0.0
    device: str = "cpu"

    @property
    def tokens_per_s(self) -> float:
        return self.tokens / self.wall_s if self.wall_s > 0 else 0.0

    @property
    def median_step_ms(self) -> float:
        return statistics.median(self.step_times_s) * 1000 if self.step_times_s else 0.0

    @property
    def p95_step_ms(self) -> float:
        """Tail latency. Offload strategies show their stalls here, not in the
        median, so a median-only comparison flatters them."""
        if not self.step_times_s:
            return 0.0
        ordered = sorted(self.step_times_s)
        return ordered[min(len(ordered) - 1, int(0.95 * len(ordered)))] * 1000

    def as_dict(self) -> dict[str, object]:
        data = asdict(self)
        data.pop("step_times_s")
        data.update(
            tokens_per_s=round(self.tokens_per_s, 2),
            median_step_ms=round(self.median_step_ms, 3),
            p95_step_ms=round(self.p95_step_ms, 3),
        )
        return data

    def format(self) -> str:
        return (
            f"{self.name:<24} {self.tokens_per_s:>10.1f} tok/s  "
            f"median {self.median_step_ms:>8.2f} ms  p95 {self.p95_step_ms:>8.2f} ms  "
            f"peak alloc {self.peak_allocated_mb:>8.1f} MB  "
            f"peak reserved {self.peak_reserved_mb:>8.1f} MB"
        )


def run_benchmark(
    name: str,
    runner: Runner,
    steps: int = 20,
    warmup_steps: int = 3,
    device: torch.device | str = "cpu",
) -> BenchmarkResult:
    """Time ``runner`` and record peak memory.

    Warmup steps are excluded from timings and memory peaks: the first
    iterations pay for allocator growth and kernel autotuning, which would
    otherwise be attributed to the strategy under test.
    """
    device = torch.device(device)
    is_cuda = device.type == "cuda"

    for _ in range(warmup_steps):
        runner()
    _synchronize(is_cuda)

    if is_cuda:
        torch.cuda.reset_peak_memory_stats(device)

    tokens = 0
    step_times: list[float] = []
    started = time.perf_counter()
    for _ in range(steps):
        step_started = time.perf_counter()
        tokens += runner()
        _synchronize(is_cuda)
        step_times.append(time.perf_counter() - step_started)
    wall = time.perf_counter() - started

    return BenchmarkResult(
        name=name,
        steps=steps,
        tokens=tokens,
        wall_s=wall,
        step_times_s=step_times,
        peak_allocated_mb=(
            torch.cuda.max_memory_allocated(device) / _BYTES_PER_MB if is_cuda else 0.0
        ),
        peak_reserved_mb=(
            torch.cuda.max_memory_reserved(device) / _BYTES_PER_MB if is_cuda else 0.0
        ),
        device=str(device),
    )


def compare(results: Sequence[BenchmarkResult], baseline: str | None = None) -> str:
    """Table of results, with slowdown and memory saving against a baseline."""
    lines = [result.format() for result in results]
    reference = next((r for r in results if r.name == baseline), None)
    if reference is not None:
        lines.append("")
        for result in results:
            if result is reference or reference.tokens_per_s == 0:
                continue
            slowdown = reference.tokens_per_s / result.tokens_per_s if result.tokens_per_s else 0.0
            saved = reference.peak_reserved_mb - result.peak_reserved_mb
            lines.append(
                f"{result.name} vs {reference.name}: "
                f"{slowdown:.2f}x slower, {saved:+.1f} MB peak reserved"
            )
    return "\n".join(lines)


class _StackedMLP(torch.nn.Module):
    """Stand-in for a transformer stack: uniform layers, known execution order.

    Enough to exercise the harness and, later, layer streaming, without pulling
    in a model library.
    """

    def __init__(self, layers: int, hidden: int) -> None:
        super().__init__()
        self.blocks = torch.nn.ModuleList(
            torch.nn.Sequential(
                torch.nn.Linear(hidden, hidden),
                torch.nn.GELU(),
                torch.nn.Linear(hidden, hidden),
            )
            for _ in range(layers)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = x + block(x)
        return x


def _synchronize(is_cuda: bool) -> None:
    if is_cuda:
        torch.cuda.synchronize()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--layers", type=int, default=8)
    parser.add_argument("--hidden", type=int, default=1024)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--seq", type=int, default=128)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--warmup-steps", type=int, default=3)
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to benchmark on.",
    )
    parser.add_argument("--json", action="store_true", help="Emit JSON instead of a table.")
    args = parser.parse_args(argv)

    device = torch.device(args.device)
    torch.manual_seed(0)
    model = _StackedMLP(args.layers, args.hidden).to(device).eval()
    inputs = torch.randn(args.batch, args.seq, args.hidden, device=device)
    tokens_per_step = args.batch * args.seq

    def runner() -> int:
        with torch.no_grad():
            model(inputs)
        return tokens_per_step

    result = run_benchmark(
        "baseline:resident",
        runner,
        steps=args.steps,
        warmup_steps=args.warmup_steps,
        device=device,
    )
    print(json.dumps(result.as_dict(), indent=2) if args.json else compare([result]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

import json

from benchmarks.harness import BenchmarkResult, compare, main, run_benchmark


def test_run_benchmark_counts_tokens_and_excludes_warmup():
    calls = []

    def runner() -> int:
        calls.append(1)
        return 32

    result = run_benchmark("test", runner, steps=5, warmup_steps=2)

    assert len(calls) == 7
    assert result.steps == 5
    assert result.tokens == 160
    assert result.tokens_per_s > 0
    assert len(result.step_times_s) == 5
    assert result.device == "cpu"


def test_result_reports_median_and_tail_latency():
    result = BenchmarkResult(
        name="r", steps=4, tokens=4, wall_s=1.0, step_times_s=[0.01, 0.02, 0.03, 0.40]
    )
    assert result.median_step_ms == 25.0
    assert result.p95_step_ms == 400.0
    assert result.tokens_per_s == 4.0


def test_empty_result_does_not_divide_by_zero():
    result = BenchmarkResult(name="r", steps=0, tokens=0, wall_s=0.0)
    assert result.tokens_per_s == 0.0
    assert result.median_step_ms == 0.0
    assert result.p95_step_ms == 0.0


def test_as_dict_is_json_serializable():
    result = BenchmarkResult(name="r", steps=1, tokens=8, wall_s=0.5, step_times_s=[0.5])
    payload = json.loads(json.dumps(result.as_dict()))
    assert payload["tokens_per_s"] == 16.0
    assert "step_times_s" not in payload


def test_compare_reports_slowdown_against_baseline():
    baseline = BenchmarkResult(
        name="baseline", steps=1, tokens=100, wall_s=1.0, peak_reserved_mb=1000.0
    )
    streamed = BenchmarkResult(
        name="streamed", steps=1, tokens=50, wall_s=1.0, peak_reserved_mb=400.0
    )
    report = compare([baseline, streamed], baseline="baseline")

    assert "2.00x slower" in report
    assert "+600.0 MB" in report


def test_cli_smoke_run(capsys):
    exit_code = main(
        ["--layers", "2", "--hidden", "32", "--batch", "1", "--seq", "4", "--steps", "2", "--json"]
    )
    payload = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert payload["name"] == "baseline:resident"
    assert payload["tokens"] == 8

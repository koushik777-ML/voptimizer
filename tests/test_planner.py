import torch
from conftest import BranchingStack, SharedBlockStack, Stack

from voptimizer import ExecutionPlanner
from voptimizer.planner import ExecutionPlan, PlanStep, module_param_bytes


def trace(model, hidden=8):
    return ExecutionPlanner().trace(model, torch.randn(2, hidden))


def test_trace_records_leaf_order():
    plan = trace(Stack(layers=2))

    assert plan.names == (
        "blocks.0.up",
        "blocks.0.down",
        "blocks.1.up",
        "blocks.1.down",
    )
    assert [step.index for step in plan] == [0, 1, 2, 3]
    assert plan[0].module_type == "Linear"


def test_parameterless_modules_are_excluded():
    """GELU has nothing to prefetch; including it only dilutes the window."""
    plan = trace(Stack(layers=1))
    assert "blocks.0.act" not in plan.names


def test_include_predicate_overrides_the_default_filter():
    planner = ExecutionPlanner(include=lambda name, module: True)
    plan = planner.trace(Stack(layers=1), torch.randn(2, 8))
    assert "blocks.0.act" in plan.names


def test_shared_module_occupies_multiple_steps():
    plan = trace(SharedBlockStack())

    assert plan.names.count("block.up") == 2
    assert len(plan) == 4
    # First invocation only, so a caller can resnap a drifted cursor.
    assert plan.index_of("block.up") == 0
    assert plan.index_of("missing") is None


def test_total_param_bytes_counts_shared_weights_once():
    model = SharedBlockStack(hidden=8)
    plan = trace(model)
    expected = module_param_bytes(model.block.up) + module_param_bytes(model.block.down)
    assert plan.total_param_bytes == expected


def test_tracing_restores_training_mode_and_leaves_no_hooks():
    model = Stack(layers=1)
    model.train()
    trace(model)

    assert model.training is True
    assert not model.blocks[0].up._forward_pre_hooks


def test_trace_follows_the_branch_actually_taken():
    model = BranchingStack()
    model.take_left = False
    plan = trace(model)
    assert plan.names == ("right",)


def test_successors_returns_the_next_distinct_modules():
    plan = trace(Stack(layers=3))
    assert [step.name for step in plan.successors(0, window=2)] == [
        "blocks.0.down",
        "blocks.1.up",
    ]
    assert plan.successors(len(plan) - 1, window=2) == ()
    assert plan.successors(0, window=0) == ()


def test_successors_skips_repeats_of_the_same_module():
    plan = trace(SharedBlockStack())
    assert [step.name for step in plan.successors(0, window=3)] == ["block.down", "block.up"]


def test_successors_stops_at_the_byte_budget():
    steps = [
        PlanStep(index=i, name=f"layer.{i}", module_type="Linear", param_bytes=100)
        for i in range(4)
    ]
    plan = ExecutionPlan.of(steps)

    assert len(plan.successors(0, window=3, budget_bytes=250)) == 2
    assert len(plan.successors(0, window=3, budget_bytes=1000)) == 3


def test_budget_always_yields_one_step():
    """An oversized layer must still be prefetched, or execution stalls."""
    steps = [
        PlanStep(index=0, name="a", module_type="Linear", param_bytes=1),
        PlanStep(index=1, name="huge", module_type="Linear", param_bytes=10**9),
    ]
    plan = ExecutionPlan.of(steps)
    assert [step.name for step in plan.successors(0, window=2, budget_bytes=10)] == ["huge"]


def test_plan_step_reports_size_in_gb():
    step = PlanStep(index=0, name="a", module_type="Linear", param_bytes=1024**3)
    assert step.param_gb == 1.0

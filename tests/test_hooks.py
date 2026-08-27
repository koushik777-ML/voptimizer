import pytest
import torch
from conftest import BranchingStack, SharedBlockStack, Stack

from voptimizer import ExecutionPlanner, LayerEvent, ModuleHookManager


def build(model, hidden=8, **manager_kwargs):
    inputs = torch.randn(2, hidden)
    plan = ExecutionPlanner().trace(model, inputs)
    starts: list[LayerEvent] = []
    ends: list[LayerEvent] = []
    manager = ModuleHookManager(
        plan, on_layer_start=starts.append, on_layer_end=ends.append, **manager_kwargs
    )
    return manager, model, inputs, starts, ends


def test_events_fire_in_execution_order():
    manager, model, inputs, starts, ends = build(Stack(layers=2))
    with manager.attach(model):
        model(inputs)

    assert [e.name for e in starts] == [
        "blocks.0.up",
        "blocks.0.down",
        "blocks.1.up",
        "blocks.1.down",
    ]
    assert [e.step_index for e in starts] == [0, 1, 2, 3]
    assert [e.name for e in ends] == [e.name for e in starts]
    assert manager.stats.forwards == 1
    assert manager.stats.steps == 4
    assert manager.stats.mispredictions == 0


def test_start_event_carries_the_prefetch_window():
    manager, model, inputs, starts, _ = build(Stack(layers=3), prefetch_window=2)
    with manager.attach(model):
        model(inputs)

    assert starts[0].prefetch_names == ("blocks.0.down", "blocks.1.up")
    assert starts[-1].prefetch_names == ()


def test_prefetch_window_respects_the_byte_budget():
    manager, model, inputs, starts, _ = build(
        Stack(layers=3), prefetch_window=3, prefetch_budget_bytes=1
    )
    with manager.attach(model):
        model(inputs)

    # Budget always yields one step so prefetch never stalls.
    assert len(starts[0].prefetch) == 1


def test_cursor_resets_on_every_forward():
    manager, model, inputs, starts, _ = build(Stack(layers=2))
    with manager.attach(model):
        model(inputs)
        model(inputs)

    assert manager.stats.forwards == 2
    assert manager.stats.mispredictions == 0
    assert [e.step_index for e in starts] == [0, 1, 2, 3, 0, 1, 2, 3]


def test_divergent_branch_is_reported_as_unplanned():
    """A branch the trace never took must still be visible to the policy."""
    model = BranchingStack()
    manager, model, inputs, starts, _ = build(model)
    model.take_left = False

    with manager.attach(model):
        model(inputs)

    assert [e.name for e in starts] == ["right"]
    assert starts[0].predicted is False
    assert starts[0].step_index == -1
    assert starts[0].prefetch == ()
    assert manager.stats.unplanned == 1


def test_cursor_resnaps_after_an_out_of_order_step():
    model = Stack(layers=2)
    inputs = torch.randn(2, 8)
    plan = ExecutionPlanner().trace(model, inputs)

    starts: list[LayerEvent] = []
    manager = ModuleHookManager(plan, on_layer_start=starts.append)
    with manager.attach(model):
        # Skipping ahead desyncs the cursor; the next layer must resnap.
        model.blocks[1].up(inputs)
        assert manager.cursor == 3
        model.blocks[0].down(inputs)

    assert [e.step_index for e in starts] == [2, 1]
    assert manager.cursor == 2
    assert manager.stats.mispredictions == 2
    assert manager.stats.prediction_accuracy == 0.0


def test_shared_module_advances_through_both_steps():
    manager, model, inputs, starts, _ = build(SharedBlockStack())
    with manager.attach(model):
        model(inputs)

    assert [e.step_index for e in starts] == [0, 1, 2, 3]
    assert manager.stats.mispredictions == 0


def test_paused_suppresses_events():
    """Checkpoint recompute replays a forward; those are not new steps."""
    manager, model, inputs, starts, ends = build(Stack(layers=2))
    with manager.attach(model):
        with manager.paused():
            model(inputs)
        assert starts == []
        assert manager.stats.forwards == 0

        model(inputs)

    assert len(starts) == 4
    assert len(ends) == 4


def test_detach_removes_every_hook():
    manager, model, inputs, starts, _ = build(Stack(layers=1))
    manager.attach(model)
    assert manager.attached is True

    manager.detach()
    assert manager.attached is False
    model(inputs)

    assert starts == []
    assert not model.blocks[0].up._forward_pre_hooks
    assert not model.blocks[0].up._forward_hooks
    assert not model._forward_pre_hooks

    manager.detach()  # idempotent


def test_double_attach_is_rejected():
    manager, model, _, _, _ = build(Stack(layers=1))
    with manager.attach(model):
        with pytest.raises(RuntimeError, match="already attached"):
            manager.attach(model)


def test_hooks_do_not_change_model_output():
    model = Stack(layers=2)
    inputs = torch.randn(2, 8)
    with torch.no_grad():
        expected = model(inputs)

    manager, model, _, _, _ = build(model)
    with manager.attach(model), torch.no_grad():
        actual = model(inputs)

    assert torch.equal(expected, actual)


def test_manager_works_without_callbacks():
    plan = ExecutionPlanner().trace(Stack(layers=1), torch.randn(2, 8))
    model = Stack(layers=1)
    with ModuleHookManager(plan).attach(model):
        model(torch.randn(2, 8))

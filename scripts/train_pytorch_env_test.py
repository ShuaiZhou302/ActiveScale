from __future__ import annotations

import pytest
import torch

from scripts import train_pytorch


@pytest.mark.parametrize(
    ("value", "expected"),
    [("auto", None), ("1", True), ("true", True), ("0", False), ("off", False)],
)
def test_gradient_clip_foreach(monkeypatch: pytest.MonkeyPatch, value: str, expected: bool | None) -> None:
    monkeypatch.setenv("PI05_GRAD_CLIP_FOREACH", value)
    assert train_pytorch._gradient_clip_foreach() is expected  # noqa: SLF001


def test_gradient_clip_foreach_rejects_invalid_value(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PI05_GRAD_CLIP_FOREACH", "sometimes")
    with pytest.raises(ValueError, match="must be auto, 0, or 1"):
        train_pytorch._gradient_clip_foreach()  # noqa: SLF001


@pytest.mark.parametrize(
    ("value", "expected"),
    [("auto", None), ("1", True), ("true", True), ("0", False), ("off", False)],
)
def test_adamw_foreach(monkeypatch: pytest.MonkeyPatch, value: str, expected: bool | None) -> None:
    monkeypatch.setenv("PI05_ADAMW_FOREACH", value)
    assert train_pytorch._adamw_foreach() is expected  # noqa: SLF001


def test_adamw_foreach_rejects_invalid_value(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PI05_ADAMW_FOREACH", "sometimes")
    with pytest.raises(ValueError, match="must be auto, 0, or 1"):
        train_pytorch._adamw_foreach()  # noqa: SLF001


def test_largest_gradient_tensors_reports_dominant_parameter() -> None:
    model = torch.nn.Sequential(torch.nn.Linear(3, 2), torch.nn.Linear(2, 1))
    for parameter in model.parameters():
        parameter.grad = torch.ones_like(parameter)
    model[1].weight.grad.fill_(100.0)

    rows = train_pytorch._largest_gradient_tensors(model, limit=2)  # noqa: SLF001

    assert rows[0][0] == "1.weight"
    assert rows[0][1] > rows[1][1]


def test_batch_numeric_ranges_reports_nonfinite_and_skips_images() -> None:
    observation = type("Observation", (), {})()
    observation.state = torch.tensor([[1.0, float("nan")]])
    observation.flow_actions = torch.tensor([[3.0, -8.0]])
    observation.images = {"front": torch.ones(1, 3, 4, 4)}

    rows = train_pytorch._batch_numeric_ranges(observation, torch.tensor([[2.0]]))  # noqa: SLF001
    by_name = {name: (max_abs, nonfinite) for name, max_abs, nonfinite in rows}

    assert by_name["observation.state"] == (1.0, 1)
    assert by_name["observation.flow_actions"] == (8.0, 0)
    assert all("images" not in name for name in by_name)

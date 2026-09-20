"""Training semantics, isolation and restart safety for fight fusion branches."""
import argparse
import copy
import json
import random

import numpy as np
import pytest
import torch
from torch import nn

from scripts import train_fight_fusion as training


def identity_forward(model):
    return lambda window: None if window.get("unknown") else model(window["x"]).squeeze()


def test_video_mil_uses_only_maximum_and_matches_full_graph():
    model = nn.Linear(1, 1, bias=False)
    model.weight.data.fill_(.4)
    reference = copy.deepcopy(model)
    windows = [{"x": torch.tensor([value])} for value in [1., 3., 2.]]
    criterion = nn.BCEWithLogitsLoss()
    loss = training.backward_row(model, windows, {"label": 1, "label_kind": "video"},
                                 identity_forward(model), criterion, divisor=1)
    full = criterion(torch.stack([reference(w["x"]).squeeze() for w in windows]).amax(), torch.tensor(1.))
    full.backward()
    assert loss == pytest.approx(float(full.detach()))
    torch.testing.assert_close(model.weight.grad, reference.weight.grad)


def test_tied_mil_maxima_share_gradient():
    model = nn.Linear(2, 1, bias=False)
    model.weight.data.fill_(.5)
    reference = copy.deepcopy(model)
    windows = [{"x": torch.tensor([1., 0.])}, {"x": torch.tensor([0., 1.])}]
    criterion = nn.BCEWithLogitsLoss()
    training.backward_row(model, windows, {"label": 1}, identity_forward(model), criterion, divisor=1)
    criterion(torch.stack([reference(w["x"]).squeeze() for w in windows]).amax(), torch.tensor(1.)).backward()
    torch.testing.assert_close(model.weight.grad, reference.weight.grad)


def test_frame_labels_mean_loss_per_video_and_skip_unknown():
    model = nn.Linear(1, 1, bias=False)
    model.weight.data.fill_(.25)
    reference = copy.deepcopy(model)
    windows = [{"x": torch.tensor([1.]), "label": 0},
               {"x": torch.tensor([2.]), "label": 1},
               {"x": torch.tensor([999.]), "label": 1, "unknown": True},
               {"x": torch.tensor([999.]), "label": None}]
    criterion = nn.BCEWithLogitsLoss()
    loss = training.backward_row(model, windows, {"label": 1, "label_kind": "frame"},
                                 identity_forward(model), criterion, divisor=1)
    expected = torch.stack([criterion(reference(w["x"]).squeeze(), torch.tensor(float(w["label"])))
                            for w in windows[:2]]).mean()
    expected.backward()
    assert loss == pytest.approx(float(expected.detach()))
    torch.testing.assert_close(model.weight.grad, reference.weight.grad)


def test_unknown_row_never_creates_a_negative_gradient():
    model = nn.Linear(1, 1)
    windows = [{"x": torch.tensor([1.]), "unknown": True}]
    result = training.backward_row(model, windows, {"label": 1}, identity_forward(model), nn.BCEWithLogitsLoss())
    assert result is None
    assert all(parameter.grad is None for parameter in model.parameters())


def test_mil_dropout_and_bn_replay_matches_full_graph():
    training.seed_everything(42)
    model = nn.Sequential(nn.BatchNorm1d(2), nn.Dropout(.3), nn.Linear(2, 1))
    reference = copy.deepcopy(model)
    windows = [{"x": torch.randn(4, 2)} for _ in range(3)]
    initial = training.rng_state()
    criterion = nn.BCEWithLogitsLoss()
    training.backward_row(model, windows, {"label": 1}, lambda w: model(w["x"]).mean(), criterion, divisor=1)
    actual_rng = torch.get_rng_state().clone()
    training.restore_rng(initial)
    logits = torch.stack([reference(w["x"]).mean() for w in windows])
    criterion(logits.amax(), torch.tensor(1.)).backward()
    for actual, expected in zip(model.parameters(), reference.parameters()):
        torch.testing.assert_close(actual.grad, expected.grad)
    torch.testing.assert_close(model[0].running_mean, reference[0].running_mean)
    torch.testing.assert_close(model[0].running_var, reference[0].running_var)
    assert torch.equal(torch.get_rng_state(), actual_rng)


class TinyRGB(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = nn.Linear(2, 2)
        self.layer4 = nn.Sequential(nn.Linear(2, 2), nn.BatchNorm1d(2))
        self.avgpool = nn.Identity()
        self.fc = nn.Linear(2, 2)


def test_roi_logit_aggregates_candidates_after_classifier():
    model = TinyRGB()
    model.fc.weight.data.copy_(torch.tensor([[0., 0.], [1., 0.]]))
    model.fc.bias.data.zero_()
    window = {"roi_pooled": torch.tensor([[1., 0.], [3., 0.], [2., 0.]])}
    actual = training.window_logit(model, window, "roi", "head", torch.device("cpu"))
    assert float(actual) == 3.
    assert training.window_logit(model, {"roi_pooled": torch.empty(0, 2)}, "roi", "head", torch.device("cpu")) is None


def test_rgb_head_then_layer4_leaves_early_layers_and_bn_frozen():
    model = TinyRGB()
    optimizer = training.configure_optimizer(model, "roi", "head")
    assert all(p.requires_grad for p in model.fc.parameters())
    assert not any(p.requires_grad for p in model.layer4.parameters())
    assert optimizer.param_groups[0]["lr"] == .001
    optimizer = training.configure_optimizer(model, "roi", "finetune")
    training.training_mode(model, "roi", "finetune")
    assert not any(p.requires_grad for p in model.stem.parameters())
    assert all(p.requires_grad for p in model.layer4.parameters())
    assert not model.layer4[1].training
    assert optimizer.param_groups[0]["lr"] == .0001


def test_ntu_warmup_and_lower_backbone_learning_rate():
    model = nn.Module()
    model.backbone = nn.Sequential(nn.Linear(2, 2), nn.BatchNorm1d(2))
    model.pair_head = nn.Linear(2, 1)
    training.configure_optimizer(model, "skeleton_ntu", "head")
    training.training_mode(model, "skeleton_ntu", "head")
    assert not model.backbone.training
    assert not any(p.requires_grad for p in model.backbone.parameters())
    optimizer = training.configure_optimizer(model, "skeleton_ntu", "finetune")
    assert [group["lr"] for group in optimizer.param_groups] == [.0001, .00001]
    assert all(p.requires_grad for p in model.parameters())


def manifest_rows():
    return [dict(sample_id=f"{split}-{label}-{i}", dataset="test", group=f"{split}-{label}-{i}",
                 sha256=f"{split}-{label}-{i}", split=split, label=label, label_kind="video")
            for split in ("train", "branch_val", "test", "fusion_fit", "calibration", "legacy_test")
            for label in (0, 1) for i in range(2)]


def test_only_training_and_branch_validation_are_observed():
    rows = manifest_rows()
    train, validation = training.training_rows({"rows": rows})
    assert {r["split"] for r in train} == {"train"}
    assert {r["split"] for r in validation} == {"branch_val"}
    rows[4]["group"] = rows[0]["group"]
    with pytest.raises(ValueError, match="group leakage"):
        training.training_rows({"rows": rows})


def test_checkpoint_retains_rng_optimizer_order_and_cursor(tmp_path, monkeypatch):
    monkeypatch.setattr(training, "check_disk", lambda path: None)
    training.seed_everything(43)
    model = nn.Linear(2, 1)
    optimizer = torch.optim.AdamW(model.parameters())
    scaler = torch.amp.GradScaler("cpu", enabled=False)
    model(torch.ones(2)).sum().backward()
    optimizer.step()
    state = dict(order=[2, 0, 1], cursor=2, epoch=3)
    path = tmp_path / "resume.pt"
    training.checkpoint(path, model, optimizer, scaler, state, "test-signature")
    expected = (random.random(), float(np.random.rand()), torch.rand(2))
    saved = torch.load(path, weights_only=True)
    training.restore_rng(saved["rng"])
    actual = (random.random(), float(np.random.rand()), torch.rand(2))
    assert saved["runner_state"] == state
    assert saved["optimizer"]["state"]
    assert saved["config_signature"] == "test-signature"
    assert actual[:2] == expected[:2]
    assert torch.equal(actual[2], expected[2])
    assert not path.with_suffix(".pt.tmp").exists()


@pytest.mark.parametrize("kind", ["json", "torch"])
def test_atomic_write_retries_only_transient_permission_errors(tmp_path, monkeypatch, kind):
    path = tmp_path / ("result.json" if kind == "json" else "result.pt")
    original = training.Path.replace
    calls, delays = [], []

    def temporary_busy(source, target):
        calls.append(str(target))
        if len(calls) < 3:
            raise PermissionError("Windows reader briefly holds destination")
        return original(source, target)

    monkeypatch.setattr(training.Path, "replace", temporary_busy)
    monkeypatch.setattr(training.time, "sleep", delays.append)
    writer = training.atomic_json if kind == "json" else training.atomic_torch
    writer(path, {"status": "complete"})
    assert len(calls) == 3 and len(delays) == 2
    value = json.loads(path.read_text()) if kind == "json" else torch.load(path, weights_only=True)
    assert value == {"status": "complete"}


def test_atomic_replace_permission_retry_is_bounded_and_other_errors_escape(tmp_path, monkeypatch):
    tick, calls = [0.], [0]

    def clock():
        tick[0] += 1.
        return tick[0]

    def denied(source, target):
        calls[0] += 1
        raise PermissionError("persistent")

    monkeypatch.setattr(training.time, "monotonic", clock)
    monkeypatch.setattr(training.time, "sleep", lambda value: None)
    monkeypatch.setattr(training.Path, "replace", denied)
    with pytest.raises(PermissionError, match="persistent"):
        training.replace_with_retry(tmp_path / "tmp", tmp_path / "dest")
    assert calls[0] == 3

    def other_failure(source, target):
        raise OSError("disk failure")

    monkeypatch.setattr(training.Path, "replace", other_failure)
    with pytest.raises(OSError, match="disk failure"):
        training.replace_with_retry(tmp_path / "tmp", tmp_path / "dest")


def test_lazy_feature_cache_miss_does_not_consume_training_rng():
    class Lazy:
        def __len__(self):
            return 1

        def __getitem__(self, index):
            random.random()
            np.random.rand()
            torch.rand(3)
            return {"index": index}

    training.seed_everything(44)
    before = training.rng_state()
    assert list(training.IsolatedWindows(Lazy())) == [{"index": 0}]
    actual = (random.random(), float(np.random.rand()), torch.rand(3))
    training.restore_rng(before)
    expected = (random.random(), float(np.random.rand()), torch.rand(3))
    assert actual[:2] == expected[:2]
    assert torch.equal(actual[2], expected[2])


def test_validation_unknowns_remain_in_recall_denominator():
    _, metrics = training.choose_threshold([0, 0, 1, 1], [.1, .2, .9, None], limit=.05)
    assert metrics["recall"][1] == .5
    assert metrics["samples"] == 4
    assert metrics["coverage"] == .75


def test_validation_sigmoid_preserves_float32_precision():
    model = TinyRGB()
    model.fc.weight.data.zero_()
    model.fc.bias.data.copy_(torch.tensor([0., 12.]))
    row = dict(sample_id="source", dataset="test", label=1)

    class Store:
        def get(self, row, full):
            assert full
            return [dict(start=0., end=4., global_pooled=torch.ones(2))]

    scores, _ = training.evaluate(model, [row], Store(), "global", "head", torch.device("cpu"))
    assert .999 < scores[0] < 1.


def test_production_budgets_and_smoke_are_separate():
    assert training.stages_for("global") == [("head", 10), ("finetune", 40)]
    assert training.stages_for("skeleton_random") == [("finetune", 50)]
    assert training.stages_for("skeleton_ntu") == [("head", 10), ("finetune", 40)]
    assert training.stages_for("global", smoke=True, max_epochs=2) == [("head", 1), ("finetune", 1)]
    args = training.parser().parse_args(["--branch", "global", "--seed", "42", "--max-epochs", "1"])
    with pytest.raises(ValueError, match="smoke-only"):
        training.run(args)


def test_production_requires_explicit_source_seal_and_exact_manifest_hash(tmp_path):
    manifest = dict(status="sealed", training_allowed=True, rows=[])
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="seal is missing"):
        training.validate_manifest_seal(manifest, path)
    seal_path = path.with_suffix(".seal.json")
    seal_path.write_text(json.dumps(dict(manifest_sha256=training.file_sha(path))), encoding="utf-8")
    training.validate_manifest_seal(manifest, path)
    with pytest.raises(ValueError, match="training-authorized"):
        training.validate_manifest_seal(dict(manifest, status="complete"), path)
    with pytest.raises(ValueError, match="training-authorized"):
        training.validate_manifest_seal(dict(manifest, training_allowed=False), path)
    path.write_text(json.dumps(dict(manifest, rows=["modified"])), encoding="utf-8")
    with pytest.raises(ValueError, match="hash mismatch"):
        training.validate_manifest_seal(manifest, path)
    training.validate_manifest_seal(dict(status="partial"), path, smoke=True)


def test_tiny_training_writes_selection_without_reading_other_splits(tmp_path, monkeypatch):
    from backend.vision import fight_fusion

    monkeypatch.setattr(fight_fusion, "make_rgb_model", TinyRGB)
    monkeypatch.setattr(training, "check_disk", lambda path: None)
    original_sha = training.file_sha
    monkeypatch.setattr(training, "file_sha", lambda path: original_sha(path) if training.Path(path).exists() else "test-only-missing-code")
    rows = manifest_rows()
    manifest = {"rows": rows}
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    class Store:
        signature = "test-feature-signature"

        def __init__(self):
            self.accessed = []

        def get(self, row, epoch=None, full=False):
            assert row["split"] in ("train", "branch_val")
            self.accessed.append(row["split"])
            return [dict(start=0., end=4., global_pooled=torch.tensor([float(row["label"]), 1.]))]

    store = Store()
    destination = tmp_path / "smoke/global/seed42"
    destination.mkdir(parents=True)
    args = argparse.Namespace(branch="global", seed=42, smoke=True, max_epochs=1, device="cpu")
    training._train_with_store(args, manifest, manifest_path, destination, store)
    selection = json.loads((destination / "selection.json").read_text(encoding="utf-8"))
    assert selection["status"] == "complete"
    assert selection["smoke"] is True
    assert selection["test_accessed"] is False
    assert selection["epochs_trained"] == 1
    assert set(store.accessed) == {"train", "branch_val"}
    assert (destination / "resume.pt").is_file()
    assert selection["sha256"] == original_sha(destination / "selected_best.pt")
    checkpoint = torch.load(destination / "selected_best.pt", weights_only=True)
    assert checkpoint["config"]["data_access"] == ["train", "branch_val"]
    before = len(store.accessed)
    training._train_with_store(args, manifest, manifest_path, destination, store)
    assert len(store.accessed) == before


def test_resume_mid_epoch_reproduces_uninterrupted_selected_weights(tmp_path, monkeypatch):
    from backend.vision import fight_fusion

    monkeypatch.setattr(fight_fusion, "make_rgb_model", TinyRGB)
    monkeypatch.setattr(training, "check_disk", lambda path: None)
    monkeypatch.setattr(training, "smoke_subset", lambda rows: rows)
    original_sha = training.file_sha
    monkeypatch.setattr(training, "file_sha", lambda path: original_sha(path) if training.Path(path).exists() else "test-code")
    rows = [dict(sample_id=f"{split}-{label}-{i}", dataset="test", group=f"{split}-{label}-{i}",
                 sha256=f"{split}-{label}-{i}", split=split, label=label, label_kind="video")
            for split in ("train", "branch_val") for label in (0, 1) for i in range(4)]
    manifest = {"rows": rows}
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    class Store:
        signature = "resume-test"

        def get(self, row, epoch=None, full=False):
            return [dict(start=0., end=4., global_pooled=torch.tensor([float(row["label"]), 1.]))]

    args = argparse.Namespace(branch="global", seed=42, smoke=True, max_epochs=1, device="cpu")
    uninterrupted = tmp_path / "uninterrupted"
    interrupted = tmp_path / "interrupted"
    uninterrupted.mkdir()
    interrupted.mkdir()
    training._train_with_store(args, manifest, manifest_path, uninterrupted, Store())
    original_checkpoint = training.checkpoint
    calls = [0]

    def interrupted_checkpoint(path, model, optimizer, scaler, state, config_signature):
        original_checkpoint(path, model, optimizer, scaler, state, config_signature)
        if state["cursor"] == 4 and state["phase"] == "train":
            calls[0] += 1
            raise RuntimeError("simulated interruption")

    tick = [0.]

    def monotonic():
        tick[0] += 61.
        return tick[0]

    monkeypatch.setattr(training.time, "monotonic", monotonic)
    monkeypatch.setattr(training, "checkpoint", interrupted_checkpoint)
    with pytest.raises(RuntimeError, match="simulated interruption"):
        training._train_with_store(args, manifest, manifest_path, interrupted, Store())
    assert calls[0] == 1
    resume = torch.load(interrupted / "resume.pt", weights_only=True)
    assert resume["runner_state"]["cursor"] == 4
    assert resume["runner_state"]["optimizer_steps"] == 1
    monkeypatch.setattr(training, "checkpoint", original_checkpoint)
    training._train_with_store(args, manifest, manifest_path, interrupted, Store())
    first = torch.load(uninterrupted / "selected_best.pt", weights_only=True)
    second = torch.load(interrupted / "selected_best.pt", weights_only=True)
    assert first["threshold"] == second["threshold"]
    for name in first["state_dict"]:
        torch.testing.assert_close(first["state_dict"][name], second["state_dict"][name], rtol=0, atol=0)

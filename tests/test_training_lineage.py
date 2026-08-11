import pytest
import torch

import repro.train_sudoku as train_sudoku
from repro.train_sudoku import (
    MixedSudokuTrainingDataset,
    build_model,
    restore_optimizer_state,
    sha256_json,
    verify_manifest_seal,
)


def sealed(value):
    return {**value, "seal": {"sha256": sha256_json(value)}}


def test_manifest_seal_accepts_untouched_content():
    manifest = sealed({"schema": "test", "training": {"target_steps": 50000}})
    assert verify_manifest_seal(manifest) == manifest["seal"]["sha256"]


def test_manifest_seal_rejects_parent_tampering():
    manifest = sealed({"schema": "test", "training": {"target_steps": 50000}})
    manifest["training"]["target_steps"] = 1300000
    with pytest.raises(RuntimeError, match="manifest seal mismatch"):
        verify_manifest_seal(manifest)


def test_paper_table_10_final_convolution_is_explicit_arm():
    released = build_model(729, 729, final_conv_kernel=1)
    assert released.model.ebm.conv5.kernel_size == (1, 1)
    paper = build_model(729, 729, final_conv_kernel=3)
    assert paper.model.ebm.conv5.kernel_size == (3, 3)
    assert paper.model.ebm.conv5.padding == (1, 1)


def test_unknown_final_convolution_is_rejected():
    with pytest.raises(ValueError, match="must be 1 or 3"):
        build_model(729, 729, final_conv_kernel=5)


def test_sudoku_search_state_negative_refinement_is_explicit_and_opt_in():
    released = build_model(729, 729)
    assert released.sudoku_negative_opt_steps == 0
    refined = build_model(729, 729, sudoku_negative_opt_steps=2)
    assert refined.sudoku_negative_opt_steps == 2
    with pytest.raises(ValueError, match="non-negative integer"):
        build_model(729, 729, sudoku_negative_opt_steps=-1)


class _TinyDataset:
    def __init__(self, count):
        self.features = torch.zeros(count, 9, 9, 9)
        self.labels = torch.zeros(count, 9, 9, 9)

    def __len__(self):
        return len(self.features)


def test_mixed_dataset_exposes_declared_rrn_fraction(monkeypatch):
    monkeypatch.setattr(
        train_sudoku, "SudokuDataset", lambda *_args, **_kwargs: _TinyDataset(9)
    )
    monkeypatch.setattr(
        train_sudoku, "SudokuRRNDataset",
        lambda *_args, **kwargs: _TinyDataset(kwargs["limit"]),
    )
    dataset = MixedSudokuTrainingDataset(rrn_count=3)
    assert dataset.standard_count == 9
    assert dataset.rrn_count == 3
    assert len(dataset) == 12
    with pytest.raises(ValueError, match="mixed RRN count"):
        MixedSudokuTrainingDataset(rrn_count=10)


def test_optimizer_reset_is_only_applied_at_new_lineage_boundary():
    class RecordingOptimizer:
        def __init__(self):
            self.loaded = None

        def load_state_dict(self, state):
            self.loaded = state

    payload = {"optimizer": {"state": "parent"}}
    optimizer = RecordingOptimizer()
    restored = restore_optimizer_state(
        optimizer, payload, new_lineage=True,
        reset_optimizer_on_lineage_start=True,
    )
    assert not restored
    assert optimizer.loaded is None

    restored = restore_optimizer_state(
        optimizer, payload, new_lineage=False,
        reset_optimizer_on_lineage_start=True,
    )
    assert restored
    assert optimizer.loaded == payload["optimizer"]

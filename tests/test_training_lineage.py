import pytest

from repro.train_sudoku import build_model, sha256_json, verify_manifest_seal


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

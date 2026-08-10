import pytest

from repro.train_sudoku import sha256_json, verify_manifest_seal


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

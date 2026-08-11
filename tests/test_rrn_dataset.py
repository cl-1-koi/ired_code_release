from sat_dataset import load_rrn_dataset


def test_rrn_loader_limit_is_applied_before_tensor_materialization(tmp_path):
    puzzle = "1" + "0" * 80
    solution = "123456789" * 9
    (tmp_path / "train.csv").write_text(
        "\n".join(f"{puzzle},{solution}" for _ in range(3)) + "\n"
    )

    features, labels = load_rrn_dataset(tmp_path, "train", limit=2)

    assert features.shape == (2, 9, 9, 9)
    assert labels.shape == (2, 9, 9, 9)

import torch
import torch.nn.functional as F

from repro.sudoku_metrics import sudoku_batch_metrics


SOLUTION = torch.tensor(
    [
        [1, 2, 3, 4, 5, 6, 7, 8, 9],
        [4, 5, 6, 7, 8, 9, 1, 2, 3],
        [7, 8, 9, 1, 2, 3, 4, 5, 6],
        [2, 3, 4, 5, 6, 7, 8, 9, 1],
        [5, 6, 7, 8, 9, 1, 2, 3, 4],
        [8, 9, 1, 2, 3, 4, 5, 6, 7],
        [3, 4, 5, 6, 7, 8, 9, 1, 2],
        [6, 7, 8, 9, 1, 2, 3, 4, 5],
        [9, 1, 2, 3, 4, 5, 6, 7, 8],
    ],
    dtype=torch.long,
) - 1


def encoded(digits):
    return (F.one_hot(digits, num_classes=9).float() - 0.5) * 2


def test_exact_valid_solution_metrics():
    label = encoded(SOLUTION)[None]
    mask = torch.zeros_like(label)
    mask[:, 0, :, :] = 1
    result = sudoku_batch_metrics(label.reshape(1, -1), label.reshape(1, -1), mask.reshape(1, -1))
    assert result.aggregate() == {
        "n": 1,
        "strict_valid_rate": 1.0,
        "clue_consistency_rate": 1.0,
        "exact_solution_rate": 1.0,
        "cell_accuracy": 1.0,
        "unknown_cell_accuracy": 1.0,
        "satnet_constraint_fraction": 1.0,
    }


def test_invalid_prediction_is_not_counted_as_a_solution():
    label = encoded(SOLUTION)[None]
    prediction_digits = SOLUTION.clone()
    prediction_digits[4, 4] = prediction_digits[4, 3]
    prediction = encoded(prediction_digits)[None]
    mask = torch.zeros_like(label)
    mask[:, 0, :, :] = 1
    result = sudoku_batch_metrics(prediction, label, mask)
    assert not bool(result.strict_valid[0])
    assert not bool(result.exact_solution[0])
    assert bool(result.clue_consistent[0])
    assert float(result.unknown_cell_accuracy[0]) < 1.0

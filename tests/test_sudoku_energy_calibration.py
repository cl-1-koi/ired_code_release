import torch
import torch.nn.functional as F

from repro.sudoku_energy_calibration import (
    corrupt_solutions,
    exact_conflict_energy,
)
from repro.sudoku_exact_annealing import initialize_boxes, row_column_energy
from repro.sudoku_search_calibration import spearman
from repro.sudoku_metrics import decode_digits


SOLUTION = torch.tensor([
    [0, 1, 2, 3, 4, 5, 6, 7, 8],
    [3, 4, 5, 6, 7, 8, 0, 1, 2],
    [6, 7, 8, 0, 1, 2, 3, 4, 5],
    [1, 2, 3, 4, 5, 6, 7, 8, 0],
    [4, 5, 6, 7, 8, 0, 1, 2, 3],
    [7, 8, 0, 1, 2, 3, 4, 5, 6],
    [2, 3, 4, 5, 6, 7, 8, 0, 1],
    [5, 6, 7, 8, 0, 1, 2, 3, 4],
    [8, 0, 1, 2, 3, 4, 5, 6, 7],
])


def encoded_solution(batch: int = 1) -> torch.Tensor:
    values = F.one_hot(SOLUTION, num_classes=9).float()
    return ((values - 0.5) * 2).reshape(1, 729).repeat(batch, 1)


def test_exact_conflict_energy_is_zero_only_for_valid_fixture():
    valid = SOLUTION.unsqueeze(0)
    invalid = valid.clone()
    invalid[0, 0, 0] = invalid[0, 0, 1]
    energies = exact_conflict_energy(torch.cat((valid, invalid), dim=0))
    assert energies.tolist()[0] == 0
    assert energies.tolist()[1] > 0


def test_corruptions_have_exact_distance_and_preserve_clues():
    labels = encoded_solution(batch=2)
    masks = torch.zeros_like(labels)
    masks[:, : 9 * 5] = 1
    corrupt = corrupt_solutions(labels, masks, 7, [10, 11], seed=123)
    before = decode_digits(labels)
    after = decode_digits(corrupt)
    changed = before != after
    assert changed.sum(dim=(1, 2)).tolist() == [7, 7]
    clue_mask = masks.reshape(2, 9, 9, 9)[..., 0].bool()
    assert not (changed & clue_mask).any()


def test_corruption_is_deterministic_by_dataset_index():
    labels = encoded_solution(batch=1)
    masks = torch.zeros_like(labels)
    left = corrupt_solutions(labels, masks, 8, [42], seed=7)
    right = corrupt_solutions(labels, masks, 8, [42], seed=7)
    other = corrupt_solutions(labels, masks, 8, [43], seed=7)
    assert torch.equal(left, right)
    assert not torch.equal(left, other)


def test_exact_annealer_initialization_preserves_clues_and_boxes():
    import random

    clues = torch.zeros((9, 9), dtype=torch.bool)
    clues[:, 0] = True
    puzzle = SOLUTION.clone()
    grid, mutable = initialize_boxes(puzzle, clues, random.Random(3))
    assert torch.equal(grid[clues], puzzle[clues])
    box_energy = exact_conflict_energy(grid.unsqueeze(0)) - row_column_energy(grid)
    assert int(box_energy[0]) == 0
    assert mutable


def test_restart_rank_correlation_is_within_vector_and_handles_ties():
    assert spearman(torch.tensor([1.0, 2.0, 3.0]), torch.tensor([4.0, 5.0, 6.0])) == 1.0
    assert spearman(torch.tensor([1.0, 2.0, 3.0]), torch.tensor([6.0, 5.0, 4.0])) == -1.0
    assert spearman(torch.tensor([1.0, 1.0]), torch.tensor([2.0, 3.0])) is None

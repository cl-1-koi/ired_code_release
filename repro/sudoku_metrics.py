"""Per-board Sudoku metrics that do not collapse a batch to its last sample."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


def decode_digits(values: torch.Tensor) -> torch.Tensor:
    """Decode flattened or spatial nine-channel values to digits 0..8."""
    if values.ndim == 2 and values.shape[-1] == 729:
        values = values.reshape(-1, 9, 9, 9)
    if values.ndim != 4 or values.shape[1:] != (9, 9, 9):
        raise ValueError(f"expected (B,729) or (B,9,9,9), got {tuple(values.shape)}")
    return values.argmax(dim=-1)


def clue_cells(mask: torch.Tensor) -> torch.Tensor:
    """Return the Boolean 9x9 clue mask from the released expanded mask."""
    if mask.ndim == 2 and mask.shape[-1] == 729:
        mask = mask.reshape(-1, 9, 9, 9)
    if mask.ndim != 4 or mask.shape[1:] != (9, 9, 9):
        raise ValueError(f"expected (B,729) or (B,9,9,9), got {tuple(mask.shape)}")
    return mask[..., 0].bool()


def strict_validity(digits: torch.Tensor) -> torch.Tensor:
    """Return one exact row/column/box validity Boolean per board."""
    if digits.ndim != 3 or digits.shape[1:] != (9, 9):
        raise ValueError(f"expected (B,9,9), got {tuple(digits.shape)}")
    onehot = F.one_hot(digits.long(), num_classes=9)
    rows = (onehot.sum(dim=2) == 1).all(dim=(1, 2))
    cols = (onehot.sum(dim=1) == 1).all(dim=(1, 2))
    boxes = onehot.reshape(-1, 3, 3, 3, 3, 9).sum(dim=(2, 4))
    boxes = (boxes == 1).all(dim=(1, 2, 3))
    return rows & cols & boxes


def satnet_constraint_fraction(digits: torch.Tensor) -> torch.Tensor:
    """Reproduce the released ``sudoku_score`` as a per-board fraction.

    This checks only whether the digit sums of the intersecting row, column,
    and box equal 36. It is reported beside, never in place of, strict validity.
    """
    row_sum_ok = digits.sum(dim=2, keepdim=True) == 36
    col_sum_ok = digits.sum(dim=1, keepdim=True) == 36
    box_sum_ok = (
        digits.reshape(-1, 3, 3, 3, 3).sum(dim=(2, 4), keepdim=True) == 36
    ).expand(-1, -1, 3, -1, 3).reshape(-1, 9, 9)
    return (row_sum_ok & col_sum_ok & box_sum_ok).float().mean(dim=(1, 2))


@dataclass(frozen=True)
class SudokuBatchMetrics:
    predicted_digits: torch.Tensor
    label_digits: torch.Tensor
    strict_valid: torch.Tensor
    clue_consistent: torch.Tensor
    exact_solution: torch.Tensor
    cell_accuracy: torch.Tensor
    unknown_cell_accuracy: torch.Tensor
    satnet_constraint_fraction: torch.Tensor

    def aggregate(self) -> dict[str, float | int]:
        return {
            "n": int(self.predicted_digits.shape[0]),
            "strict_valid_rate": float(self.strict_valid.float().mean()),
            "clue_consistency_rate": float(self.clue_consistent.float().mean()),
            "exact_solution_rate": float(self.exact_solution.float().mean()),
            "cell_accuracy": float(self.cell_accuracy.mean()),
            "unknown_cell_accuracy": float(self.unknown_cell_accuracy.mean()),
            "satnet_constraint_fraction": float(self.satnet_constraint_fraction.mean()),
        }


def sudoku_batch_metrics(
    prediction: torch.Tensor, label: torch.Tensor, mask: torch.Tensor
) -> SudokuBatchMetrics:
    predicted = decode_digits(prediction)
    target = decode_digits(label)
    clues = clue_cells(mask)
    if predicted.shape != target.shape or predicted.shape != clues.shape:
        raise ValueError("prediction, label, and clue mask batches do not align")
    correct = predicted == target
    unknown = ~clues
    clue_denominator = clues.sum(dim=(1, 2)).clamp_min(1)
    unknown_denominator = unknown.sum(dim=(1, 2)).clamp_min(1)
    return SudokuBatchMetrics(
        predicted_digits=predicted,
        label_digits=target,
        strict_valid=strict_validity(predicted),
        clue_consistent=((correct & clues).sum(dim=(1, 2)) == clue_denominator),
        exact_solution=correct.all(dim=(1, 2)),
        cell_accuracy=correct.float().mean(dim=(1, 2)),
        unknown_cell_accuracy=(correct & unknown).sum(dim=(1, 2)).float()
        / unknown_denominator,
        satnet_constraint_fraction=satnet_constraint_fraction(predicted),
    )

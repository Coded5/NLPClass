from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any, Sequence

from .data import LabeledRow
from .labels import POLARITIES


@dataclass(frozen=True)
class LossSpec:
    name: str
    gamma: float = 0.0
    beta: float = 0.999
    neutral_multiplier: float = 1.0
    conflict_multiplier: float = 1.0

    def validate(self) -> None:
        if self.name not in {'weighted-ce', 'weighted-focal', 'class-balanced-focal'}:
            raise ValueError(f'Unsupported polarity loss: {self.name}')
        if self.gamma < 0:
            raise ValueError('Focal gamma must be nonnegative')
        if not 0 < self.beta < 1:
            raise ValueError('Class-balance beta must be in (0, 1)')
        if self.neutral_multiplier < 1 or self.conflict_multiplier < 1:
            raise ValueError('Neutral and conflict multipliers must be at least 1')

    @property
    def slug(self) -> str:
        suffix = ''
        if self.neutral_multiplier != 1 or self.conflict_multiplier != 1:
            suffix = (
                f'-n{self.neutral_multiplier:g}'
                f'-c{self.conflict_multiplier:g}'
            )
        if self.name == 'weighted-ce':
            return f'{self.name}{suffix}'
        if self.name == 'weighted-focal':
            return f'{self.name}-g{self.gamma:g}{suffix}'
        return f'{self.name}-b{self.beta:g}-g{self.gamma:g}{suffix}'

    def serialized(self) -> dict[str, float | str]:
        value: dict[str, float | str] = {
            'name': self.name, 'gamma': self.gamma, 'beta': self.beta,
        }
        if self.neutral_multiplier != 1 or self.conflict_multiplier != 1:
            value.update({
                'neutral_multiplier': self.neutral_multiplier,
                'conflict_multiplier': self.conflict_multiplier,
            })
        return value


SCREENING_LOSSES = (
    LossSpec('weighted-ce'),
    LossSpec('weighted-focal', gamma=1.0),
    LossSpec('weighted-focal', gamma=2.0),
    LossSpec('class-balanced-focal', gamma=2.0, beta=0.999),
)


def polarity_counts(rows: Sequence[LabeledRow]) -> list[int]:
    counts = Counter(row.polarity for row in rows)
    return [counts[label] for label in POLARITIES]


def inverse_frequency_weights(torch: Any, counts: Sequence[int]) -> Any:
    if not counts or any(count <= 0 for count in counts):
        raise ValueError('Every class must have at least one training example')
    total = sum(counts)
    return torch.tensor(
        [total / (len(counts) * count) for count in counts], dtype=torch.float32
    )


def class_balanced_weights(torch: Any, counts: Sequence[int], beta: float) -> Any:
    if not 0 < beta < 1:
        raise ValueError('Class-balance beta must be in (0, 1)')
    if not counts or any(count <= 0 for count in counts):
        raise ValueError('Every class must have at least one training example')
    weights = torch.tensor(
        [(1.0 - beta) / (1.0 - beta ** count) for count in counts],
        dtype=torch.float32,
    )
    return weights / weights.mean()


def loss_weights(torch: Any, rows: Sequence[LabeledRow], spec: LossSpec) -> Any:
    spec.validate()
    counts = polarity_counts(rows)
    if spec.name == 'class-balanced-focal':
        weights = class_balanced_weights(torch, counts, spec.beta)
    else:
        weights = inverse_frequency_weights(torch, counts)
    multipliers = torch.tensor(
        [1.0, 1.0, spec.neutral_multiplier, spec.conflict_multiplier],
        dtype=weights.dtype,
    )
    return weights * multipliers


def polarity_loss(
    torch: Any,
    logits: Any,
    targets: Any,
    weights: Any,
    spec: LossSpec,
) -> Any:
    """Return mean weighted CE/focal loss for non-empty targets."""
    spec.validate()
    if targets.numel() == 0:
        return logits.sum() * 0.0
    per_example = torch.nn.functional.cross_entropy(
        logits, targets, weight=weights, reduction='none'
    )
    denominator = weights[targets].sum().clamp_min(torch.finfo(logits.dtype).eps)
    if spec.name == 'weighted-ce':
        return per_example.sum() / denominator
    probabilities = torch.softmax(logits, dim=-1)
    target_probability = probabilities.gather(1, targets.unsqueeze(1)).squeeze(1)
    return (((1.0 - target_probability) ** spec.gamma) * per_example).sum() / denominator

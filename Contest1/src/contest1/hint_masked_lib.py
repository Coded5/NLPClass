#!/usr/bin/env python3
"""Shared building blocks for hint-masked GPT-2 predictive-keyboard training.

The first-character hint is part of the training objective: at every word
boundary GPT-2 is scored only over the root BPE tokens of training words that
begin with the target word's first letter. The exact same candidate set is used
at inference. This module is imported by both ``model.ipynb`` and
``contest1.train_hint_masked`` so the two cannot drift apart.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Literal, Sequence

import numpy as np
import pandas as pd
import torch
from torch import Tensor, nn
from torch.utils.data import IterableDataset
from tqdm.auto import tqdm


@dataclass
class TrainingLexicon:
    """Deterministic first-letter candidate vocabulary derived from a corpus."""

    word_counts: Counter
    training_lines: int
    training_tokens: int
    # word -> BPE token ids of " " + word
    word_bpe: dict[str, tuple[int, ...]]
    # first character -> most frequent full word (lexicographic tie-break)
    fallback_by_hint: dict[str, str]
    hint_chars: list[str]
    hint_to_idx: dict[str, int]
    # first character -> sorted list of root BPE token ids
    candidate_ids_by_hint: dict[str, list[int]]
    # first character -> {root BPE token id: most frequent word with that root}
    hint_token_word: dict[str, dict[int, str]]
    # first character -> root BPE token id -> words ordered by corpus frequency
    hint_root_words: dict[str, dict[int, list[str]]]
    # every word that can be produced by the root decoder
    predictable_words: set[str]
    # every corpus word with a non-empty BPE representation
    sequence_predictable_words: set[str]


@dataclass(frozen=True)
class CandidateScore:
    """Model and lexicon features for one full-word candidate."""

    word: str
    token_ids: tuple[int, ...]
    root_id: int
    root_rank: int
    within_root_rank: int
    root_logit: float
    root_log_probability: float
    root_margin: float
    root_entropy: float
    suffix_log_probability: float
    boundary_log_probability: float | None
    word_count: int

    @property
    def suffix_token_count(self) -> int:
        return max(0, len(self.token_ids) - 1)

    @property
    def suffix_mean_log_probability(self) -> float:
        return self.suffix_log_probability / max(1, self.suffix_token_count)


RerankMode = Literal["cross_root", "within_root"]


def build_boundary_token_ids(tokenizer) -> list[int]:
    """Token ids that can begin the next whitespace-delimited word or EOS."""
    boundary_ids = []
    for token_id in range(len(tokenizer)):
        text = tokenizer.decode(
            [token_id], clean_up_tokenization_spaces=False, skip_special_tokens=False
        )
        if text and text[0].isspace():
            boundary_ids.append(token_id)
    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    if eos_token_id is not None:
        boundary_ids.append(int(eos_token_id))
    return sorted(set(boundary_ids))


def count_corpus_words(
    path: str | Path,
    *,
    max_lines: int | None = None,
    progress_every: int = 100_000,
) -> tuple[Counter, int, int]:
    counts: Counter = Counter()
    line_count = 0
    token_count = 0
    with Path(path).open("r", encoding="utf-8") as corpus:
        for line_number, line in enumerate(corpus):
            if max_lines is not None and line_number >= max_lines:
                break
            words = line.split()
            counts.update(words)
            line_count += 1
            token_count += len(words)
            if progress_every and line_number and line_number % progress_every == 0:
                print(f"counted {line_number:,} lines", flush=True)
    return counts, line_count, token_count


def build_training_lexicon(
    tokenizer,
    train_path: str | Path,
    *,
    max_lines: int | None = None,
    progress_every: int = 100_000,
) -> TrainingLexicon:
    """One pass over the corpus: word BPE ids, per-hint root candidates.

    Candidate groups are keyed by (first character, first BPE token). A rare
    word such as "ike" has no space-merged BPE token and tokenizes as
    [" ", "ike"], so its first token is the bare space, which can begin words
    of many different first characters; grouping by hint keeps the mapping
    unambiguous.
    """
    word_counts, training_lines, training_tokens = count_corpus_words(
        train_path, max_lines=max_lines, progress_every=progress_every
    )
    fallback_by_hint: dict[str, str] = {}
    fallback_counts: dict[str, int] = {}
    word_bpe: dict[str, tuple[int, ...]] = {}
    # (first character, root token id) -> most frequent full word.
    hint_root_count: dict[tuple[str, int], int] = {}
    hint_root_word: dict[tuple[str, int], str] = {}
    hint_root_candidates: dict[tuple[str, int], list[str]] = defaultdict(list)

    for word, count in word_counts.items():
        if not word:
            continue
        hint = word[0]
        previous_count = fallback_counts.get(hint, -1)
        previous_word = fallback_by_hint.get(hint, word)
        if count > previous_count or (count == previous_count and word < previous_word):
            fallback_by_hint[hint] = word
            fallback_counts[hint] = count

        encoded = tokenizer.encode(" " + word, add_special_tokens=False)
        word_bpe[word] = tuple(encoded)
        if encoded:
            key = (hint, encoded[0])
            hint_root_candidates[key].append(word)
            root_count = hint_root_count.get(key, -1)
            root_word = hint_root_word.get(key)
            if count > root_count or (
                count == root_count and (root_word is None or word < root_word)
            ):
                hint_root_count[key] = count
                hint_root_word[key] = word

    hint_chars = sorted({word[0] for word in word_counts if word})
    hint_to_idx = {hint: index for index, hint in enumerate(hint_chars)}

    candidate_ids_by_hint: dict[str, list[int]] = defaultdict(list)
    hint_token_word: dict[str, dict[int, str]] = defaultdict(dict)
    for (hint, root), word in hint_root_word.items():
        candidate_ids_by_hint[hint].append(root)
        hint_token_word[hint][root] = word
    candidate_ids_by_hint = {
        hint: sorted(set(token_ids)) for hint, token_ids in candidate_ids_by_hint.items()
    }
    hint_token_word = {hint: dict(mapping) for hint, mapping in hint_token_word.items()}
    hint_root_words: dict[str, dict[int, list[str]]] = defaultdict(dict)
    for (hint, root), words in hint_root_candidates.items():
        hint_root_words[hint][root] = sorted(
            words, key=lambda word: (-word_counts[word], word)
        )
    hint_root_words = {hint: dict(mapping) for hint, mapping in hint_root_words.items()}
    predictable_words = set(hint_root_word.values())
    sequence_predictable_words = set(word_bpe)
    return TrainingLexicon(
        word_counts=word_counts,
        training_lines=training_lines,
        training_tokens=training_tokens,
        word_bpe=word_bpe,
        fallback_by_hint=fallback_by_hint,
        hint_chars=hint_chars,
        hint_to_idx=hint_to_idx,
        candidate_ids_by_hint=candidate_ids_by_hint,
        hint_token_word=hint_token_word,
        hint_root_words=hint_root_words,
        predictable_words=predictable_words,
        sequence_predictable_words=sequence_predictable_words,
    )


def dev_answer_coverage(
    answers: Sequence[str],
    hints: Sequence[str],
    predictable_words: set[str],
    fallback_by_hint: dict[str, str],
) -> float:
    reachable = [
        answer in predictable_words or answer == fallback_by_hint.get(hint, hint)
        for answer, hint in zip(answers, hints)
    ]
    return float(np.mean(reachable)) if reachable else 0.0


def count_pack_tokens(
    path: str | Path,
    word_bpe: dict[str, tuple[int, ...]],
    max_lines: int | None = None,
) -> tuple[int, int]:
    """Exact packed token count (word BPE ids + one EOS per non-empty line)."""
    total = 0
    lines_used = 0
    with Path(path).open("r", encoding="utf-8") as corpus:
        for line_number, line in enumerate(corpus):
            if max_lines is not None and line_number >= max_lines:
                break
            words = line.split()
            if not words:
                continue
            total += 1 + sum(len(word_bpe[word]) for word in words)
            lines_used += 1
    return total, lines_used


class HintMaskedPackedDataset(IterableDataset):
    """Stream words and pack their BPE ids into fixed blocks.

    Each sample is a (token_ids, hint_codes) pair. hint_codes[i] is
    hint_to_idx[word[0]] + 1 when token i starts an eligible training word (a
    word that is not the first word of its line), else 0. First words of lines
    are excluded because there is no usable left context after the EOS token.
    """

    def __init__(
        self,
        path: str | Path,
        word_bpe: dict[str, tuple[int, ...]],
        hint_to_idx: dict[str, int],
        eos_token_id: int,
        block_size: int,
        max_lines: int | None = None,
    ) -> None:
        super().__init__()
        if block_size < 2:
            raise ValueError("block_size must be at least 2")
        self.path = Path(path)
        self.word_bpe = word_bpe
        self.hint_to_idx = hint_to_idx
        self.eos_token_id = eos_token_id
        self.block_size = block_size
        self.max_lines = max_lines

    def __iter__(self):
        buffer_ids: list[int] = []
        buffer_hints: list[int] = []
        with self.path.open("r", encoding="utf-8") as corpus:
            for line_number, line in enumerate(corpus):
                if self.max_lines is not None and line_number >= self.max_lines:
                    break
                words = line.split()
                if not words:
                    continue
                for word_index, word in enumerate(words):
                    token_ids = self.word_bpe[word]
                    if not token_ids:
                        continue
                    hint_code = (
                        self.hint_to_idx[word[0]] + 1 if word_index > 0 else 0
                    )
                    for token_position, token_id in enumerate(token_ids):
                        buffer_ids.append(token_id)
                        buffer_hints.append(hint_code if token_position == 0 else 0)
                buffer_ids.append(self.eos_token_id)
                buffer_hints.append(0)
                while len(buffer_ids) >= self.block_size:
                    ids = torch.tensor(buffer_ids[: self.block_size], dtype=torch.long)
                    hints = torch.tensor(
                        buffer_hints[: self.block_size], dtype=torch.long
                    )
                    del buffer_ids[: self.block_size]
                    del buffer_hints[: self.block_size]
                    yield ids, hints


def collate_hint_blocks(
    samples: list[tuple[Tensor, Tensor]],
) -> tuple[Tensor, Tensor, Tensor]:
    """Shift annotations by one position so position p predicts token p+1.

    Returns (input_ids, label_root_ids, label_hint_ids). label_root_ids[p] is
    the root BPE id of the next token when that token is an eligible word start
    (-100 otherwise); label_hint_ids[p] is that hint's index (-1 otherwise).
    """
    ids = torch.stack([sample[0] for sample in samples])
    hints = torch.stack([sample[1] for sample in samples])
    batch_size, block_size = ids.shape
    label_root = torch.full_like(ids, -100)
    label_hint = torch.full_like(ids, -1)
    if block_size > 1:
        next_hints = hints[:, 1:]
        active = next_hints > 0
        label_root[:, :-1][active] = ids[:, 1:][active]
        label_hint[:, :-1][active] = next_hints[active] - 1
    return ids, label_root, label_hint


def build_hint_candidate_tables(
    hint_chars: list[str],
    candidate_ids_by_hint: dict[str, list[int]],
    vocab_size: int,
    target_device,
) -> tuple[list[Tensor], list[Tensor]]:
    """Per-hint candidate id tensors and token-id -> local-index lookups."""
    cand_tensors: list[Tensor] = []
    local_lookup: list[Tensor] = []
    for hint in hint_chars:
        ids = torch.tensor(
            candidate_ids_by_hint[hint], dtype=torch.long, device=target_device
        )
        cand_tensors.append(ids)
        lookup = torch.full((vocab_size,), -1, dtype=torch.long, device=target_device)
        lookup[ids] = torch.arange(len(ids), dtype=torch.long, device=target_device)
        local_lookup.append(lookup)
    return cand_tensors, local_lookup


def hint_masked_loss(
    logits: Tensor,
    label_root_ids: Tensor,
    label_hint_ids: Tensor,
    cand_tensors: list[Tensor],
    local_lookup: list[Tensor],
) -> tuple[Tensor, int, int]:
    """Restricted word-start cross-entropy.

    For every active position p, the logits at p are restricted to the BPE
    roots of words whose first character is the position's hint; the target is
    the root id of the word actually following position p. Continuation BPE
    tokens and EOS are never scored.
    """
    roots = label_root_ids.reshape(-1)
    hints = label_hint_ids.reshape(-1)
    flat_logits = logits.reshape(-1, logits.shape[-1])
    return _hint_group_loss(
        roots,
        hints,
        cand_tensors,
        local_lookup,
        lambda _hint_index, positions, cand: flat_logits[positions][:, cand],
        logits.device,
    )


def hint_masked_hidden_loss(
    hidden_states: Tensor,
    label_root_ids: Tensor,
    label_hint_ids: Tensor,
    cand_tensors: list[Tensor],
    local_lookup: list[Tensor],
    output_weight: Tensor,
    output_bias: Tensor | None = None,
) -> tuple[Tensor, int, int]:
    """Restricted loss without materializing full-vocabulary sequence logits."""
    roots = label_root_ids.reshape(-1)
    hints = label_hint_ids.reshape(-1)
    flat_hidden = hidden_states.reshape(-1, hidden_states.shape[-1])

    def candidate_rows(_hint_index: int, positions: Tensor, cand: Tensor) -> Tensor:
        rows = flat_hidden[positions] @ output_weight[cand].transpose(0, 1)
        if output_bias is not None:
            rows = rows + output_bias[cand]
        return rows

    return _hint_group_loss(
        roots,
        hints,
        cand_tensors,
        local_lookup,
        candidate_rows,
        hidden_states.device,
    )


def _hint_group_loss(
    roots: Tensor,
    hints: Tensor,
    cand_tensors: list[Tensor],
    local_lookup: list[Tensor],
    candidate_rows,
    device,
) -> tuple[Tensor, int, int]:
    loss_sum = torch.zeros((), dtype=torch.float32, device=device)
    correct = 0
    active_total = 0
    for hint_index, cand in enumerate(cand_tensors):
        active = (roots >= 0) & (hints == hint_index)
        positions = torch.nonzero(active, as_tuple=False).squeeze(1)
        if positions.numel() == 0:
            continue
        rows = candidate_rows(hint_index, positions, cand).float()
        local = local_lookup[hint_index][roots[positions]]
        if int((local < 0).sum().item()) > 0:
            bad = roots[positions][local < 0].tolist()
            raise ValueError(
                f"target root id is not in the hint candidate set: {bad[:5]}"
            )
        log_probs = rows.log_softmax(dim=1)
        target_log_prob = log_probs.gather(1, local.view(-1, 1)).squeeze(1)
        loss_sum = loss_sum + (-target_log_prob).sum()
        correct += int((log_probs.argmax(dim=1) == local).sum().item())
        active_total += int(positions.numel())
    if active_total == 0:
        raise ValueError("batch contains no masked word-start targets")
    return loss_sum / active_total, correct, active_total


def _position_ids(attention_mask: Tensor) -> Tensor:
    position_ids = attention_mask.long().cumsum(dim=1) - 1
    return position_ids.masked_fill(attention_mask == 0, 0)


@torch.inference_mode()
def score_candidate_batch(
    language_model: nn.Module,
    tokenizer,
    items: list[tuple[list[int], CandidateScore]],
    *,
    batch_size: int,
    boundary_token_ids: Sequence[int] | None = None,
) -> list[CandidateScore]:
    """Score candidates from different contexts in shared model batches."""
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if any(not context_ids for context_ids, _candidate in items):
        raise ValueError("context_ids must not be empty")
    if not items:
        return []

    model_device = next(language_model.parameters()).device
    pad_token_id = tokenizer.pad_token_id
    model_config = getattr(language_model, "config", None)
    max_positions = getattr(model_config, "max_position_embeddings", None)
    if max_positions is None:
        max_positions = getattr(model_config, "n_positions", None)
    include_boundary = boundary_token_ids is not None
    boundary_ids = None
    if boundary_token_ids is not None:
        boundary_ids = torch.tensor(
            sorted(set(boundary_token_ids)), dtype=torch.long, device=model_device
        )
        if boundary_ids.numel() == 0:
            raise ValueError("boundary_token_ids must not be empty")

    scored: list[CandidateScore] = []
    for start in range(0, len(items), batch_size):
        chunk = items[start : start + batch_size]
        chunk_scores: list[CandidateScore | None] = [None] * len(chunk)
        pending = [
            (index, item)
            for index, item in enumerate(chunk)
            if include_boundary or item[1].suffix_token_count
        ]
        for index, (_context_ids, candidate) in enumerate(chunk):
            if not include_boundary and not candidate.suffix_token_count:
                chunk_scores[index] = candidate
        if not pending:
            scored.extend(candidate for candidate in chunk_scores if candidate is not None)
            continue

        sequences = []
        for _index, (context_ids, candidate) in pending:
            candidate_ids = (
                candidate.token_ids if include_boundary else candidate.token_ids[:-1]
            )
            if max_positions is not None:
                if len(candidate_ids) >= max_positions:
                    raise ValueError("candidate word exceeds the model positional limit")
                context_ids = context_ids[-(max_positions - len(candidate_ids)) :]
            sequences.append(context_ids + list(candidate_ids))
        max_length = max(map(len, sequences))
        input_ids = torch.full(
            (len(sequences), max_length),
            pad_token_id,
            dtype=torch.long,
            device=model_device,
        )
        attention_mask = torch.zeros_like(input_ids)
        for row, sequence in enumerate(sequences):
            input_ids[row, -len(sequence) :] = torch.tensor(
                sequence, dtype=torch.long, device=model_device
            )
            attention_mask[row, -len(sequence) :] = 1
        keep_logits = max(
            len(candidate.token_ids) if include_boundary else candidate.suffix_token_count
            for _index, (_context_ids, candidate) in pending
        )
        with torch.autocast(
            device_type=model_device.type,
            dtype=torch.float16,
            enabled=model_device.type == "cuda",
        ):
            logits = language_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=_position_ids(attention_mask),
                logits_to_keep=keep_logits,
            ).logits.float()

        for row, (chunk_index, (_context_ids, candidate)) in enumerate(pending):
            suffix_length = candidate.suffix_token_count
            suffix_score = 0.0
            if suffix_length:
                end = -1 if include_boundary else None
                start_position = -(suffix_length + 1) if include_boundary else -suffix_length
                suffix_logits = logits[row, start_position:end, :]
                suffix_targets = torch.tensor(
                    candidate.token_ids[1:], dtype=torch.long, device=model_device
                )
                suffix_score = float(
                    suffix_logits.log_softmax(dim=1)
                    .gather(1, suffix_targets.view(-1, 1))
                    .sum()
                )
            boundary_score = None
            if boundary_ids is not None:
                boundary_logits = logits[row, -1].log_softmax(dim=0)
                boundary_score = float(
                    torch.logsumexp(boundary_logits.index_select(0, boundary_ids), dim=0)
                )
            chunk_scores[chunk_index] = CandidateScore(
                    word=candidate.word,
                    token_ids=candidate.token_ids,
                    root_id=candidate.root_id,
                    root_rank=candidate.root_rank,
                    within_root_rank=candidate.within_root_rank,
                    root_logit=candidate.root_logit,
                    root_log_probability=candidate.root_log_probability,
                    root_margin=candidate.root_margin,
                    root_entropy=candidate.root_entropy,
                    suffix_log_probability=suffix_score,
                    boundary_log_probability=boundary_score,
                    word_count=candidate.word_count,
                )
        if any(candidate is None for candidate in chunk_scores):
            raise RuntimeError("candidate scorer failed to produce every requested score")
        scored.extend(candidate for candidate in chunk_scores if candidate is not None)
    return scored


@torch.inference_mode()
def score_word_candidates(
    language_model: nn.Module,
    tokenizer,
    context_ids: list[int],
    candidates: list[CandidateScore],
    *,
    batch_size: int,
    boundary_token_ids: Sequence[int] | None = None,
) -> list[CandidateScore]:
    """Add suffix and optional completed-word boundary scores to candidates."""
    return score_candidate_batch(
        language_model,
        tokenizer,
        [(context_ids, candidate) for candidate in candidates],
        batch_size=batch_size,
        boundary_token_ids=boundary_token_ids,
    )


def _expanded_cache(past_key_values, row_indices: Tensor):
    cache_type = type(past_key_values)
    legacy = (
        past_key_values.to_legacy_cache()
        if hasattr(past_key_values, "to_legacy_cache")
        else past_key_values
    )
    expanded = tuple(
        tuple(state.index_select(0, row_indices) for state in layer)
        for layer in legacy
    )
    from_legacy_cache = getattr(cache_type, "from_legacy_cache", None)
    return from_legacy_cache(expanded) if from_legacy_cache else expanded


@torch.inference_mode()
def score_candidate_batch_with_cache(
    language_model: nn.Module,
    tokenizer,
    past_key_values,
    context_attention_mask: Tensor,
    items: list[tuple[int, CandidateScore]],
    *,
    batch_size: int,
    boundary_token_ids: Sequence[int] | None = None,
) -> list[CandidateScore]:
    """Score candidate continuations while reusing their context KV cache."""
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if not items:
        return []
    model_device = context_attention_mask.device
    include_boundary = boundary_token_ids is not None
    boundary_ids = None
    if boundary_token_ids is not None:
        boundary_ids = torch.tensor(
            sorted(set(boundary_token_ids)), dtype=torch.long, device=model_device
        )
        if boundary_ids.numel() == 0:
            raise ValueError("boundary_token_ids must not be empty")

    scored: list[CandidateScore] = []
    for start in range(0, len(items), batch_size):
        chunk = items[start : start + batch_size]
        chunk_scores: list[CandidateScore | None] = [None] * len(chunk)
        pending = [
            (index, item)
            for index, item in enumerate(chunk)
            if include_boundary or item[1].suffix_token_count
        ]
        for index, (_row, candidate) in enumerate(chunk):
            if not include_boundary and not candidate.suffix_token_count:
                chunk_scores[index] = candidate
        if pending:
            row_indices = torch.tensor(
                [row for _index, (row, _candidate) in pending],
                dtype=torch.long,
                device=model_device,
            )
            sequences = [
                list(candidate.token_ids if include_boundary else candidate.token_ids[:-1])
                for _index, (_row, candidate) in pending
            ]
            max_length = max(map(len, sequences))
            input_ids = torch.full(
                (len(sequences), max_length),
                tokenizer.pad_token_id,
                dtype=torch.long,
                device=model_device,
            )
            candidate_mask = torch.zeros_like(input_ids)
            position_ids = torch.zeros_like(input_ids)
            context_lengths = context_attention_mask.sum(dim=1).index_select(
                0, row_indices
            )
            for row, sequence in enumerate(sequences):
                length = len(sequence)
                input_ids[row, :length] = torch.tensor(
                    sequence, dtype=torch.long, device=model_device
                )
                candidate_mask[row, :length] = 1
                position_ids[row, :length] = context_lengths[row] + torch.arange(
                    length, dtype=torch.long, device=model_device
                )
            full_attention_mask = torch.cat(
                (
                    context_attention_mask.index_select(0, row_indices),
                    candidate_mask,
                ),
                dim=1,
            )
            with torch.autocast(
                device_type=model_device.type,
                dtype=torch.float16,
                enabled=model_device.type == "cuda",
            ):
                logits = language_model(
                    input_ids=input_ids,
                    attention_mask=full_attention_mask,
                    position_ids=position_ids,
                    past_key_values=_expanded_cache(past_key_values, row_indices),
                    use_cache=False,
                ).logits.float()

            for output_row, (chunk_index, (_source_row, candidate)) in enumerate(pending):
                suffix_length = candidate.suffix_token_count
                suffix_score = 0.0
                if suffix_length:
                    suffix_logits = logits[output_row, :suffix_length, :]
                    suffix_targets = torch.tensor(
                        candidate.token_ids[1:], dtype=torch.long, device=model_device
                    )
                    suffix_score = float(
                        suffix_logits.log_softmax(dim=1)
                        .gather(1, suffix_targets.view(-1, 1))
                        .sum()
                    )
                boundary_score = None
                if boundary_ids is not None:
                    boundary_position = len(candidate.token_ids) - 1
                    boundary_logits = logits[
                        output_row, boundary_position
                    ].log_softmax(dim=0)
                    boundary_score = float(
                        torch.logsumexp(
                            boundary_logits.index_select(0, boundary_ids), dim=0
                        )
                    )
                chunk_scores[chunk_index] = CandidateScore(
                    word=candidate.word,
                    token_ids=candidate.token_ids,
                    root_id=candidate.root_id,
                    root_rank=candidate.root_rank,
                    within_root_rank=candidate.within_root_rank,
                    root_logit=candidate.root_logit,
                    root_log_probability=candidate.root_log_probability,
                    root_margin=candidate.root_margin,
                    root_entropy=candidate.root_entropy,
                    suffix_log_probability=suffix_score,
                    boundary_log_probability=boundary_score,
                    word_count=candidate.word_count,
                )
        if any(candidate is None for candidate in chunk_scores):
            raise RuntimeError("cached candidate scorer missed a requested score")
        scored.extend(candidate for candidate in chunk_scores if candidate is not None)
    return scored


def rerank_candidate_scores(
    candidates: Sequence[CandidateScore],
    *,
    suffix_weight: float = 1.0,
    suffix_length_penalty: float = 0.0,
    frequency_weight: float = 0.0,
    boundary_weight: float = 0.0,
) -> list[CandidateScore]:
    """Rank candidates with independently tunable score components."""
    if suffix_length_penalty < 0:
        raise ValueError("suffix_length_penalty must be non-negative")

    def score(candidate: CandidateScore) -> float:
        length_scale = max(1, candidate.suffix_token_count) ** suffix_length_penalty
        value = candidate.root_log_probability
        value += suffix_weight * candidate.suffix_log_probability / length_scale
        value += frequency_weight * math.log1p(candidate.word_count)
        if candidate.boundary_log_probability is not None:
            value += boundary_weight * candidate.boundary_log_probability
        return value

    return sorted(candidates, key=lambda candidate: (-score(candidate), candidate.word))


@torch.inference_mode()
def _score_word_candidates(
    language_model: nn.Module,
    tokenizer,
    context_ids: list[int],
    candidates: list[tuple[str, tuple[int, ...], float]],
    *,
    batch_size: int,
) -> list[tuple[str, float]]:
    features = [
        CandidateScore(
            word=word,
            token_ids=tokens,
            root_id=int(tokens[0]),
            root_rank=0,
            within_root_rank=0,
            root_logit=root_score,
            root_log_probability=root_score,
            root_margin=0.0,
            root_entropy=0.0,
            suffix_log_probability=0.0,
            boundary_log_probability=None,
            word_count=0,
        )
        for word, tokens, root_score in candidates
    ]
    scored = score_word_candidates(
        language_model, tokenizer, context_ids, features, batch_size=batch_size
    )
    return [
        (candidate.word, candidate.root_log_probability + candidate.suffix_log_probability)
        for candidate in scored
    ]


@torch.inference_mode()
def score_candidates(
    language_model: nn.Module,
    tokenizer,
    contexts: list[str],
    hints: list[str],
    *,
    candidate_ids_by_hint: dict[str, list[int]],
    hint_root_words: dict[str, dict[int, list[str]]],
    word_bpe: dict[str, tuple[int, ...]],
    word_counts: Counter | dict[str, int] | None = None,
    words_per_root: int = 3,
    root_beam: int = 5,
    mode: RerankMode = "cross_root",
    cross_root_confidence_gate: float | None = None,
    batch_size: int = 128,
    candidate_batch_size: int = 64,
    max_context_tokens: int = 256,
    boundary_token_ids: Sequence[int] | None = None,
    score_suffixes: bool = True,
) -> list[list[CandidateScore]]:
    """Return component scores for an explicit full-word candidate pool."""
    if len(contexts) != len(hints):
        raise ValueError("contexts and hints must have equal length")
    if any(len(hint) != 1 for hint in hints):
        raise ValueError("each hint must contain exactly one character")
    if min(words_per_root, root_beam, batch_size, candidate_batch_size) < 1:
        raise ValueError("candidate sizes and batch sizes must be positive")
    if max_context_tokens < 1:
        raise ValueError("max_context_tokens must be positive")
    if mode not in ("cross_root", "within_root"):
        raise ValueError(f"unsupported rerank mode: {mode}")
    if cross_root_confidence_gate is not None and not 0 <= cross_root_confidence_gate <= 1:
        raise ValueError("cross-root confidence gate must be between zero and one")

    model_device = next(language_model.parameters()).device
    model_config = getattr(language_model, "config", None)
    model_limit = getattr(model_config, "max_position_embeddings", None)
    if model_limit is None:
        model_limit = getattr(model_config, "n_positions", max_context_tokens)
    effective_context_limit = min(max_context_tokens, model_limit)
    if model_limit is not None:
        possible_words = (
            word
            for roots in hint_root_words.values()
            for words in roots.values()
            for word in words[:words_per_root]
        )
        max_candidate_tokens = max(
            (len(word_bpe.get(word, ())) for word in possible_words), default=1
        )
        reserved_tokens = (
            max_candidate_tokens
            if boundary_token_ids is not None
            else max(0, max_candidate_tokens - 1)
        )
        effective_context_limit = min(
            effective_context_limit, model_limit - reserved_tokens
        )
        if effective_context_limit < 1:
            raise ValueError("candidate words leave no room for model context")
    candidate_tensors = {
        hint: torch.tensor(ids, dtype=torch.long, device=model_device)
        for hint, ids in candidate_ids_by_hint.items()
    }
    counts = word_counts or {}
    all_scores: list[list[CandidateScore]] = []
    was_training = language_model.training
    language_model.eval()
    try:
        for start in range(0, len(contexts), batch_size):
            batch_contexts = contexts[start : start + batch_size]
            batch_hints = hints[start : start + batch_size]
            encoded = tokenizer(
                batch_contexts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=effective_context_limit,
            ).to(model_device)
            with torch.autocast(
                device_type=model_device.type,
                dtype=torch.float16,
                enabled=model_device.type == "cuda",
            ):
                model_output = language_model(
                    **encoded,
                    position_ids=_position_ids(encoded["attention_mask"]),
                    logits_to_keep=1,
                    use_cache=True,
                )
                next_token_logits = model_output.logits[:, -1, :]

            candidate_groups: list[list[CandidateScore]] = []
            for row, hint in enumerate(batch_hints):
                candidate_ids = candidate_tensors.get(hint)
                if candidate_ids is None or candidate_ids.numel() == 0:
                    candidate_groups.append([])
                    continue
                candidate_logits = next_token_logits[row].index_select(0, candidate_ids)
                root_log_probs = candidate_logits.float().log_softmax(dim=0)
                root_probs = root_log_probs.exp()
                root_entropy = float(-(root_probs * root_log_probs).sum())
                top_two = torch.topk(candidate_logits, k=min(2, candidate_ids.numel())).values
                root_margin = (
                    float(top_two[0] - top_two[1]) if top_two.numel() > 1 else math.inf
                )
                top_root_probability = float(root_log_probs.max().exp())
                within_root_only = mode == "within_root" or (
                    cross_root_confidence_gate is not None
                    and top_root_probability > cross_root_confidence_gate
                )
                selected_root_count = 1 if within_root_only else root_beam
                selected_root_count = min(selected_root_count, candidate_ids.numel())
                root_positions = torch.topk(
                    candidate_logits, k=selected_root_count
                ).indices
                root_words = hint_root_words.get(hint, {})
                candidates: list[CandidateScore] = []
                for root_rank, position in enumerate(root_positions.tolist(), start=1):
                    root = int(candidate_ids[position])
                    words = root_words.get(root, [])[:words_per_root]
                    for within_root_rank, word in enumerate(words, start=1):
                        tokens = word_bpe.get(word, ())
                        if not tokens:
                            continue
                        candidates.append(
                            CandidateScore(
                                word=word,
                                token_ids=tokens,
                                root_id=root,
                                root_rank=root_rank,
                                within_root_rank=within_root_rank,
                                root_logit=float(candidate_logits[position]),
                                root_log_probability=float(root_log_probs[position]),
                                root_margin=root_margin,
                                root_entropy=root_entropy,
                                suffix_log_probability=0.0,
                                boundary_log_probability=None,
                                word_count=int(counts.get(word, 0)),
                            )
                        )
                candidate_groups.append(candidates)
            cached_jobs = [
                (row, candidate)
                for row, candidates in enumerate(candidate_groups)
                for candidate in candidates
            ]
            past_key_values = getattr(model_output, "past_key_values", None)
            if not score_suffixes and boundary_token_ids is None:
                scored = [candidate for _row, candidate in cached_jobs]
            elif past_key_values is not None:
                scored = score_candidate_batch_with_cache(
                    language_model,
                    tokenizer,
                    past_key_values,
                    encoded["attention_mask"],
                    cached_jobs,
                    batch_size=candidate_batch_size,
                    boundary_token_ids=boundary_token_ids,
                )
            else:
                context_ids = [
                    encoded["input_ids"][row][
                        encoded["attention_mask"][row].bool()
                    ].tolist()
                    for row in range(len(batch_contexts))
                ]
                scored = score_candidate_batch(
                    language_model,
                    tokenizer,
                    [
                        (context_ids[row], candidate)
                        for row, candidate in cached_jobs
                    ],
                    batch_size=candidate_batch_size,
                    boundary_token_ids=boundary_token_ids,
                )
            score_offset = 0
            for candidates in candidate_groups:
                next_offset = score_offset + len(candidates)
                all_scores.append(scored[score_offset:next_offset])
                score_offset = next_offset
            if score_offset != len(scored):
                raise RuntimeError(
                    "candidate scorer returned an unexpected number of rows"
                )
    finally:
        language_model.train(was_training)
    return all_scores


@torch.inference_mode()
def predict_top_k(
    language_model: nn.Module,
    tokenizer,
    contexts: list[str],
    hints: list[str],
    *,
    candidate_ids_by_hint: dict[str, list[int]],
    hint_token_word: dict[str, dict[int, str]],
    fallback_by_hint: dict[str, str],
    hint_root_words: dict[str, dict[int, list[str]]] | None = None,
    word_bpe: dict[str, tuple[int, ...]] | None = None,
    word_counts: Counter | dict[str, int] | None = None,
    rerank_words_per_root: int = 1,
    rerank_root_beam: int | None = None,
    rerank_batch_size: int = 64,
    rerank_mode: RerankMode = "cross_root",
    rerank_suffix_weight: float = 1.0,
    rerank_suffix_length_penalty: float = 0.0,
    rerank_frequency_weight: float = 0.0,
    rerank_boundary_weight: float = 0.0,
    boundary_token_ids: Sequence[int] | None = None,
    top_k: int = 5,
    batch_size: int = 128,
    max_context_tokens: int = 256,
) -> list[list[str]]:
    """Top-k hint-constrained predictions.

    The model scores only the root BPE tokens of words matching each hint, then
    each chosen root is mapped to its representative word for that hint.
    """
    if len(contexts) != len(hints):
        raise ValueError("contexts and hints must have equal length")
    if top_k < 1 or batch_size < 1 or rerank_batch_size < 1:
        raise ValueError("top_k and batch_size must be positive")
    if max_context_tokens < 1:
        raise ValueError("max_context_tokens must be positive")
    if rerank_words_per_root < 1:
        raise ValueError("rerank_words_per_root must be positive")
    if rerank_root_beam is not None and rerank_root_beam < 1:
        raise ValueError("rerank_root_beam must be positive")
    if rerank_mode not in ("cross_root", "within_root"):
        raise ValueError(f"unsupported rerank mode: {rerank_mode}")
    if rerank_words_per_root > 1 and (hint_root_words is None or word_bpe is None):
        raise ValueError(
            "collision reranking requires hint_root_words and word_bpe"
        )
    if any(len(hint) != 1 for hint in hints):
        raise ValueError("each hint must contain exactly one character")

    if rerank_words_per_root > 1:
        beam = rerank_root_beam or top_k
        scoring_beam = max(top_k, beam) if rerank_mode == "within_root" else beam
        candidate_rows = score_candidates(
            language_model,
            tokenizer,
            contexts,
            hints,
            candidate_ids_by_hint=candidate_ids_by_hint,
            hint_root_words=hint_root_words or {},
            word_bpe=word_bpe or {},
            word_counts=word_counts,
            words_per_root=rerank_words_per_root,
            root_beam=scoring_beam,
            mode="cross_root",
            batch_size=batch_size,
            candidate_batch_size=rerank_batch_size,
            max_context_tokens=max_context_tokens,
            boundary_token_ids=boundary_token_ids,
        )
        predictions = []
        for hint, candidates in zip(hints, candidate_rows):
            if rerank_mode == "within_root":
                rerank_pool = [
                    candidate for candidate in candidates if candidate.root_rank == 1
                ]
            else:
                rerank_pool = candidates
            ranked = rerank_candidate_scores(
                rerank_pool,
                suffix_weight=rerank_suffix_weight,
                suffix_length_penalty=rerank_suffix_length_penalty,
                frequency_weight=rerank_frequency_weight,
                boundary_weight=rerank_boundary_weight,
            )
            ranked_words = [candidate.word for candidate in ranked]
            if rerank_mode == "within_root":
                ranked_words.extend(
                    candidate.word
                    for candidate in candidates
                    if candidate.root_rank > 1 and candidate.within_root_rank == 1
                )
            predictions.append(
                ranked_words[:top_k]
                if ranked_words
                else [fallback_by_hint.get(hint, hint)]
            )
        return predictions

    was_training = language_model.training
    model_device = next(language_model.parameters()).device
    model_config = getattr(language_model, "config", None)
    model_limit = getattr(model_config, "max_position_embeddings", None)
    if model_limit is None:
        model_limit = getattr(model_config, "n_positions", max_context_tokens)
    effective_context_limit = min(max_context_tokens, model_limit)
    language_model.eval()
    candidate_tensors = {
        hint: torch.tensor(ids, dtype=torch.long, device=model_device)
        for hint, ids in candidate_ids_by_hint.items()
    }
    predictions: list[list[str]] = []

    for start in range(0, len(contexts), batch_size):
        batch_contexts = contexts[start : start + batch_size]
        batch_hints = hints[start : start + batch_size]
        encoded = tokenizer(
            batch_contexts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=effective_context_limit,
        ).to(model_device)
        with torch.autocast(
            device_type=model_device.type,
            dtype=torch.float16,
            enabled=model_device.type == "cuda",
        ):
            next_token_logits = language_model(
                **encoded,
                position_ids=_position_ids(encoded["attention_mask"]),
                logits_to_keep=1,
            ).logits[:, -1, :]

        for row, hint in enumerate(batch_hints):
            candidate_ids = candidate_tensors.get(hint)
            if candidate_ids is None or candidate_ids.numel() == 0:
                predictions.append([fallback_by_hint.get(hint, hint)])
                continue

            candidate_logits = next_token_logits[row].index_select(0, candidate_ids)
            count = min(top_k, candidate_ids.numel())
            positions = torch.topk(candidate_logits, k=count).indices
            hint_words = hint_token_word.get(hint, {})
            predictions.append(
                [hint_words[int(candidate_ids[position].item())] for position in positions]
            )

    language_model.train(was_training)
    return predictions


def evaluate_model(
    language_model: nn.Module,
    tokenizer,
    frame: pd.DataFrame,
    *,
    candidate_ids_by_hint: dict[str, list[int]],
    hint_token_word: dict[str, dict[int, str]],
    fallback_by_hint: dict[str, str],
    predictable_words: set[str],
    hint_root_words: dict[str, dict[int, list[str]]] | None = None,
    word_bpe: dict[str, tuple[int, ...]] | None = None,
    word_counts: Counter | dict[str, int] | None = None,
    rerank_words_per_root: int = 1,
    rerank_root_beam: int | None = None,
    rerank_batch_size: int = 64,
    rerank_mode: RerankMode = "cross_root",
    rerank_suffix_weight: float = 1.0,
    rerank_suffix_length_penalty: float = 0.0,
    rerank_frequency_weight: float = 0.0,
    rerank_boundary_weight: float = 0.0,
    boundary_token_ids: Sequence[int] | None = None,
    label: str = "model",
    top_k: int = 5,
    eval_batch_size: int = 128,
    max_context_tokens: int = 256,
) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    """Evaluate hint-constrained top-1/top-k accuracy on ``frame``."""
    if frame.empty:
        raise ValueError("evaluation frame must not be empty")
    top_predictions = predict_top_k(
        language_model,
        tokenizer,
        frame["context"].tolist(),
        frame["first letter"].tolist(),
        candidate_ids_by_hint=candidate_ids_by_hint,
        hint_token_word=hint_token_word,
        fallback_by_hint=fallback_by_hint,
        hint_root_words=hint_root_words,
        word_bpe=word_bpe,
        word_counts=word_counts,
        rerank_words_per_root=rerank_words_per_root,
        rerank_root_beam=rerank_root_beam,
        rerank_batch_size=rerank_batch_size,
        rerank_mode=rerank_mode,
        rerank_suffix_weight=rerank_suffix_weight,
        rerank_suffix_length_penalty=rerank_suffix_length_penalty,
        rerank_frequency_weight=rerank_frequency_weight,
        rerank_boundary_weight=rerank_boundary_weight,
        boundary_token_ids=boundary_token_ids,
        top_k=top_k,
        batch_size=eval_batch_size,
        max_context_tokens=max_context_tokens,
    )
    details = frame.copy()
    details["prediction"] = [items[0] for items in top_predictions]
    details["top_k_predictions"] = top_predictions
    details["correct"] = details["prediction"].eq(details["answer"])
    details["top_k_correct"] = [
        answer in candidates
        for answer, candidates in zip(details["answer"], top_predictions)
    ]
    reachable_words = predictable_words
    if rerank_words_per_root > 1 and hint_root_words is not None:
        reachable_words = {
            word
            for roots in hint_root_words.values()
            for words in roots.values()
            for word in words[:rerank_words_per_root]
        }
    reachable = [
        answer in reachable_words or answer == fallback_by_hint.get(hint, hint)
        for answer, hint in zip(details["answer"], details["first letter"])
    ]
    alphanumeric_hints = details["first letter"].str.isalnum()
    alphanumeric_correct = details.loc[alphanumeric_hints, "correct"]
    non_alphanumeric_correct = details.loc[~alphanumeric_hints, "correct"]
    metrics: dict[str, object] = {
        "model": label,
        "examples": len(details),
        "top_1_accuracy": float(details["correct"].to_numpy().mean()),
        f"top_{top_k}_accuracy": float(details["top_k_correct"].to_numpy().mean()),
        "alphanumeric_top_1_accuracy": (
            float(alphanumeric_correct.to_numpy().mean())
            if len(alphanumeric_correct)
            else None
        ),
        "non_alphanumeric_top_1_accuracy": (
            float(non_alphanumeric_correct.to_numpy().mean())
            if len(non_alphanumeric_correct)
            else None
        ),
        "candidate_coverage": float(np.mean(reachable)),
    }
    per_hint = (
        details.groupby("first letter", as_index=False)
        .agg(
            examples=("answer", "size"),
            top_1_accuracy=("correct", "mean"),
            top_k_accuracy=("top_k_correct", "mean"),
        )
        .sort_values(by=["examples", "first letter"], ascending=[False, True])
    )
    return metrics, details, per_hint


def load_prediction_resources(path: str | Path) -> dict:
    """Load JSON prediction metadata and restore integer/token tuple types."""
    with Path(path).open("r", encoding="utf-8") as file:
        resources = json.load(file)
    resources["hint_token_word"] = {
        hint: {int(root): word for root, word in mapping.items()}
        for hint, mapping in resources["hint_token_word"].items()
    }
    if "hint_root_words" in resources:
        resources["hint_root_words"] = {
            hint: {int(root): words for root, words in mapping.items()}
            for hint, mapping in resources["hint_root_words"].items()
        }
    if "word_bpe" in resources:
        resources["word_bpe"] = {
            word: tuple(token_ids)
            for word, token_ids in resources["word_bpe"].items()
        }
    return resources


def write_test_predictions(
    language_model: nn.Module,
    tokenizer,
    test_csv_path: str | Path,
    out_path: str | Path,
    *,
    candidate_ids_by_hint: dict[str, list[int]],
    hint_token_word: dict[str, dict[int, str]],
    fallback_by_hint: dict[str, str],
    hint_root_words: dict[str, dict[int, list[str]]] | None = None,
    word_bpe: dict[str, tuple[int, ...]] | None = None,
    word_counts: Counter | dict[str, int] | None = None,
    rerank_words_per_root: int = 1,
    rerank_root_beam: int | None = None,
    rerank_batch_size: int = 64,
    rerank_mode: RerankMode = "cross_root",
    rerank_suffix_weight: float = 1.0,
    rerank_suffix_length_penalty: float = 0.0,
    rerank_frequency_weight: float = 0.0,
    rerank_boundary_weight: float = 0.0,
    boundary_token_ids: Sequence[int] | None = None,
    max_context_tokens: int = 256,
    batch_size: int = 128,
) -> int:
    """Write one predicted word per test row (header: context, first letter).

    The file is written atomically and the output is validated to have exactly
    one line per input row.
    """
    test_df = pd.read_csv(test_csv_path, keep_default_na=False)
    if list(test_df.columns) != ["context", "first letter"]:
        raise ValueError(
            f"Expected columns ['context', 'first letter'], got {list(test_df.columns)}"
        )
    if test_df[["context", "first letter"]].eq("").to_numpy().any():
        raise ValueError("Test data contains empty fields")
    if not test_df["first letter"].str.len().eq(1).all():
        raise ValueError("Every first-character hint must contain exactly one character")

    predictions = predict_top_k(
        language_model,
        tokenizer,
        test_df["context"].tolist(),
        test_df["first letter"].tolist(),
        candidate_ids_by_hint=candidate_ids_by_hint,
        hint_token_word=hint_token_word,
        fallback_by_hint=fallback_by_hint,
        hint_root_words=hint_root_words,
        word_bpe=word_bpe,
        word_counts=word_counts,
        rerank_words_per_root=rerank_words_per_root,
        rerank_root_beam=rerank_root_beam,
        rerank_batch_size=rerank_batch_size,
        rerank_mode=rerank_mode,
        rerank_suffix_weight=rerank_suffix_weight,
        rerank_suffix_length_penalty=rerank_suffix_length_penalty,
        rerank_frequency_weight=rerank_frequency_weight,
        rerank_boundary_weight=rerank_boundary_weight,
        boundary_token_ids=boundary_token_ids,
        top_k=1,
        batch_size=batch_size,
        max_context_tokens=max_context_tokens,
    )
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [items[0] for items in predictions]
    if len(lines) != len(test_df):
        raise ValueError(
            f"produced {len(lines)} predictions for {len(test_df)} test rows"
        )
    if any("\n" in line or not line for line in lines):
        raise ValueError("predictions must be single non-empty words")
    temporary = out_path.with_name(out_path.name + ".tmp")
    temporary.write_text("\n".join(lines) + "\n", encoding="utf-8")
    temporary.replace(out_path)
    return len(lines)

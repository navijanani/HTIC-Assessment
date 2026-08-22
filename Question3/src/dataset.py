"""Utilities for checking the raw Aksharantar Hindi splits."""

from __future__ import annotations
import csv
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Iterable, Sequence

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

from .config import CONFIG

if TYPE_CHECKING:
    from .vocabulary import CharacterVocabulary


@dataclass(frozen=True)
class WordPair:
    """One Romanized input and its Devanagari target."""

    source: str
    target: str


class TransliterationDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """Encode transliteration pairs without padding individual samples."""

    def __init__(self, pairs: list[WordPair], source_vocab: CharacterVocabulary, target_vocab: CharacterVocabulary) -> None:
        self.pairs = pairs
        self.source_vocab = source_vocab
        self.target_vocab = target_vocab

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        pair = self.pairs[index]
        source_ids = self.source_vocab.encode(pair.source)
        target_ids = self.target_vocab.encode(pair.target, add_boundaries=True)
        return torch.tensor(source_ids, dtype=torch.long), torch.tensor(target_ids, dtype=torch.long)


def read_pairs(path: Path) -> list[WordPair]:
    """Read a headerless two-column CSV without changing the source file."""
    pairs: list[WordPair] = []
    with path.open("r", encoding="utf-8", newline="") as csv_file:
        for row_number, row in enumerate(csv.reader(csv_file), start=1):
            if len(row) != 2:
                raise ValueError(f"{path} row {row_number} has {len(row)} columns; expected 2")
            pairs.append(WordPair(row[0], row[1]))
    return pairs


def collate_pairs(
    batch: Sequence[tuple[torch.Tensor, torch.Tensor]],
    source_pad_id: int,
    target_pad_id: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pad source and target sequences to their batch-specific maximum lengths."""
    source_sequences, target_sequences = zip(*batch)
    padded_sources = pad_sequence(source_sequences, batch_first=True, padding_value=source_pad_id)
    padded_targets = pad_sequence(target_sequences, batch_first=True, padding_value=target_pad_id)
    return padded_sources, padded_targets


def create_dataloader(
    dataset: TransliterationDataset,
    *,
    shuffle: bool = False,
) -> DataLoader[tuple[torch.Tensor, torch.Tensor]]:
    """Create a DataLoader using the configured batch size and vocabulary padding IDs."""
    return DataLoader(
        dataset,
        batch_size=CONFIG.batch_size,
        shuffle=shuffle,
        collate_fn=lambda batch: collate_pairs(
            batch,
            dataset.source_vocab.pad_id,
            dataset.target_vocab.pad_id,
        ),
    )


def verify_training_dataloader() -> None:
    """Build and display one padded batch using training data only."""
    from .vocabulary import build_vocabularies

    training_pairs = read_pairs(CONFIG.train_path)
    source_vocab, target_vocab = build_vocabularies(
        [pair.source for pair in training_pairs],
        [pair.target for pair in training_pairs],
    )
    selected_pairs = list(
        dict.fromkeys(
            (
                min(training_pairs, key=lambda pair: len(pair.source)),
                max(training_pairs, key=lambda pair: len(pair.source)),
                min(training_pairs, key=lambda pair: len(pair.target)),
                max(training_pairs, key=lambda pair: len(pair.target)),
            )
        )
    )
    dataset = TransliterationDataset(selected_pairs, source_vocab, target_vocab)
    sources, targets = next(iter(create_dataloader(dataset)))

    print(f"Source tensor shape: {tuple(sources.shape)}")
    print(f"Target tensor shape: {tuple(targets.shape)}")
    print(f"First source sequence: {sources[0].tolist()}")
    print(f"First target sequence: {targets[0].tolist()}")
    print(f"Source PAD ID: {source_vocab.pad_id}")
    print(f"Target PAD ID: {target_vocab.pad_id}")
    print(f"Source padding present: {bool((sources == source_vocab.pad_id).any())}")
    print(f"Target padding present: {bool((targets == target_vocab.pad_id).any())}")


def _character_set(values: Iterable[str]) -> list[str]:
    """Return the sorted unique characters found in a collection of strings."""
    return sorted({character for value in values for character in value})


def summarize_split(name: str, path: Path, pairs: list[WordPair]) -> None:
    """Print the requested inspection metrics for one split."""
    source_lengths = [len(pair.source) for pair in pairs]
    target_lengths = [len(pair.target) for pair in pairs]
    duplicate_count = len(pairs) - len({(pair.source, pair.target) for pair in pairs})
    missing_count = sum(not pair.source or not pair.target for pair in pairs)

    def length_summary(lengths: list[int]) -> str:
        if not lengths:
            return "no rows"
        return f"min={min(lengths)}, max={max(lengths)}, average={sum(lengths) / len(lengths):.2f}"

    print(f"\n{name} ({path.name})")
    print(f"  rows: {len(pairs)}")
    print(f"  missing/empty source or target values: {missing_count}")
    print(f"  duplicate (source, target) pairs: {duplicate_count}")
    print(f"  source lengths: {length_summary(source_lengths)}")
    print(f"  target lengths: {length_summary(target_lengths)}")
    source_characters = _character_set(pair.source for pair in pairs)
    target_characters = _character_set(pair.target for pair in pairs)
    print(f"  unique source characters: {len(source_characters)}")
    print(f"  unique target characters: {len(target_characters)}")
    print(f"  source character set: {source_characters}")
    print(f"  target character set: {target_characters}")
    print("  examples:")
    for pair in pairs[:3]:
        print(f"    {pair.source} -> {pair.target}")


def inspect_dataset() -> None:
    """Inspect all configured splits and report their actual structure."""
    split_paths = {
        "train": CONFIG.train_path,
        "validation": CONFIG.valid_path,
        "test": CONFIG.test_path,
    }
    for name, path in split_paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"Dataset file not found: {path}")
        summarize_split(name, path, read_pairs(path))


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    inspect_dataset()
    verify_training_dataloader()
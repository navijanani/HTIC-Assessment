"""Utilities for checking the raw Aksharantar Hindi splits."""

from __future__ import annotations

import csv
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from .config import CONFIG


@dataclass(frozen=True)
class WordPair:
    """One Romanized input and its Devanagari target."""

    source: str
    target: str


def read_pairs(path: Path) -> list[WordPair]:
    """Read a headerless two-column CSV without changing the source file."""
    pairs: list[WordPair] = []
    with path.open("r", encoding="utf-8", newline="") as csv_file:
        for row_number, row in enumerate(csv.reader(csv_file), start=1):
            if len(row) != 2:
                raise ValueError(f"{path} row {row_number} has {len(row)} columns; expected 2")
            pairs.append(WordPair(row[0], row[1]))
    return pairs


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
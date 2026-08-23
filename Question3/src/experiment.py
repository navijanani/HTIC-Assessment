"""Compare RNN, GRU, and LSTM configurations on the same data."""

from __future__ import annotations

import csv
import random
import time
from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import torch
from torch import nn
from torch.optim import Adam

from . import decoder as decoder_module
from . import encoder as encoder_module
from . import seq2seq as seq2seq_module
from .config import CONFIG
from .dataset import TransliterationDataset, create_dataloader, read_pairs
from .evaluate import _character_accuracy, _generate_batch
from .seq2seq import Seq2Seq
from .vocabulary import build_vocabularies


Batch = tuple[torch.Tensor, torch.Tensor]
CELL_TYPES = ("RNN", "GRU", "LSTM")


def _set_cell_config(cell_type: str) -> None:
    """Apply one recurrent-cell setting to all existing model modules."""
    cell_config = replace(CONFIG, recurrent_cell=cell_type)
    encoder_module.CONFIG = cell_config
    decoder_module.CONFIG = cell_config
    seq2seq_module.CONFIG = cell_config


def _seed_everything() -> None:
    """Reset random sources before each comparable experiment."""
    random.seed(CONFIG.random_seed)
    torch.manual_seed(CONFIG.random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(CONFIG.random_seed)


def _build_loaders() -> tuple[torch.utils.data.DataLoader[Batch], torch.utils.data.DataLoader[Batch], torch.utils.data.DataLoader[Batch], object, object]:
    """Build all loaders with vocabularies created from training data only."""
    training_pairs = read_pairs(CONFIG.train_path)
    validation_pairs = read_pairs(CONFIG.valid_path)
    test_pairs = read_pairs(CONFIG.test_path)
    source_vocab, target_vocab = build_vocabularies(
        [pair.source for pair in training_pairs],
        [pair.target for pair in training_pairs],
    )
    train_dataset = TransliterationDataset(training_pairs, source_vocab, target_vocab)
    validation_dataset = TransliterationDataset(validation_pairs, source_vocab, target_vocab)
    test_dataset = TransliterationDataset(test_pairs, source_vocab, target_vocab)
    return (
        create_dataloader(train_dataset, shuffle=True),
        create_dataloader(validation_dataset),
        create_dataloader(test_dataset),
        source_vocab,
        target_vocab,
    )


def _loss_for_loader(
    model: Seq2Seq,
    loader: torch.utils.data.DataLoader[Batch],
    loss_function: nn.Module,
    device: torch.device,
    training: bool,
    optimizer: Adam | None = None,
) -> float:
    """Run one training or validation pass and return its mean loss."""
    model.train(training)
    total_loss = 0.0
    for source_ids, target_ids in loader:
        source_ids = source_ids.to(device)
        target_ids = target_ids.to(device)
        if training and optimizer is not None:
            optimizer.zero_grad()
        logits = model(
            source_ids,
            target_ids,
            teacher_forcing_ratio=CONFIG.teacher_forcing_ratio if training else 0.0,
        )
        loss = loss_function(
            logits.reshape(-1, logits.shape[-1]),
            target_ids[:, 1:].reshape(-1),
        )
        if training and optimizer is not None:
            loss.backward()
            optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


def _test_accuracy(
    model: Seq2Seq,
    test_loader: torch.utils.data.DataLoader[Batch],
    target_vocab: object,
    max_length: int,
    device: torch.device,
) -> float:
    """Generate test predictions and return positional character accuracy."""
    model.eval()
    correct = 0
    compared = 0
    pair_index = 0
    with torch.no_grad():
        for source_ids, _ in test_loader:
            source_ids = source_ids.to(device)
            predictions = _generate_batch(model, source_ids, target_vocab, max_length)
            for prediction in predictions:
                target = test_loader.dataset.pairs[pair_index].target
                matches, length = _character_accuracy(target, prediction)
                correct += matches
                compared += length
                pair_index += 1
    return correct / compared if compared else 0.0


def run_experiment() -> None:
    """Train and compare all three recurrent-cell configurations."""
    original_encoder_config = encoder_module.CONFIG
    original_decoder_config = decoder_module.CONFIG
    original_seq2seq_config = seq2seq_module.CONFIG
    results: list[dict[str, object]] = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    try:
        for cell_type in CELL_TYPES:
            _set_cell_config(cell_type)
            _seed_everything()
            train_loader, validation_loader, test_loader, source_vocab, target_vocab = _build_loaders()
            model = Seq2Seq(len(source_vocab), len(target_vocab), target_vocab.sos_id).to(device)
            loss_function = nn.CrossEntropyLoss(ignore_index=target_vocab.pad_id)
            optimizer = Adam(model.parameters(), lr=CONFIG.learning_rate)
            best_validation_loss = float("inf")
            best_state = None
            started_at = time.perf_counter()

            for epoch in range(CONFIG.epochs):
                _loss_for_loader(model, train_loader, loss_function, device, True, optimizer)
                with torch.no_grad():
                    validation_loss = _loss_for_loader(
                        model, validation_loader, loss_function, device, False
                    )
                if validation_loss < best_validation_loss:
                    best_validation_loss = validation_loss
                    best_state = deepcopy(model.state_dict())
                print(
                    f"{cell_type} epoch {epoch + 1}/{CONFIG.epochs} - "
                    f"validation loss: {validation_loss:.4f}"
                )

            if best_state is None:
                raise RuntimeError(f"No checkpoint state was created for {cell_type}")
            model.load_state_dict(best_state)
            max_length = max(len(pair.target) for pair in train_loader.dataset.pairs) + 1
            test_accuracy = _test_accuracy(model, test_loader, target_vocab, max_length, device)
            training_time = time.perf_counter() - started_at
            parameter_count = sum(parameter.numel() for parameter in model.parameters())
            results.append(
                {
                    "cell_type": cell_type,
                    "parameter_count": parameter_count,
                    "best_validation_loss": best_validation_loss,
                    "test_character_accuracy": test_accuracy,
                    "training_time_seconds": training_time,
                }
            )
            print(
                f"{cell_type}: parameters={parameter_count}, "
                f"test accuracy={test_accuracy:.4f}, time={training_time:.2f}s"
            )
    finally:
        encoder_module.CONFIG = original_encoder_config
        decoder_module.CONFIG = original_decoder_config
        seq2seq_module.CONFIG = original_seq2seq_config

    output_path = Path(__file__).resolve().parents[1] / "results" / "experiments.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as output_file:
        fieldnames = list(results[0])
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    print(f"Experiment results saved to: {output_path}")


if __name__ == "__main__":
    run_experiment()

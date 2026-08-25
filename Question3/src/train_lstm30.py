"""Training entry point for the transliteration Seq2Seq model."""

from __future__ import annotations

import random
from dataclasses import replace
from pathlib import Path

import torch
from torch import nn
from torch.optim import Adam

from .config import CONFIG
from .dataset import TransliterationDataset, create_dataloader, read_pairs
from .seq2seq import Seq2Seq
from .vocabulary import CharacterVocabulary, build_vocabularies


Batch = tuple[torch.Tensor, torch.Tensor]


def _build_data() -> tuple[torch.utils.data.DataLoader[Batch], torch.utils.data.DataLoader[Batch], CharacterVocabulary, CharacterVocabulary]:
    """Build train and validation loaders using training-only vocabularies."""
    training_pairs = read_pairs(CONFIG.train_path)
    validation_pairs = read_pairs(CONFIG.valid_path)
    source_vocab, target_vocab = build_vocabularies(
        [pair.source for pair in training_pairs],
        [pair.target for pair in training_pairs],
    )
    training_dataset = TransliterationDataset(training_pairs, source_vocab, target_vocab)
    validation_dataset = TransliterationDataset(validation_pairs, source_vocab, target_vocab)
    return (
        create_dataloader(training_dataset, shuffle=True),
        create_dataloader(validation_dataset),
        source_vocab,
        target_vocab,
    )


def _run_epoch(
    model: Seq2Seq,
    loader: torch.utils.data.DataLoader[Batch],
    loss_function: nn.Module,
    device: torch.device,
    optimizer: Adam | None = None,
) -> float:
    """Run one training or validation pass and return average loss."""
    is_training = optimizer is not None
    model.train(is_training)
    total_loss = 0.0

    for source_ids, target_ids in loader:
        source_ids = source_ids.to(device)
        target_ids = target_ids.to(device)
        if is_training:
            optimizer.zero_grad()

        logits = model(
            source_ids,
            target_ids,
            teacher_forcing_ratio=CONFIG.teacher_forcing_ratio if is_training else 0.0,
        )
        loss = loss_function(logits.reshape(-1, logits.shape[-1]), target_ids[:, 1:].reshape(-1))

        if is_training:
            loss.backward()
            optimizer.step()
        total_loss += loss.item()

    return total_loss / len(loader)


def train() -> None:
    """Train the model and save the checkpoint with the best validation loss."""
    random.seed(CONFIG.random_seed)
    torch.manual_seed(CONFIG.random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(CONFIG.random_seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, validation_loader, source_vocab, target_vocab = _build_data()
    model = Seq2Seq(len(source_vocab), len(target_vocab), target_vocab.sos_id).to(device)
    loss_function = nn.CrossEntropyLoss(ignore_index=target_vocab.pad_id)
    optimizer = Adam(model.parameters(), lr=CONFIG.learning_rate)

    checkpoint_path = Path(__file__).resolve().parents[1] / "checkpoints" / "best_model.pt"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    best_validation_loss = float("inf")

    for epoch in range(1, CONFIG.epochs + 1):
        training_loss = _run_epoch(model, train_loader, loss_function, device, optimizer)
        with torch.no_grad():
            validation_loss = _run_epoch(model, validation_loader, loss_function, device)

        print(
            f"Epoch {epoch}/{CONFIG.epochs} - "
            f"train loss: {training_loss:.4f} - "
            f"validation loss: {validation_loss:.4f}"
        )

        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "source_vocab": source_vocab.token_to_id,
                    "target_vocab": target_vocab.token_to_id,
                    "validation_loss": validation_loss,
                    "epoch": epoch,
                },
                checkpoint_path,
            )
            print(f"  Saved best checkpoint: {checkpoint_path}")


def run_training_smoke_test() -> None:
    """Run one temporary epoch and confirm that a checkpoint was written."""
    original_config = CONFIG
    checkpoint_path = Path(__file__).resolve().parents[1] / "checkpoints" / "best_model.pt"
    try:
        globals()["CONFIG"] = replace(original_config, epochs=1)
        train()
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"Smoke test checkpoint was not created: {checkpoint_path}")
        print(f"Smoke test checkpoint confirmed: {checkpoint_path}")
    finally:
        globals()["CONFIG"] = original_config


if __name__ == "__main__":
    train()

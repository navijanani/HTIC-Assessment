"""Autoregressive evaluation on the held-out Hindi test split."""

from __future__ import annotations

import csv
from itertools import zip_longest
from pathlib import Path

import torch

from .config import CONFIG
from .dataset import TransliterationDataset, create_dataloader, read_pairs
from .seq2seq import Seq2Seq
from .vocabulary import CharacterVocabulary, build_vocabularies


Batch = tuple[torch.Tensor, torch.Tensor]


def _build_test_data() -> tuple[torch.utils.data.DataLoader[Batch], CharacterVocabulary, CharacterVocabulary]:
    """Build the test loader with vocabularies created from training text only."""
    training_pairs = read_pairs(CONFIG.train_path)
    test_pairs = read_pairs(CONFIG.test_path)
    source_vocab, target_vocab = build_vocabularies(
        [pair.source for pair in training_pairs],
        [pair.target for pair in training_pairs],
    )
    test_dataset = TransliterationDataset(test_pairs, source_vocab, target_vocab)
    return create_dataloader(test_dataset), source_vocab, target_vocab


def _generate_batch(
    model: Seq2Seq,
    source_ids: torch.Tensor,
    target_vocab: CharacterVocabulary,
    max_length: int,
) -> list[str]:
    """Generate one decoded string per source item without teacher forcing."""
    _, state = model.encoder(source_ids)
    decoder_input = torch.full(
        (source_ids.shape[0],),
        target_vocab.sos_id,
        dtype=torch.long,
        device=source_ids.device,
    )
    generated_ids: list[list[int]] = [[] for _ in range(source_ids.shape[0])]
    finished = torch.zeros(source_ids.shape[0], dtype=torch.bool, device=source_ids.device)

    for _ in range(max_length):
        logits, state = model.decoder(decoder_input, state)
        next_ids = logits.argmax(dim=1)
        for index, token_id in enumerate(next_ids.tolist()):
            if not finished[index]:
                if token_id == target_vocab.eos_id:
                    finished[index] = True
                else:
                    generated_ids[index].append(token_id)
        decoder_input = next_ids
        if bool(finished.all()):
            break

    return [target_vocab.decode(ids) for ids in generated_ids]


def _character_accuracy(target: str, prediction: str) -> tuple[int, int]:
    """Return positional character matches and comparison length."""
    missing_marker = object()
    matches = sum(
        target_character == predicted_character
        for target_character, predicted_character in zip_longest(
            target, prediction, fillvalue=missing_marker
        )
    )
    return matches, max(len(target), len(prediction))


def evaluate() -> None:
    """Load the best model, evaluate the test split, and save predictions."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_loader, source_vocab, target_vocab = _build_test_data()
    model = Seq2Seq(len(source_vocab), len(target_vocab), target_vocab.sos_id).to(device)

    checkpoint_path = Path(__file__).resolve().parents[1] / "checkpoints" / "best_model.pt"
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    training_pairs = read_pairs(CONFIG.train_path)
    max_length = max(len(pair.target) for pair in training_pairs) + 1
    predictions: list[tuple[str, str, str]] = []
    correct_characters = 0
    compared_characters = 0

    with torch.no_grad():
        for source_ids, target_ids in test_loader:
            source_ids = source_ids.to(device)
            predicted_targets = _generate_batch(model, source_ids, target_vocab, max_length)
            for row_index, prediction in enumerate(predicted_targets):
                source_text = test_loader.dataset.pairs[len(predictions)].source
                target_text = test_loader.dataset.pairs[len(predictions)].target
                predictions.append((source_text, target_text, prediction))
                matches, length = _character_accuracy(target_text, prediction)
                correct_characters += matches
                compared_characters += length

    accuracy = correct_characters / compared_characters if compared_characters else 0.0
    exact_matches = sum(target == prediction for _, target, prediction in predictions)
    exact_accuracy = exact_matches / len(predictions) if predictions else 0.0
    print(f"Test examples: {len(predictions)}")
    print(f"Character-level accuracy: {accuracy:.4f}")
    print(f"Exact word accuracy: {exact_accuracy:.4f} ({exact_matches}/{len(predictions)})")
    print("Examples:")
    for source_text, target_text, prediction in predictions[:3]:
        print(f"  {source_text} -> target: {target_text} -> prediction: {prediction}")

    output_path = Path(__file__).resolve().parents[1] / "results" / "predictions" / "test_predictions.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.writer(output_file)
        writer.writerow(["source", "target", "prediction"])
        writer.writerows(predictions)
    print(f"Predictions saved to: {output_path}")


if __name__ == "__main__":
    evaluate()

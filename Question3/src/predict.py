"""Interactive Romanized Hindi to Devanagari prediction."""

from __future__ import annotations

from pathlib import Path

import torch

from .config import CONFIG
from .dataset import read_pairs
from .evaluate import _generate_batch
from .seq2seq import Seq2Seq
from .vocabulary import build_vocabularies


def load_prediction_model() -> tuple[Seq2Seq, object, object, int, torch.device]:
    """Load the trained model and vocabularies built from training data only."""
    training_pairs = read_pairs(CONFIG.train_path)
    source_vocab, target_vocab = build_vocabularies(
        [pair.source for pair in training_pairs],
        [pair.target for pair in training_pairs],
    )
    model = Seq2Seq(len(source_vocab), len(target_vocab), target_vocab.sos_id)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_path = Path(__file__).resolve().parents[1] / "checkpoints" / "best_model.pt"
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    max_length = max(len(pair.target) for pair in training_pairs) + 1
    return model, source_vocab, target_vocab, max_length, device


def predict_interactively() -> None:
    """Read Romanized words and print their autoregressive predictions."""
    model, source_vocab, target_vocab, max_length, device = load_prediction_model()
    print("Enter a Romanized Hindi word, or type 'quit' to stop.")

    while True:
        text = input("Romanized input: ").strip()
        if text.lower() in {"quit", "exit"}:
            print("Goodbye.")
            break
        if not text:
            continue

        source_ids = torch.tensor(
            [source_vocab.encode(text)],
            dtype=torch.long,
            device=device,
        )
        with torch.no_grad():
            prediction = _generate_batch(model, source_ids, target_vocab, max_length)[0]
        print(f"Input: {text}")
        print(f"Prediction: {prediction}")


if __name__ == "__main__":
    predict_interactively()

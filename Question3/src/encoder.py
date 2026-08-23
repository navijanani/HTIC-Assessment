"""Character-level encoder for Romanized Hindi input sequences."""

from __future__ import annotations

from dataclasses import replace

import torch
from torch import nn

from .config import CONFIG


class Encoder(nn.Module):
    """Encode padded source character IDs with a configurable recurrent cell."""

    def __init__(self, source_vocab_size: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(source_vocab_size, CONFIG.embedding_dim)
        recurrent_cells = {
            "RNN": nn.RNN,
            "GRU": nn.GRU,
            "LSTM": nn.LSTM,
        }
        cell_type = CONFIG.recurrent_cell.upper()
        if cell_type not in recurrent_cells:
            raise ValueError("CONFIG.recurrent_cell must be RNN, GRU, or LSTM")

        self.recurrent = recurrent_cells[cell_type](
            input_size=CONFIG.embedding_dim,
            hidden_size=CONFIG.hidden_dim,
            num_layers=CONFIG.encoder_layers,
            batch_first=True,
        )

    def forward(
        self, source_ids: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor | tuple[torch.Tensor, torch.Tensor]]:
        """Return one output per input position and the final recurrent state."""
        embedded_source = self.embedding(source_ids)
        outputs, hidden_state = self.recurrent(embedded_source)
        return outputs, hidden_state


def verify_encoder() -> None:
    """Run one training source batch through each supported recurrent cell."""
    from .dataset import TransliterationDataset, create_dataloader, read_pairs
    from .vocabulary import build_vocabularies

    training_pairs = read_pairs(CONFIG.train_path)
    source_vocab, target_vocab = build_vocabularies(
        [pair.source for pair in training_pairs],
        [pair.target for pair in training_pairs],
    )
    training_dataset = TransliterationDataset(training_pairs, source_vocab, target_vocab)
    source_batch, _ = next(iter(create_dataloader(training_dataset)))

    original_config = CONFIG
    try:
        for cell_type in ("RNN", "GRU", "LSTM"):
            globals()["CONFIG"] = replace(original_config, recurrent_cell=cell_type)
            encoder = Encoder(len(source_vocab))
            with torch.no_grad():
                outputs, hidden_state = encoder(source_batch)

            print(f"{cell_type} input shape: {tuple(source_batch.shape)}")
            print(f"{cell_type} output shape: {tuple(outputs.shape)}")
            if isinstance(hidden_state, tuple):
                print(
                    f"{cell_type} hidden-state shapes: "
                    f"{tuple(hidden_state[0].shape)}, {tuple(hidden_state[1].shape)}"
                )
            else:
                print(f"{cell_type} hidden-state shape: {tuple(hidden_state.shape)}")
    finally:
        globals()["CONFIG"] = original_config


if __name__ == "__main__":
    verify_encoder()

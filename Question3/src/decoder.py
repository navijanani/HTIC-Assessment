"""Character-level decoder for target transliteration sequences."""

from __future__ import annotations

from dataclasses import replace

import torch
from torch import nn

from .config import CONFIG


RecurrentState = torch.Tensor | tuple[torch.Tensor, torch.Tensor]


class Decoder(nn.Module):
    """Decode one target character at a time with a configurable recurrent cell."""

    def __init__(self, target_vocab_size: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(target_vocab_size, CONFIG.embedding_dim)

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
            num_layers=CONFIG.decoder_layers,
            batch_first=True,
        )
        self.output_layer = nn.Linear(CONFIG.hidden_dim, target_vocab_size)

    def forward(
        self,
        target_ids: torch.Tensor,
        state: RecurrentState,
    ) -> tuple[torch.Tensor, RecurrentState]:
        """Process one target character and return logits with the new state."""
        if target_ids.ndim == 0:
            target_ids = target_ids.unsqueeze(0)
        elif target_ids.ndim == 2 and target_ids.shape[1] == 1:
            target_ids = target_ids.squeeze(1)
        elif target_ids.ndim != 1:
            raise ValueError("target_ids must contain one ID per batch item")

        embedded_target = self.embedding(target_ids).unsqueeze(1)
        recurrent_output, new_state = self.recurrent(
            embedded_target,
            state,
        )
        logits = self.output_layer(recurrent_output[:, 0, :])
        return logits, new_state


def verify_decoder() -> None:
    """Run one training batch through matching encoder and decoder cells."""
    from . import encoder as encoder_module
    from .dataset import TransliterationDataset, create_dataloader, read_pairs
    from .vocabulary import build_vocabularies

    training_pairs = read_pairs(CONFIG.train_path)
    source_vocab, target_vocab = build_vocabularies(
        [pair.source for pair in training_pairs],
        [pair.target for pair in training_pairs],
    )
    training_dataset = TransliterationDataset(training_pairs, source_vocab, target_vocab)
    source_batch, target_batch = next(iter(create_dataloader(training_dataset)))

    original_decoder_config = CONFIG
    original_encoder_config = encoder_module.CONFIG
    try:
        for cell_type in ("RNN", "GRU", "LSTM"):
            test_config = replace(original_decoder_config, recurrent_cell=cell_type)
            globals()["CONFIG"] = test_config
            encoder_module.CONFIG = replace(original_encoder_config, recurrent_cell=cell_type)

            encoder = encoder_module.Encoder(len(source_vocab))
            decoder = Decoder(len(target_vocab))
            with torch.no_grad():
                _, encoder_state = encoder(source_batch)
                logits, decoder_state = decoder(target_batch[:, 0], encoder_state)

            print(f"{cell_type} target input shape: {tuple(target_batch[:, 0].shape)}")
            print(f"{cell_type} decoder logits shape: {tuple(logits.shape)}")
            if isinstance(decoder_state, tuple):
                print(f"{cell_type} hidden-state shape: {tuple(decoder_state[0].shape)}")
                print(f"{cell_type} cell-state shape: {tuple(decoder_state[1].shape)}")
            else:
                print(f"{cell_type} hidden-state shape: {tuple(decoder_state.shape)}")
    finally:
        globals()["CONFIG"] = original_decoder_config
        encoder_module.CONFIG = original_encoder_config


if __name__ == "__main__":
    verify_decoder()

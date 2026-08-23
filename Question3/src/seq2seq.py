"""Sequence-to-sequence model connecting the project encoder and decoder."""

from __future__ import annotations

import torch
from torch import nn

from .config import CONFIG
from .decoder import Decoder, RecurrentState
from .encoder import Encoder


class Seq2Seq(nn.Module):
    """Generate a target sequence from a padded source sequence."""

    def __init__(
        self,
        source_vocab_size: int,
        target_vocab_size: int,
        target_sos_id: int,
    ) -> None:
        super().__init__()
        self.encoder = Encoder(source_vocab_size)
        self.decoder = Decoder(target_vocab_size)
        self.target_sos_id = target_sos_id
        self.target_vocab_size = target_vocab_size

    def forward(
        self,
        source_ids: torch.Tensor,
        target_ids: torch.Tensor,
        teacher_forcing_ratio: float | None = None,
    ) -> torch.Tensor:
        """Return decoder logits for each target step after the initial SOS."""
        if source_ids.ndim != 2 or target_ids.ndim != 2:
            raise ValueError("source_ids and target_ids must be two-dimensional batches")
        if target_ids.shape[1] < 2:
            raise ValueError("target_ids must contain SOS and at least one target character")

        ratio = CONFIG.teacher_forcing_ratio if teacher_forcing_ratio is None else teacher_forcing_ratio
        if not 0.0 <= ratio <= 1.0:
            raise ValueError("teacher_forcing_ratio must be between 0 and 1")

        _, decoder_state = self.encoder(source_ids)
        batch_size, target_length = target_ids.shape
        decoder_logits = torch.empty(
            batch_size,
            target_length - 1,
            self.target_vocab_size,
            device=source_ids.device,
        )
        decoder_input = torch.full(
            (batch_size,),
            self.target_sos_id,
            dtype=torch.long,
            device=source_ids.device,
        )

        for step in range(target_length - 1):
            step_logits, decoder_state = self.decoder(decoder_input, decoder_state)
            decoder_logits[:, step, :] = step_logits

            predicted_ids = step_logits.argmax(dim=1)
            use_teacher = torch.rand((), device=source_ids.device).item() < ratio
            decoder_input = target_ids[:, step + 1] if use_teacher else predicted_ids

        return decoder_logits


def verify_seq2seq() -> None:
    """Check Seq2Seq output shapes with each cell and teacher-forcing extreme."""
    from dataclasses import replace

    from . import decoder as decoder_module
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
    expected_shape = (target_batch.shape[0], target_batch.shape[1] - 1, len(target_vocab))

    original_config = CONFIG
    original_encoder_config = encoder_module.CONFIG
    original_decoder_config = decoder_module.CONFIG
    try:
        for cell_type in ("RNN", "GRU", "LSTM"):
            test_config = replace(original_config, recurrent_cell=cell_type)
            globals()["CONFIG"] = test_config
            encoder_module.CONFIG = test_config
            decoder_module.CONFIG = test_config

            model = Seq2Seq(len(source_vocab), len(target_vocab), target_vocab.sos_id)
            for ratio in (0.0, 1.0):
                with torch.no_grad():
                    logits = model(source_batch, target_batch, teacher_forcing_ratio=ratio)
                assert logits.shape == expected_shape
                print(f"{cell_type}, teacher_forcing_ratio={ratio}")
                print(f"  source shape: {tuple(source_batch.shape)}")
                print(f"  target shape: {tuple(target_batch.shape)}")
                print(f"  output logits shape: {tuple(logits.shape)}")
    finally:
        globals()["CONFIG"] = original_config
        encoder_module.CONFIG = original_encoder_config
        decoder_module.CONFIG = original_decoder_config


if __name__ == "__main__":
    verify_seq2seq()
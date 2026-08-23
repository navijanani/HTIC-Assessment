"""Operation-count estimates for the complete transliteration network."""

from __future__ import annotations

from .config import CONFIG
from .dataset import read_pairs
from .vocabulary import build_vocabularies


CELL_GATE_COUNTS = {
    "RNN": 1,
    "GRU": 3,
    "LSTM": 4,
}


def recurrent_mac_count(cell_type: str, e: int, h: int, t: int, v: int) -> int:
    """Estimate complete-network multiply-accumulate operations for one sequence."""
    gates = CELL_GATE_COUNTS[cell_type]
    recurrent_macs = 2 * t * gates * (e * h + h * h)
    output_projection_macs = t * h * v
    return recurrent_macs + output_projection_macs


def theoretical_parameter_count(gate_count: int, v: int, e: int, h: int) -> int:
    """Count parameters with the assignment's single vocabulary size."""
    return v * e + 2 * gate_count * (e * h + h * h + h) + h * v + v


def verify_model_parameters(source_vocab_size: int = 30, target_vocab_size: int = 68) -> None:
    """Compare theoretical counts with parameter totals from actual Seq2Seq models."""
    from dataclasses import replace

    from . import decoder as decoder_module
    from . import encoder as encoder_module
    from . import seq2seq as seq2seq_module

    original_config = CONFIG
    original_encoder_config = encoder_module.CONFIG
    original_decoder_config = decoder_module.CONFIG
    original_seq2seq_config = seq2seq_module.CONFIG
    try:
        for cell_type in CELL_GATE_COUNTS:
            test_config = replace(original_config, recurrent_cell=cell_type)
            globals()["CONFIG"] = test_config
            encoder_module.CONFIG = test_config
            decoder_module.CONFIG = test_config
            seq2seq_module.CONFIG = test_config

            model = seq2seq_module.Seq2Seq(
                source_vocab_size,
                target_vocab_size,
                target_sos_id=1,
            )
            actual_count = sum(parameter.numel() for parameter in model.parameters())
            theoretical_count = theoretical_parameter_count(
                CELL_GATE_COUNTS[cell_type],
                target_vocab_size,
                test_config.embedding_dim,
                test_config.hidden_dim,
            )
            print(f"{cell_type}: theoretical={theoretical_count:,}, actual PyTorch={actual_count:,} parameters")
    finally:
        globals()["CONFIG"] = original_config
        encoder_module.CONFIG = original_encoder_config
        decoder_module.CONFIG = original_decoder_config
        seq2seq_module.CONFIG = original_seq2seq_config


def analyze_complexity() -> None:
    """Print formulas and values using the project configuration and train data."""
    training_pairs = read_pairs(CONFIG.train_path)
    _, target_vocab = build_vocabularies(
        [pair.source for pair in training_pairs],
        [pair.target for pair in training_pairs],
    )

    e = CONFIG.embedding_dim
    h = CONFIG.hidden_dim
    t = max(len(pair.target) for pair in training_pairs) + 1
    v = len(target_vocab)

    print("Counting convention: multiply-accumulate operations (MACs) only.")
    print("Excluded: embedding lookups, biases, activations, padding, and control flow.")
    print("Included: encoder and decoder recurrent computations, plus one H x V decoder output projection per step.")
    print(f"Variables: E={e}, H={h}, T={t}, V={v}")
    print("Assignment formula: C = 2*T*G*(E*H + H*H) + T*H*V")
    print("Here, G = 1 for RNN, 3 for GRU, and 4 for LSTM.")

    for cell_type, gate_count in CELL_GATE_COUNTS.items():
        value = recurrent_mac_count(cell_type, e, h, t, v)
        formula = f"2*{t}*{gate_count}*({e}*{h} + {h}*{h}) + {t}*{h}*{v}"
        print(f"{cell_type}: {formula} = {value:,} MACs")

    source_vocab_size = 30
    target_vocab_size = 68
    print("\nParameter-count analysis")
    print("Parameter convention: trainable weights and biases only.")
    print("The source embedding uses source vocabulary size 30; the decoder output layer uses target vocabulary size 68.")
    print("Assignment formula: P = V*E + 2*G*(E*H + H*H + H) + H*V + V")
    print(f"Assignment values: V={target_vocab_size}, E={e}, H={h}")
    print("Actual project uses separate source and target embeddings: Vs*E + Vt*E + 2*G*(E*H + H*H + H) + H*Vt + Vt")
    print(f"Actual project values: Vs={source_vocab_size}, Vt={target_vocab_size}, E={e}, H={h}")

    for cell_type, gate_count in CELL_GATE_COUNTS.items():
        value = theoretical_parameter_count(gate_count, target_vocab_size, e, h)
        formula = (
            f"{target_vocab_size}*{e} + 2*{gate_count}*({e}*{h} + {h}*{h} + {h}) "
            f"+ {h}*{target_vocab_size} + {target_vocab_size}"
        )
        print(f"{cell_type}: {formula} = {value:,} parameters")

    print("\nActual PyTorch model parameter verification")
    verify_model_parameters(source_vocab_size, target_vocab_size)


if __name__ == "__main__":
    analyze_complexity()

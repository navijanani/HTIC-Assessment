"""Central configuration for the transliteration experiments."""

from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"


@dataclass(frozen=True)
class Config:
    """Settings shared by later data and model stages."""

    random_seed: int = 42
    train_path: Path = RAW_DATA_DIR / "hin_train.csv"
    valid_path: Path = RAW_DATA_DIR / "hin_valid.csv"
    test_path: Path = RAW_DATA_DIR / "hin_test.csv"
    embedding_dim: int = 128
    hidden_dim: int = 256
    encoder_layers: int = 1
    decoder_layers: int = 1
    recurrent_cell: str = "GRU"
    batch_size: int = 64
    learning_rate: float = 0.001
    epochs: int = 20
    teacher_forcing_ratio: float = 0.5


CONFIG = Config()
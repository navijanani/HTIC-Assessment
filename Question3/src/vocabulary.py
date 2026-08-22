"""Character-level vocabularies for the transliteration task."""

from __future__ import annotations

from dataclasses import dataclass

from .config import CONFIG
from .dataset import read_pairs


PAD_TOKEN = "<PAD>"
SOS_TOKEN = "<SOS>"
EOS_TOKEN = "<EOS>"
UNK_TOKEN = "<UNK>"
SPECIAL_TOKENS = (PAD_TOKEN, SOS_TOKEN, EOS_TOKEN, UNK_TOKEN)


@dataclass
class CharacterVocabulary:
    """Map individual characters and special tokens to integer IDs."""

    token_to_id: dict[str, int]

    @classmethod
    def from_texts(cls, texts: list[str]) -> "CharacterVocabulary":
        tokens = list(SPECIAL_TOKENS)
        tokens.extend(sorted({character for text in texts for character in text}))
        return cls({token: index for index, token in enumerate(tokens)})

    @property
    def id_to_token(self) -> dict[int, str]:
        return {index: token for token, index in self.token_to_id.items()}

    @property
    def pad_id(self) -> int:
        return self.token_to_id[PAD_TOKEN]

    @property
    def sos_id(self) -> int:
        return self.token_to_id[SOS_TOKEN]

    @property
    def eos_id(self) -> int:
        return self.token_to_id[EOS_TOKEN]

    @property
    def unk_id(self) -> int:
        return self.token_to_id[UNK_TOKEN]

    def __len__(self) -> int:
        return len(self.token_to_id)

    def encode(self, text: str, *, add_boundaries: bool = False) -> list[int]:
        """Convert text to IDs, optionally wrapping it with SOS and EOS."""
        encoded = [self.token_to_id.get(character, self.unk_id) for character in text]
        if add_boundaries:
            return [self.sos_id, *encoded, self.eos_id]
        return encoded

    def decode(self, ids: list[int], *, remove_special_tokens: bool = True) -> str:
        """Convert IDs to text, ignoring special tokens by default."""
        tokens = [self.id_to_token.get(identifier, UNK_TOKEN) for identifier in ids]
        if remove_special_tokens:
            boundary_tokens = {PAD_TOKEN, SOS_TOKEN, EOS_TOKEN}
            tokens = [token for token in tokens if token not in boundary_tokens]
        return "".join(tokens)


def build_vocabularies(source_texts: list[str], target_texts: list[str]) -> tuple[CharacterVocabulary, CharacterVocabulary]:
    """Build independent source and target vocabularies from training text only."""
    return CharacterVocabulary.from_texts(source_texts), CharacterVocabulary.from_texts(target_texts)


def verify_training_vocabularies() -> None:
    """Print small checks for vocabularies built only from the training split."""
    training_pairs = read_pairs(CONFIG.train_path)
    source_vocab, target_vocab = build_vocabularies(
        [pair.source for pair in training_pairs],
        [pair.target for pair in training_pairs],
    )
    example_pair = training_pairs[0]

    print(f"Source vocabulary size: {len(source_vocab)}")
    print(f"Target vocabulary size: {len(target_vocab)}")
    print("Source mappings:")
    for character in ("a", "h", "r"):
        print(f"  {character!r} -> {source_vocab.token_to_id[character]}")
    print("Target mappings:")
    for character in example_pair.target[:3]:
        print(f"  {character!r} -> {target_vocab.token_to_id[character]}")

    source_ids = source_vocab.encode(example_pair.source)
    target_ids = target_vocab.encode(example_pair.target, add_boundaries=True)
    print(f"Source round trip: {example_pair.source} -> {source_ids} -> {source_vocab.decode(source_ids)}")
    print(f"Target round trip: {example_pair.target} -> {target_ids} -> {target_vocab.decode(target_ids)}")

    unknown_character = "1"
    unknown_ids = source_vocab.encode(unknown_character)
    print(f"Unknown character: {unknown_character!r} -> {unknown_ids} -> {source_vocab.decode(unknown_ids)}")


if __name__ == "__main__":
    verify_training_vocabularies()
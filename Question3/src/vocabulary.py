"""Character-level vocabularies for the transliteration task."""

from __future__ import annotations

from dataclasses import dataclass


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
from typing import Dict

from uniformers.utils import ALLITERATION_LEVELS, METERS, QUATRAIN_RHYME_SCHEMES
from uniformers.models.bygpt5 import ByGPT5Tokenizer


class Poetry2Tokens():
    def __init__(
        self,
        tokenizer,
        alliteration_levels=ALLITERATION_LEVELS,
        meters=METERS,
        rhyme_schemes=QUATRAIN_RHYME_SCHEMES,
    ):
        if len(alliteration_levels) + len(meters) + len(rhyme_schemes) > len(tokenizer.additional_special_tokens):
            # TODO tokenizer.add_special_tokens({"additional_special_tokens": ["<...>"]})
            raise ValueError("Number of special poetry tokens exceeds vocabulary size!")

        if isinstance(tokenizer, ByGPT5Tokenizer):
            tokenizer.add_bos_token = True
            tokenizer.add_eos_token = True
            tokenizer.bos_token = tokenizer.eos_token

        self.tokenizer = tokenizer
        self._alliteration_levels = alliteration_levels
        self._meters = meters
        self._rhyme_schemes = rhyme_schemes
        self._additional_special_ids = [tokenizer._convert_token_to_id(token) for token in tokenizer.additional_special_tokens]  # pyright: ignore

    @property
    def alliterations2tokens(self) -> Dict[str, str]:
        tokens = self.tokenizer.additional_special_tokens
        return {level: tokens[idx] for idx, level in enumerate(self._alliteration_levels)}

    @property
    def alliterations2ids(self) -> Dict[str, int]:
        ids = self._additional_special_ids
        return {level: ids[idx] for idx, level in enumerate(self._alliteration_levels)}

    @property
    def meters2tokens(self) -> Dict[str, str]:
        offset = len(self._alliteration_levels)
        tokens = self.tokenizer.additional_special_tokens
        return {level: tokens[idx] for idx, level in enumerate(self._meters, offset)}

    @property
    def meters2ids(self) -> Dict[str, int]:
        offset = len(self._alliteration_levels)
        ids = self._additional_special_ids
        return {level: ids[idx] for idx, level in enumerate(self._meters, offset)}

    @property
    def rhymes2tokens(self) -> Dict[str, str]:
        offset = len(self._alliteration_levels) + len(self._meters)
        tokens = self.tokenizer.additional_special_tokens
        return {level: tokens[idx] for idx, level in enumerate(self._rhyme_schemes, offset)}

    @property
    def rhymes2ids(self) -> Dict[str, int]:
        offset = len(self._alliteration_levels) + len(self._meters)
        ids = self._additional_special_ids
        return {level: ids[idx] for idx, level in enumerate(self._rhyme_schemes, offset)}

    @property
    def tokens2forms(self) -> Dict[str, str]:
        combined = self.alliterations2tokens | self.meters2tokens | self.rhymes2tokens
        return {v: k for k, v in combined.items()}

    @property
    def ids2forms(self) -> Dict[int, str]:
        combined = self.alliterations2ids | self.meters2ids | self.rhymes2ids
        return {v: k for k, v in combined.items()}


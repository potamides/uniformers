from typing import Dict, List, Optional

from transformers.models.byt5.tokenization_byt5 import ByT5Tokenizer
from ...utils.poetry import ALLITERATION_LEVELS, METERS, QUATRAIN_RHYME_SCHEMES


class ByGPT5Tokenizer(ByT5Tokenizer):
    def __init__(
        self,
        add_prefix_space=False,
        add_bos_token=False,
        add_eos_token=False,
        **kwargs
    ):
        super().__init__(
            add_prefix_space=add_prefix_space,
            add_bos_token=add_bos_token,
            add_eos_token=add_eos_token,
            **kwargs,
        )
        self.add_prefix_space = add_prefix_space
        self.add_bos_token = add_bos_token
        self.add_eos_token = add_eos_token

    def get_special_tokens_mask(
        self, token_ids_0: List[int],
        token_ids_1: Optional[List[int]] = None,
        already_has_special_tokens: bool = False
    ) -> List[int]:
        return super(ByT5Tokenizer, self).get_special_tokens_mask(
            token_ids_0=token_ids_0,
            token_ids_1=token_ids_1,
            already_has_special_tokens=already_has_special_tokens,
        )

    def build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=None):
        if self.add_bos_token:
            bos_token_ids = [self.bos_token_id]
        else:
            bos_token_ids = []
        if self.add_eos_token:
            eos_token_ids = [self.eos_token_id]
        else:
            eos_token_ids = []

        if token_ids_1 is None:
            return bos_token_ids + token_ids_0 + eos_token_ids # pyright: ignore

        return bos_token_ids + token_ids_0 + bos_token_ids + token_ids_1 + eos_token_ids # pyright: ignore

    def prepare_for_tokenization(self, text, is_split_into_words=False, **kwargs):
        if is_split_into_words or self.add_prefix_space:
            text = " " + text
        return (text, kwargs)


class ByGPT5TokenizerForPoetry(ByGPT5Tokenizer):
    def __init__(
        self,
        alliteration_levels=ALLITERATION_LEVELS,
        meters=METERS,
        rhyme_schemes=QUATRAIN_RHYME_SCHEMES,
        **kwargs
    ):
        super().__init__(**kwargs)
        if len(alliteration_levels) + len(meters) + len(rhyme_schemes) > self.vocab_size - self._utf_vocab_size:
            raise ValueError("Number of special poetry tokens exceeds vocabulary size!")

        self._alliteration_levels = alliteration_levels
        self._meters = meters
        self._rhyme_schemes = rhyme_schemes
        self._additional_special_ids = [self._convert_token_to_id(token) for token in self._additional_special_tokens]

    @property
    def alliterations2tokens(self) -> Dict[str, str]:
        tokens = self._additional_special_tokens
        return {level: tokens[idx] for idx, level in enumerate(self._alliteration_levels)}

    @property
    def alliterations2ids(self) -> Dict[str, int]:
        ids = self._additional_special_ids
        return {level: ids[idx] for idx, level in enumerate(self._alliteration_levels)}

    @property
    def meters2tokens(self) -> Dict[str, str]:
        offset = len(self._alliteration_levels) - 1
        tokens = self._additional_special_tokens
        return {level: tokens[idx] for idx, level in enumerate(self._meters, offset)}

    @property
    def meters2ids(self) -> Dict[str, int]:
        offset = len(self._alliteration_levels) - 1
        ids = self._additional_special_ids
        return {level: ids[idx] for idx, level in enumerate(self._meters, offset)}

    @property
    def rhymes2tokens(self) -> Dict[str, str]:
        offset = len(self._alliteration_levels) + len(self._meters) - 1
        tokens = self._additional_special_tokens
        return {level: tokens[idx] for idx, level in enumerate(self._rhyme_schemes, offset)}

    @property
    def rhymes2ids(self) -> Dict[str, int]:
        offset = len(self._alliteration_levels) + len(self._meters) - 1
        ids = self._additional_special_ids
        return {level: ids[idx] for idx, level in enumerate(self._rhyme_schemes, offset)}

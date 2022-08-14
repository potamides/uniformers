from typing import Dict, List, Optional, Tuple

from tokenizers import AddedToken
from transformers.tokenization_utils import PreTrainedTokenizer


PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {
    "bygpt": 1024,
    "bygpt-medium": 1024,
    "bygpt-large": 1024,
    "bygpt-xl": 1024,
}

class ByGPTTokenizer(PreTrainedTokenizer):
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES # pyright: ignore
    model_input_names = ["input_ids", "attention_mask"]

    def __init__(
        self,
        bos_token="<|bos|>",
        eos_token="<|eos|>",
        unk_token="<|unk|>",
        pad_token="<|pad|>",
        extra_ids=125,
        add_prefix_space=False,
        add_bos_token=False,
        add_eos_token=False,
        **kwargs
    ):
        bos_token = AddedToken(bos_token, lstrip=False, rstrip=False) if isinstance(bos_token, str) else bos_token
        eos_token = AddedToken(eos_token, lstrip=False, rstrip=False) if isinstance(eos_token, str) else eos_token
        unk_token = AddedToken(unk_token, lstrip=False, rstrip=False) if isinstance(unk_token, str) else unk_token
        pad_token = AddedToken(pad_token, lstrip=False, rstrip=False) if isinstance(pad_token, str) else pad_token

        if extra_ids > 0:
            additional_special_tokens = [f"<|extra_{i}|>" for i in range(extra_ids)]
        else:
            additional_special_tokens = None

        super().__init__(
            bos_token=bos_token,
            eos_token=eos_token,
            unk_token=unk_token,
            pad_token=pad_token,
            add_prefix_space=add_prefix_space,
            add_bos_token=add_bos_token,
            add_eos_token=add_eos_token,
            additional_special_tokens=additional_special_tokens,
            **kwargs,
        )
        self.add_prefix_space = add_prefix_space
        self.add_bos_token = add_bos_token
        self.add_eos_token = add_eos_token

        self._utf_vocab_size = 2**8  # utf is 8 bits

        # define special tokens dict
        self.special_tokens_encoder: Dict[str, int] = dict() # pyright: ignore
        for token in (self._bos_token, self._eos_token, self._unk_token, self._pad_token):
            if token and str(token) not in self.special_tokens_encoder:
                self.special_tokens_encoder[str(token)] = len(self.special_tokens_encoder)

        self._num_special_tokens = len(self.special_tokens_encoder)
        n = len(self.additional_special_tokens)
        for i, token in enumerate(self.additional_special_tokens):
            self.special_tokens_encoder[token] = self.vocab_size + i - n
        self.special_tokens_decoder: Dict[int, str] = {v: k for k, v in self.special_tokens_encoder.items()} # pyright: ignore

    @property
    def vocab_size(self):
        return self._utf_vocab_size + self._num_special_tokens + len(self._additional_special_tokens)

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
            return bos_token_ids + token_ids_0 + eos_token_ids

        return bos_token_ids + token_ids_0 + bos_token_ids + token_ids_1 + eos_token_ids

    def _tokenize(self, text: str) -> List[str]:
        """Take as input a string and return a list of strings (tokens) for words/sub-words"""
        tokens = [chr(i) for i in text.encode("utf-8")]
        return tokens

    def _convert_token_to_id(self, token):
        """Converts a token (str) in an id using the vocab."""
        if token in self.special_tokens_encoder:
            token_id = self.special_tokens_encoder[token]
        elif token in self.added_tokens_encoder:
            token_id = self.added_tokens_encoder[token]
        elif len(token) != 1:
            token_id = self.unk_token_id
        else:
            token_id = ord(token) + self._num_special_tokens
        return token_id

    def _convert_id_to_token(self, index):
        """Converts an index (integer) in a token (str) using the vocab."""
        if index in self.special_tokens_decoder:
            token = self.special_tokens_decoder[index]
        else:
            token = chr(index - self._num_special_tokens)
        return token

    def convert_tokens_to_string(self, tokens):
        """Converts a sequence of tokens (string) in a single string."""
        bstring = b""
        for token in tokens:
            if token in self.special_tokens_decoder:
                tok_string = self.special_tokens_decoder[token].encode("utf-8") # pyright: ignore
            elif token in self.added_tokens_decoder:
                tok_string = self.special_tokens_decoder[token].encode("utf-8") # pyright: ignore
            elif token in self.special_tokens_encoder:
                tok_string = token.encode("utf-8")
            elif token in self.added_tokens_encoder:
                tok_string = token.encode("utf-8")
            else:
                tok_string = bytes([ord(token)])
            bstring += tok_string
        string = bstring.decode("utf-8", errors="ignore")
        return string

    # ByGPTTokenizer has no vocab file
    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str] | Tuple:
        return ()

    def prepare_for_tokenization(self, text, is_split_into_words=False, **kwargs):
        if is_split_into_words or self.add_prefix_space:
            text = " " + text
        return (text, kwargs)

from collections import ChainMap
from json import load
from os.path import join
from typing import List

from datasets.download.download_manager import DownloadManager
from ipapy.ipastring import IPAString
from torch.autograd.grad_mode import no_grad
from torch.cuda import is_available as has_cuda
from torch.utils.data import DataLoader
from transformers.models.byt5.tokenization_byt5 import ByT5Tokenizer
from transformers.models.t5.modeling_t5 import T5ForConditionalGeneration
from transformers.utils.logging import get_logger

from uniformers.utils import clean_sentence

logger = get_logger("transformers")


class Phonemizer:
    # Theoretically we could support a lot more langauges but we don't need
    # them for now
    lang2charsiu = {
        "de": "ger",
        "en": "eng-us"
    }
    lang2ipadict = {
        "de": "de",
        "en": "en_US",
    }
    ipa_url = (
        "https://github.com/open-dict-data/ipa-dict/releases/download/1.0/json.zip"
    )

    def __init__(self, lang, batch_size=1, device="cuda" if has_cuda() else "cpu"):
        """
        lang: language to phonemize, expected to be in the format that sacremoses uses
        """
        super().__init__()
        self.batch_size = batch_size
        if lang in self.lang2charsiu and lang in self.lang2ipadict:
            self.lang = lang
        else:
            raise ValueError(f"Language <{lang}> not (yet) supported.")

        ipalang = self.lang2ipadict[self.lang]
        path = str(DownloadManager().download_and_extract(self.ipa_url))
        with open(join(path, "json", ipalang + ".json")) as f:
            # chain dicts because dict is in a list, might be multiple
            self.ipadict = dict(ChainMap(*load(f)[ipalang]))

        self.tokenizer = ByT5Tokenizer.from_pretrained("google/byt5-small")
        self.model = T5ForConditionalGeneration.from_pretrained(
            "charsiu/g2p_multilingual_byT5_small"
        ).to(device)  # pyright: ignore

    def __call__(self, arg: List[str] | str, tokenized: bool = False):
        if tokenized:
            return self.word_phonemize(arg)
        else:
            return self.sent_phonemize(arg)

    def word_phonemize(self, words: List[str] | str) -> List[IPAString]:
        phonemes, guess = list(), dict()
        for idx, word in enumerate([words] if isinstance(words, str) else words):
            value = self.ipadict.get(word, self.ipadict.get(word.lower()))
            phonemes.append(value)
            if not value:
                guess[idx] = word
        if guess:
            for batch_keys in DataLoader(list(guess.keys()), batch_size=self.batch_size): # pyright: ignore
                tokenized = self.tokenizer(
                    [f"<{self.lang2charsiu[self.lang]}>:{guess[key.item()]}" for key in batch_keys],
                    return_tensors="pt",
                    padding=True,
                ).to(self.model.device)
                with no_grad():
                    generated = self.model.generate(**tokenized, num_beams=5)
                decoded = self.tokenizer.batch_decode(generated, skip_special_tokens=True)
                for idx, phoneme in zip(batch_keys, decoded):
                    phonemes[idx] = phoneme

        ipa = [IPAString(unicode_string=phoneme, ignore=True) for phoneme in phonemes]
        return ipa[0] if isinstance(words, str) else ipa

    def sent_phonemize(self, batch: List[str] | str) -> List[IPAString | List[IPAString]]:
        indeces, sent_phonemes = [0], []
        sents = [batch] if isinstance(batch, str) else batch
        tokens = [clean_sentence(sent, lang=self.lang, detokenize=False) for sent in sents]
        phonemes = self.word_phonemize([word for sent in tokens for word in sent])

        for sent in tokens:
            indeces.append(len(sent) + indeces[-1])
        for idx in range(len(indeces) - 1):
            sent_phonemes.append(phonemes[indeces[idx]:indeces[idx+1]])

        return sent_phonemes[0] if isinstance(batch, str) else sent_phonemes

def alliteration_score(line: List[IPAString]) -> float:
    dividend, divisor, stressed = 0, 0, list()
    for word in line:
        for idx, char in enumerate(chars := word.cns_vwl_str.ipa_chars):
            if hasattr(char, "is_stress") and char.is_stress and idx + 1 < len(chars) and chars[idx + 1].is_letter:
                stressed.append(chars[idx + 1])
            elif char.is_letter and idx == 0:
                stressed.append(char)
    for i1, c1 in enumerate(stressed):
        for i2, c2 in enumerate(stressed[i1 + 1:], i1 + 1):
            dividend += int(c1.is_equivalent(c2)) / (i2 - i1)
    for i1, c1 in enumerate(stressed):
        for i2, c2 in enumerate(stressed[i1 + 1:], i1 + 1):
            divisor += 1 / (i2 - i1)
    return dividend / divisor if divisor else 0

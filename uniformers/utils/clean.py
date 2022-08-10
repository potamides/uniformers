from string import punctuation
from typing import List
from collections.abc import Iterable

from sacremoses import MosesDetokenizer, MosesPunctNormalizer, MosesTokenizer

# Some English datasets we use are already tokenized, so we have to be careful
# with apostrophes. Generally, this change could lead to unintended
# consequences but for the poetry domain it should be fine
ENGLISH_SPECIFIC_APOSTROPHE = [
    (r"([{0}])\s[']\s([{0}])".format(MosesTokenizer.IsAlpha), r"\1'\2"),
    (r"([{isn}])\s[']\s([s])".format(isn=MosesTokenizer.IsN), r"\1'\2"),
] + MosesTokenizer.ENGLISH_SPECIFIC_APOSTROPHE

# in German poetry the apostrophe is also used for contraction (in contrast to
# prose), so we have to adapt that as well
NON_SPECIFIC_APOSTROPHE = r"\'", "'"  # pyright: ignore


def clean_sentence(
    sentence: str,
    lang: str,
    remove_punct: bool | List[str] = True,
    protected: None | List[str] = None,
    detokenize: bool = True,
):
    mpn = MosesPunctNormalizer(lang=lang)
    md = MosesDetokenizer(lang=lang)
    mt = MosesTokenizer(lang=lang)
    mt.ENGLISH_SPECIFIC_APOSTROPHE = ENGLISH_SPECIFIC_APOSTROPHE
    mt.NON_SPECIFIC_APOSTROPHE = NON_SPECIFIC_APOSTROPHE  # pyright: ignore

    tokenized = mt.tokenize(mpn.normalize(sentence), protected_patterns=protected)
    if remove_punct:
        pct = remove_punct if isinstance(remove_punct, Iterable) else punctuation
        # remove punctuation https://stackoverflow.com/a/56847275
        tokenized = list(
            filter(lambda token: any(t not in pct for t in token), tokenized)
        )
    return md.detokenize(tokenized) if detokenize else tokenized

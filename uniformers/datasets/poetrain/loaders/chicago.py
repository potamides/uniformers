from glob import glob
from itertools import chain, islice
from os.path import basename, join
from random import Random
from re import match as matchre

from transformers.utils import logging

from uniformers.utils import find_rhymes

random = Random(0)
logger = logging.get_logger("transformers")

def chicago_loader(filepath, _):
    globpattern = join(filepath, "rhymedata-*/english_raw/*.txt")
    stanza_nr = 0
    for file in glob(globpattern):
        rhyme_pairs, dissonance_pairs = dict(), dict()
        with open(file, "rb") as f:
            idx, lines = 0, [
                line.strip() for line in f.read().decode("latin1").splitlines()
            ]
            while idx < len(lines):
                if scheme := matchre(r"RHYME((?:\s\w)+(?:\s\*)?)", lines[idx]):
                    scheme, verses = scheme.groups()[-1].split(), list()
                    idx += 2 if lines[idx + 1].startswith("RHYME-POEM") else 1

                    while (
                        idx < len(lines)
                        and ((verse := lines[idx]) or not verses)
                        and not verse.startswith(("AUTHOR", "TITLE", "RHYME"))
                    ):
                        if verse:
                            verses.append(verse)
                        idx += 1

                    if len(verses) == 0:
                        logger.debug(f"Skipping empty poem at {basename(file)}:{idx}.")
                        continue

                    if scheme[-1] == "*":  # special value used to signal repetition
                        rep = [scheme[:-1]] * len(verses)
                        scheme = [chr(ord(char) + idx * len(set(rep[0]))) for idx, seq in enumerate(rep) for char in seq]
                        scheme = scheme[:len(verses)] # pyright: ignore

                    r_pairs, d_pairs = find_rhymes(verses, "".join(scheme))
                    for pair, (v1, v2, label) in r_pairs.items():
                        rhyme_pairs[tuple(sorted(pair))] = (f"chicago-{stanza_nr}-{v1}-{v2}", label)
                    for pair, (v1, v2, label) in d_pairs.items():
                        dissonance_pairs[tuple(sorted(pair))] = (f"chicago-{stanza_nr}-{v1}-{v2}", label)
                    stanza_nr += 1
                else:
                    idx += 1

        # deterministically shuffle dissonance_pairs to minimize repeated verses
        dissonance_pairs = dict(random.sample(list(dissonance_pairs.items()), len(dissonance_pairs)))
        for pair, (_id, label) in chain(rhyme_pairs.items(), islice(dissonance_pairs.items(), len(rhyme_pairs))):
            yield _id, {
                "text": pair,
                "language": "en",
                "labels": label,
            }

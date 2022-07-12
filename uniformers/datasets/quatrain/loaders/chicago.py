from glob import glob
from os.path import join
from re import match as matchre

from uniformers.utils import scheme_to_label


def chicago_loader(filepath, _):
    globpattern = join(filepath, "rhymedata-*/english_raw/*.txt")
    _id = 0
    for file in glob(globpattern):
        with open(file, "rb") as f:
            idx, lines = 0, [
                line.strip() for line in f.read().decode("latin1").splitlines()
            ]
            while idx < len(lines):
                if scheme := matchre(r"RHYME((?:\s\w)+(?:\s\*)?)", lines[idx]):
                    scheme, verses = scheme.groups()[-1].split(), list()

                    idx += 1
                    while (
                        idx < len(lines)
                        and ((verse := lines[idx]) or not verses)
                        and not verse.startswith(("AUTHOR", "TITLE", "RHYME"))
                    ):
                        if verse:
                            verses.append(verse)
                        idx += 1

                    if len(verses) < 4:
                        continue
                    elif scheme[-1] == "*":  # special value used to signal repetition
                        scheme = scheme[:-1] * len(verses)

                    for i in range(len(verses) - 3):
                        verse_win, scheme_win = verses[i : i + 4], scheme[i : i + 4]

                        yield f"chicago-{_id}", {
                            "text": "\n".join(verse_win),
                            "language": "en",
                            "labels": scheme_to_label(*scheme_win),
                        }
                        _id += 1
                else:
                    idx += 1

from csv import DictReader
from re import sub

import datasets

from uniformers.utils import EMOTIONS


_DESCRIPTION = "This is PO-EMO, a corpus of German and English poetry, annotated on line level with aesthetic emotions."
_HOMEPAGE_URL = "https://github.com/tnhaider/poetry-emotion"
_VERSION = "1.0.0"
_BASE_URL = "https://raw.githubusercontent.com/tnhaider/poetry-emotion/92fcc105bed0a46649306680cd91ebe3f793b822/tsv/poemo.{}.emotion.tsv"

_LANGUAGES = {
    "en": ["english"],
    "de": ["german.complete.prodody"],
    "all": ["english", "german.complete.prodody"]
}


class POMEOConfig(datasets.builder.BuilderConfig):
    def __init__(self, *args, lang="de", **kwargs):
        super().__init__(
            *args,
            name=str(lang),
            **kwargs,
        )
        self.lang = lang


class POMEO(datasets.builder.GeneratorBasedBuilder):
    BUILDER_CONFIGS = [
        POMEOConfig(
            lang=lang,
            description=f"Language: {lang}",
            version=datasets.Version(_VERSION),
        )
        for lang in _LANGUAGES.keys()
    ]
    BUILDER_CONFIG_CLASS = POMEOConfig

    def _info(self):
        return datasets.info.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    "language": datasets.Value("string"),
                    "text": datasets.Value("string"),
                    "labels": datasets.Sequence(datasets.ClassLabel(names=EMOTIONS)) # pyright: ignore
                },
            ),
            supervised_keys=None,
            homepage=_HOMEPAGE_URL,
        )

    def _split_generators(self, dl_manager):
        def _base_urls(lang):
            return [_BASE_URL.format(name) for name in _LANGUAGES[lang]]

        download_urls = _base_urls(self.config.lang) # pyright: ignore
        path = dl_manager.download_and_extract(download_urls)
        return [
            datasets.splits.SplitGenerator(
                name=str(datasets.splits.Split.TRAIN),
                gen_kwargs={"datapath": path},
            )
        ]

    def _generate_examples(self, datapath):
        all_verses, all_labels = list(), list()
        for path in datapath:
            with open(path, newline='') as csvfile:
                headers = csvfile.readline().strip()
                if "\t" not in headers:
                    # file is not actually a tsv file...
                    for line in csvfile:
                        if "\t" in line.strip():
                            verse, *labels = line.split("\t")
                            labels = {l.strip() for label in labels for l in label.split("---")}
                            # crude detokenization...
                            verse = verse.replace("—", "-").replace(" ’ ", "' ")
                            verse = sub(r'\s"(\w)', r"'\1", verse)
                            verse = sub(r'(\w)"(\w)', r"\1'\2", verse)
                            all_verses.append(sub(r"\s([.,;:?!](?:\s+|$|-))", r'\1', verse).strip())
                            all_labels.append(sorted(labels))
                else:
                    for line in DictReader(csvfile, fieldnames=headers.split("\t"), delimiter='\t'):
                        labels = {line[f"emotion{x}anno{y}"] for x in range(1, 3) for y in range(1, 3)}
                        all_verses.append(line["line_text"].strip())
                        all_labels.append(sorted([label for label in labels if label != "NONE"]))

        for idx, (verse, label) in enumerate(zip(all_verses, all_labels)):
            label = [l.replace(" / ", "/") for l in label if l != "Nostalgia"]
            assert len(label) > 0
            yield (
                idx,
                {
                    "text": verse,
                    "labels": label,
                    "language": self.config.lang # pyright: ignore
                }
            )

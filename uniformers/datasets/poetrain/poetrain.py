"""
Combination of various datasets from the poetry domain in English and German.
Poems are extracted as verses (for meter) or pairs for (rhymes).
"""
from importlib.metadata import version
from re import sub
from typing import List

from datasets import ClassLabel, Features, Sequence, TextClassification, Value, builder
from datasets.info import DatasetInfo
from datasets.splits import Split, SplitGenerator
from transformers.utils import logging

from uniformers.datasets.poetrain.loaders import (
    chicago_loader,
    fbfv_loader,
    prosodic_loader,
    epg_loader,
    wild_loader,
    grc_loader
)
from uniformers.utils import ALL_METERS, METERS, RHYME_LABELS, normalize_characters

logger = logging.get_logger("transformers")
SUPPORTED_LANGUAGES = ("en", "de")


class PoeTrainConfig(builder.BuilderConfig):
    """BuilderConfig for SuperGLUE."""

    def __init__(self, data_urls, label_classes, normalize=True, **kwargs):
        # Construct a version identifier huggingface is happy with...
        super().__init__(version=sub(r"[^.\d]", "", ".".join(version("uniformers").split(".")[idx] for idx in [0,1,-1])), **kwargs)
        self.data_urls = data_urls
        self.label_classes = label_classes
        self.normalize = normalize


class PoeTrain(builder.GeneratorBasedBuilder):
    """A poetry corpus."""

    BUILDER_CONFIG_CLASS = PoeTrainConfig
    BUILDER_CONFIGS = [
        PoeTrainConfig(
            name="meter",
            data_urls={
                "epg64": "https://github.com/tnhaider/epg64-english-poetry-annotated/archive/696db63899347ccef7d2e20ee6fec220d3527fe7.zip",
                "prosodic": "https://raw.githubusercontent.com/quadrismegistus/prosodic/f4c28f5a3f9bec6f200dde3d2c0823f18da1990e/tagged_samples/tagged-sample-litlab-2016.txt",
                "fbfv": "https://github.com/manexagirrezabal/for_better_for_verse/archive/152df1aac3b8b806e2681221ed314383c4cf2e8d.zip",
                "wild": "https://raw.githubusercontent.com/tnhaider/metrical-tagging-in-the-wild/c70f5ab7dfd865673cf9e9a9a36e54fc5445273d/data/German/SmallGold/antikoerperchen.lines.prosody.emotions.v4.tsv"
            },
            description="A poetry corpus for meter.",
            label_classes=METERS,
        )
    ] + [
        PoeTrainConfig(
            name="rhyme",
            data_urls={
                "epg64": "https://github.com/tnhaider/epg64-english-poetry-annotated/archive/696db63899347ccef7d2e20ee6fec220d3527fe7.zip",
                "chicago": "https://github.com/sravanareddy/rhymedata/archive/ba25424061778b97b7bcb1dac5b19beac257c8c2.zip",
                "fbfv": "https://github.com/manexagirrezabal/for_better_for_verse/archive/152df1aac3b8b806e2681221ed314383c4cf2e8d.zip",
                "wild": "https://raw.githubusercontent.com/tnhaider/metrical-tagging-in-the-wild/c70f5ab7dfd865673cf9e9a9a36e54fc5445273d/data/German/SmallGold/antikoerperchen.lines.prosody.emotions.v4.tsv",
                "grc": "https://github.com/tnhaider/german-rhyme-corpus/archive/c840cd1207dc94fa14b557fbb82d4c89b4793dd3.zip"
            },
            description="A poetry corpus for rhymes.",
            label_classes=RHYME_LABELS,
        )
    ]

    def _info(self):
        features = {
            "text": {"meter": Value("string"), "rhyme": Sequence(Value("string"), 2)}[self.config.name],
            "language": Value("string"),
            "labels": ClassLabel(names=self.config.label_classes)  # pyright: ignore
        }

        if self.config.name == "meter":
            features['text'] = Value("string")
            features['original'] = ClassLabel(names=ALL_METERS) # pyright: ignore

        return DatasetInfo(
            description=str(__doc__),
            features=Features(features),
            supervised_keys=None,
            task_templates=[TextClassification()],
        )

    def _split_generators(self, dl_manager) -> List[SplitGenerator]:
        urls_to_download = self.config.data_urls  # pyright: ignore
        downloaded_files = dl_manager.download_and_extract(urls_to_download)

        return [
            SplitGenerator(
                name=str(Split.TRAIN), gen_kwargs={"datasets": downloaded_files}
            ),
        ]

    def _generate_examples(self, datasets):
        """This function returns the examples in the raw (text) form."""

        all_texts = set()
        for dataset, filepath in datasets.items():
            logger.info("Generating examples from '%s'.", filepath)
            match dataset:
                case "chicago": gtr = chicago_loader(filepath, self.config)
                case "prosodic": gtr = prosodic_loader(filepath, self.config)
                case "fbfv": gtr = fbfv_loader(filepath, self.config)
                case "epg64": gtr = epg_loader(filepath, self.config)
                case "wild": gtr = wild_loader(filepath, self.config)
                case "grc": gtr = grc_loader(filepath, self.config)
                case _: raise ValueError

            for _id, example in gtr:
                if self.config.normalize: # pyright: ignore
                    if self.config.name == "meter":
                        example["text"] = normalize_characters(example["text"])
                    else:
                        example["text"] = tuple(normalize_characters(v) for v in example["text"]) # pyright: ignore
                if not (text := example["text"]) in all_texts:
                    all_texts.add(text)
                    yield _id, example

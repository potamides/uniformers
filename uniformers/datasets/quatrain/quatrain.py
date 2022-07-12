"""
Combination of various datasets from the poetry domain in English and German.
Poems are extracted as quatrains.
"""
from importlib.metadata import version
from re import sub
from typing import List

from datasets import ClassLabel, Features, TextClassification, Value, builder
from datasets.info import DatasetInfo
from datasets.splits import Split, SplitGenerator
from transformers.utils import logging

from uniformers.datasets.quatrain.loaders import (
    chicago_loader,
    #ecpa_loader,
    fbfv_loader,
    prosodic_loader,
)
from uniformers.utils import METERS, QUATRAIN_RHYME_SCHEMES

logger = logging.get_logger("transformers")
SUPPORTED_LANGUAGES = ("en", "de")


class QuaTrainConfig(builder.BuilderConfig):
    """BuilderConfig for SuperGLUE."""

    def __init__(self, data_urls, label_classes, **kwargs):
        """BuilderConfig for QuaTrain.

        Args:
        features: *list[string]*, list of the features that will appear in the
            feature dict. Should not include "labels".
        **kwargs: keyword arguments forwarded to super.
        """

        # Construct a version identifier huggingface is happy with...
        super().__init__(version=sub(r"[^.\d]", "", ".".join(version("uniformers").split(".")[idx] for idx in [0,1,-1])), **kwargs)
        self.data_urls = data_urls
        self.label_classes = label_classes


class QuaTrain(builder.GeneratorBasedBuilder):
    """A poetry corpus."""

    BUILDER_CONFIG_CLASS = QuaTrainConfig
    BUILDER_CONFIGS = [
        QuaTrainConfig(
            name="meter",
            data_urls={
                "prosodic": "https://github.com/quadrismegistus/prosodic/archive/e665fa234ec460739ee5504a3a23177d25dd8864.zip",
                "fbfv": "https://github.com/manexagirrezabal/for_better_for_verse/archive/152df1aac3b8b806e2681221ed314383c4cf2e8d.zip"
                # ecpa seems to be machine annotated
                #"ecpa": "https://github.com/alhuber1502/ECPA/archive/f4753c7d9d2b3583e8f3cb5eda0eb1f1fca0f6e4.zip"
            },
            description="A poetry corpus for rhymes.",
            label_classes=METERS,
        )
    ] + [
        QuaTrainConfig(
            name="rhyme",
            data_urls={
                "chicago": "https://github.com/sravanareddy/rhymedata/archive/ba25424061778b97b7bcb1dac5b19beac257c8c2.zip",
            },
            description="A poetry corpus for meter.",
            label_classes=QUATRAIN_RHYME_SCHEMES,
        )
    ]

    def _info(self):
        return DatasetInfo(
            description=str(__doc__),
            features=Features(
                {
                    "text": Value("string"),
                    "language": Value("string"),
                    "labels": ClassLabel(
                        names=self.config.label_classes  # pyright: ignore
                    ),
                }
            ),
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

        hashes = set()
        for dataset, filepath in datasets.items():
            logger.info("Generating examples from '%s'.", filepath)
            match dataset:
                case "chicago": gtr = chicago_loader(filepath, self.config)
                case "prosodic": gtr = prosodic_loader(filepath, self.config)
                case "fbfv": gtr = fbfv_loader(filepath, self.config)
                case _: raise ValueError

            for _id, example in gtr:
                if not (ex_hash := hash(example["text"])) in hashes:
                    hashes.add(ex_hash)
                    yield _id, example

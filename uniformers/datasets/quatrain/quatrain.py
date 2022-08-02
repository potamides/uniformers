from glob import glob
from importlib.metadata import version
from json import load
from os.path import join
from re import sub
from typing import List

from datasets import Features, Sequence, Value, builder
from datasets.config import EXTRACTED_DATASETS_PATH
from datasets.info import DatasetInfo
from datasets.splits import Split, SplitGenerator
from datasets.utils.file_utils import hash_url_to_filename
from libarchive import memory_reader
from transformers.utils import logging

logger = logging.get_logger("transformers")


class QuaTrainConfig(builder.BuilderConfig):
    """BuilderConfig for QuaTrain."""

    def __init__(self, lang, normalize=True, **kwargs):
        # Construct a version identifier huggingface is happy with...
        super().__init__(name=lang, version=sub(r"[^.\d]", "", ".".join(version("uniformers").split(".")[idx] for idx in [0,1,-1])), **kwargs)
        self.normalize = normalize
        match lang:
            case "de":
                self.data_urls = [
                    "https://github.com/tnhaider/DLK/raw/9b896f104bd282974244a40ddeaf9432a25922ba/DLK/standard/dlk.v5.german.poetry.corpus.full.json.z01",
                    "https://github.com/tnhaider/DLK/raw/9b896f104bd282974244a40ddeaf9432a25922ba/DLK/standard/dlk.v5.german.poetry.corpus.full.json.zip",
                ]
            case "en":
                self.data_urls = "https://github.com/tnhaider/metrical-tagging-in-the-wild/raw/c70f5ab7dfd865673cf9e9a9a36e54fc5445273d/data/English/LargeCorpus/eng_gutenberg_measures_all.json.zip"
            case _:
                raise ValueError


def _multipart_extractor(files):
    raw_data, extracted_files = bytes(), list()
    for f in files:
        with open(f, "rb") as fd:
            raw_data += fd.read()

    with memory_reader(raw_data) as e:
        for entry in e:
            with open(path := join(EXTRACTED_DATASETS_PATH, hash_url_to_filename(entry.pathname)), 'wb') as f:
                for block in entry.get_blocks():
                    f.write(block)
                extracted_files.append(path)
    return extracted_files


class QuaTrain(builder.GeneratorBasedBuilder):
    """A poetry corpus."""

    BUILDER_CONFIG_CLASS = QuaTrainConfig
    BUILDER_CONFIGS = [
        QuaTrainConfig(
            lang="de",
            description="A German poetry corpus.",
        )
    ] + [
        QuaTrainConfig(
            lang="en",
            description="An English poetry corpus.",
        )
    ]

    def _info(self):
        return DatasetInfo(
            description=str(__doc__),
            features=Features({
                "text": Sequence(Value("string"), 4),
                "language": Value("string"),
            }),
            supervised_keys=None,
        )

    def _split_generators(self, dl_manager) -> List[SplitGenerator]:
        urls_to_download = self.config.data_urls  # pyright: ignore
        if type(urls_to_download) == list:
            downloaded_files = _multipart_extractor(dl_manager.download(urls_to_download))
        else:
            downloaded_files = glob(join(str(dl_manager.download_and_extract(urls_to_download)), "*"))

        return [
            SplitGenerator(
                name=str(Split.TRAIN), gen_kwargs={"datasets": downloaded_files}
            ),
        ]

    def _generate_examples(self, datasets):
        """This function returns the examples in the raw (text) form."""
        idx, skipped = 0, 0
        for filepath in datasets:
            logger.debug("Generating examples from '%s'.", filepath)
            with open(filepath) as f:
                for poem in load(f).values():
                    for stanza in poem['poem'].values():
                        lines = [line['text'] for line in stanza.values()]
                        if len(lines) >= 4:
                            for i in range(len(lines) - 3):
                                quatrain = lines[i : i + 4]
                                # some poems are not formatted correctly
                                if any(verse.endswith("-") or "\n" in verse or 60 >= len(verse) <= 10 for verse in quatrain):
                                    logger.debug("Skipping potentially garbled quatrain.")
                                    skipped += 1
                                    continue
                                yield f"{self.config.name}-{idx}", {
                                    "text": quatrain,
                                    "language": self.config.name,
                                }
                                idx += 1
        logger.debug(f"Skipped {skipped} quatrains.")

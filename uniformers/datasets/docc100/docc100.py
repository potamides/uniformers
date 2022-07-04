# based on https://huggingface.co/datasets/cc100
# The original cc100 implementation returns each line separately. This variant
# returns whole documents.
import datasets


_DESCRIPTION = """\
This corpus is an attempt to recreate the dataset used for training XLM-R. This corpus comprises of monolingual data for 100+ languages and also includes data for romanized languages (indicated by *_rom). This was constructed using the urls and paragraph indices provided by the CC-Net repository by processing January-December 2018 Commoncrawl snapshots. Each file comprises of documents separated by double-newlines and paragraphs within the same document separated by a newline. The data is generated using the open source CC-Net repository. No claims of intellectual property are made on the work of preparation of the corpus.
"""
_HOMEPAGE_URL = "https://data.statmt.org/cc-100/"
_VERSION = "1.0.0"
_BASE_URL = "https://data.statmt.org/cc-100/{}.txt.xz"

# Please note: due to the size of the data, only few examples are provided.
# However, you can pass the lang parameter in config to fetch data of any language in the corpus
_LANGUAGES = ["am", "sr", "ka"]


class Cc100Config(datasets.builder.BuilderConfig):
    def __init__(self, *args, lang=None, **kwargs):
        super().__init__(
            *args,
            name=f"{lang}",
            **kwargs,
        )
        self.lang = lang


class Cc100(datasets.builder.GeneratorBasedBuilder):
    BUILDER_CONFIGS = [
        Cc100Config(
            lang=lang,
            description=f"Language: {lang}",
            version=datasets.Version(_VERSION),
        )
        for lang in _LANGUAGES
    ]
    BUILDER_CONFIG_CLASS = Cc100Config

    def _info(self):
        return datasets.info.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    "id": datasets.Value("string"),
                    "text": datasets.Value("string"),
                },
            ),
            supervised_keys=None,
            homepage=_HOMEPAGE_URL,
        )

    def _split_generators(self, dl_manager):
        def _base_url(lang):
            return _BASE_URL.format(lang)

        download_url = _base_url(self.config.lang) # pyright: ignore
        path = dl_manager.download_and_extract(download_url)
        return [
            datasets.splits.SplitGenerator(
                name=str(datasets.splits.Split.TRAIN),
                gen_kwargs={"datapath": path},
            )
        ]

    def _generate_examples(self, datapath):
        document_counter, lines = 0, list()
        with open(datapath, encoding="utf-8") as f:
            for row in f:
                if stripped := row.strip():
                    lines.append(stripped)
                else:
                    result = (
                        document_counter,
                        {
                            "id": str(document_counter),
                            "text": "\n".join(lines),
                        },
                    )
                    document_counter += 1
                    lines.clear()
                    yield result

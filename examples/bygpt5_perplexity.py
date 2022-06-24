#!/usr/bin/env python
from os import makedirs
from os.path import join
from json import dump
from argparse import ArgumentParser

from datasets.load import load_metric, load_dataset
from transformers.models.auto.tokenization_auto import TOKENIZER_MAPPING
from transformers.models.auto.configuration_auto import CONFIG_MAPPING
from transformers.models.t5.configuration_t5 import T5Config
from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING
from uniformers.models.bygpt5 import ByGPT5LMHeadModel, ByGPT5Tokenizer, ByGPT5Config

# we need to add this to to be able to use ByGPT5 with AutoModel
CONFIG_MAPPING.register(ByGPT5Config.model_type, ByGPT5Config)
MODEL_FOR_CAUSAL_LM_MAPPING.register(ByGPT5Config, ByGPT5LMHeadModel)
TOKENIZER_MAPPING.register(ByGPT5Config, (ByGPT5Tokenizer, None))

# and this too, if we want to test the raw ByT5 decoder
MODEL_FOR_CAUSAL_LM_MAPPING.register(T5Config, ByGPT5LMHeadModel)

if __name__ == "__main__":
    argument_parser = ArgumentParser(
        description="Compute perplexity of a language model on various datasets."
    )
    argument_parser.add_argument(
        "--model_name_or_path",
        default="google/byt5-small",
        help="name of the model in huggingface hub or path if local",
    )
    argument_parser.add_argument(
        "--out_dir",
        default="data",
        help="directory where to write the json file with results",
    )
    args = argument_parser.parse_args()
    makedirs(args.out_dir, exist_ok=True)

    perplexity = lambda texts: load_metric("perplexity").compute(
        model_id=args.model_name_or_path,
        add_start_token=False,
        batch_size=1,
        input_texts=texts,
    )
    wikitext103 = load_dataset(
        "dlwh/wikitext_103_detokenized", split="test"
    )[  # pyright: ignore
        "text"
    ]
    lambada = load_dataset("craffel/openai_lambada", split="test")[  # pyright: ignore
        "text"
    ]

    perplexities = {
        "WikiText103": perplexity(wikitext103)['mean_perplexity'], # pyright: ignore
        "LAMBADA": perplexity(lambada)['mean_perplexity'], # pyright: ignore
    }

    with open(
        join(args.out_dir, f"perplexity-{args.model_name_or_path.replace('/', '-')}.json"), "w"
    ) as fp:
        dump(perplexities, fp)  # pyright: ignore
        print(perplexities)

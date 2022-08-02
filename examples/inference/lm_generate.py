#!/usr/bin/env python

from argparse import ArgumentParser

from transformers.models.auto.configuration_auto import CONFIG_MAPPING
from transformers.models.auto.modeling_auto import AutoModelForCausalLM
from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING
from transformers.models.auto.tokenization_auto import TOKENIZER_MAPPING
from transformers.models.auto.tokenization_auto import AutoTokenizer
from transformers.pipelines.text_generation import TextGenerationPipeline
from transformers.models.t5.configuration_t5 import T5Config

from uniformers.models.bygpt5 import ByGPT5Config, ByGPT5LMHeadModel, ByGPT5Tokenizer

# fix some warnings inside pipeline
# we need to add this to to be able to use ByGPT5 with AutoModel
CONFIG_MAPPING.register(ByGPT5Config.model_type, ByGPT5Config)
TOKENIZER_MAPPING.register(ByGPT5Config, (ByGPT5Tokenizer, None))
MODEL_FOR_CAUSAL_LM_MAPPING.register(ByGPT5Config, ByGPT5LMHeadModel)
MODEL_FOR_CAUSAL_LM_MAPPING.register(T5Config, ByGPT5LMHeadModel)


def generate(model_name, prompt, device, min_length=512, max_length=1024):
    pipeline = TextGenerationPipeline(
        model=AutoModelForCausalLM.from_pretrained(model_name),
        tokenizer=AutoTokenizer.from_pretrained(model_name),
        device=device,
    )
    return pipeline(
        prompt,
        min_length=min_length,
        max_length=max_length,
        do_sample=True,
        #num_beams=5,
        # same default settings as textsynth.com
        top_k=40,
        temperature=1,
        top_p=0.9,
    )[0]["generated_text"]


if __name__ == "__main__":
    parser = ArgumentParser(
        description="Generate text using a pretrained ByGPT5 model."
    )
    parser.add_argument("--prompt", help="prompt to complete", required=True)
    parser.add_argument(
        "--model_name_or_path",
        default="google/byt5-small",
        help="name of the model in huggingface hub or path if local",
    )
    parser.add_argument(
        "--device",
        default=-1,
        type=int,
        help="device ordinal for cpu/gpu, setting this to -1 will leverage cpu",
    )
    args = parser.parse_args()
    print(generate(args.model_name_or_path, args.prompt, args.device))

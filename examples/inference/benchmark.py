#!/usr/bin/env python
from argparse import ArgumentParser
from itertools import cycle
from json import dump

from torch.cuda import max_memory_allocated, reset_peak_memory_stats
from torch.utils import benchmark
from transformers.models.auto.modeling_auto import (
    AutoModelForCausalLM as AutoLM,
    AutoModelForSeq2SeqLM as AutoS2S,
)
from transformers.models.auto.tokenization_auto import AutoTokenizer

from uniformers.models.bygpt5 import ByGPT5LMHeadModel, ByGPT5Tokenizer
from uniformers.utils import Poetry2Tokens


class Quatrain:
    def __init__(self, model, tokenizer, quatrain, rhyme, meter, allit):
        self.model = model
        self.tokenizer = tokenizer
        self.quatrain = quatrain

        p2t = Poetry2Tokens(tokenizer)
        self.rhyme_token = p2t.rhymes2tokens[rhyme]
        self.meter_token = p2t.meters2tokens[meter]
        self.allit_token = p2t.alliterations2tokens[allit]

    def __str__(self):
        return "".join(
            [
                self.tokenizer.bos_token,
                self.rhyme_token,
                self.meter_token,
                self.allit_token,
                self.quatrain,
                self.tokenizer.eos_token,
            ]
        )

    def tokenize(self, separate=False):
        """
        When separate is True, devices (model input) and quatrain (labels) are
        returned separately.
        """
        devices = self.rhyme_token + self.meter_token + self.allit_token

        if self.model.config.is_encoder_decoder:
            devices_tokens = self.tokenizer(
                devices, add_special_tokens=False, return_tensors="pt"
            )
            quatrain_tokens = self.tokenizer(self.quatrain, return_tensors="pt")["input_ids"]
            if separate:
                return devices_tokens, quatrain_tokens.tolist()[0]
            else:
                devices_tokens["labels"] = quatrain_tokens
                return devices_tokens
        else:
            if separate:
                return (
                    self.tokenizer(self.tokenizer.bos_token + devices, return_tensors="pt"),
                    self.tokenizer(self.quatrain + self.tokenizer.eos_token)["input_ids"]
                )
            else:
                return self.tokenizer(
                    self.tokenizer.bos_token
                    + devices
                    + self.quatrain
                    + self.tokenizer.eos_token,
                    return_tensors="pt",
                )


def try_load(model_name):
    try:
        model, tokenizer = AutoLM.from_pretrained(
            model_name
        ), AutoTokenizer.from_pretrained(model_name)
    except (EnvironmentError, KeyError, ValueError):
        try:
            model, tokenizer = AutoS2S.from_pretrained(
                model_name
            ), AutoTokenizer.from_pretrained(model_name)
        except (EnvironmentError, KeyError, ValueError):
            model, tokenizer = ByGPT5LMHeadModel.from_pretrained(
                model_name
            ), ByGPT5Tokenizer.from_pretrained(model_name)

    return model, tokenizer


def benchmark_model(model, quatrain: Quatrain, number=1000):
    model_input, labels = quatrain.tokenize(separate=True)
    # We could supply the quatrain as labels (encoder-decoder) or with the
    # input tokens (decoder-only) but since the sequence would then be known
    # in advance the transformer implementation would parallelize all
    # computations. This makes sense in general but here we want to simulate
    # the generation process so this should done sequentially. This is why we
    # use 'prefix_allowed_tokens_fn'.
    timer = benchmark.Timer(
        stmt="model.generate(**input, pad_token_id=-1, max_length=384, prefix_allowed_tokens_fn=lambda *_: [next(labels)])",
        globals={
            "model": model.to("cuda"),
            "input": model_input.to("cuda"),
            "labels": cycle(labels)
        },
    )

    reset_peak_memory_stats()
    time = timer.timeit(number).median
    memory = max_memory_allocated()
    parameters = sum(t.numel() for t in model.parameters())

    return {"time": time, "memory": memory, "parameters": parameters}

def main(models, number):
    results = dict()
    sample_sequence = "\n".join(
        [
            "When I consider how my light is spent,",
            "Ere half my days, in this dark world and wide,",
            "And that one Talent which is death to hide",
            "Lodged with me useless, though my Soul more bent",
        ]
    )

    for model_type, model_name in models.items():
        if model_name:
            model, tokenizer = try_load(model_name)
            quatrain = Quatrain(model, tokenizer, sample_sequence, "ABBA", "iambus", "medium")
            results[model_type.replace("_", "-")] = benchmark_model(model, quatrain, number)

    return results

if __name__ == "__main__":
    pass

    parser = ArgumentParser(
        description="Benchmark all poetry generation models."
    )
    parser.add_argument(
        "--number",
        type=int,
        default=1000,
        help="number of runs the timeit module should perform",
    )
    parser.add_argument(
        "--output",
        help="path to the output json file (do not print to stdout)",
    )

    for size in ["small", "base", "medium"]:
        parser.add_argument(
            f"--bygpt5-{size}",
            default=f"nllg/poetry-bygpt5-{size}-en",
            help=f"name of the bygpt5-{size} model in huggingface hub or path if local",
        )
    for model in ["byt5-small", "mt5-small", "gpt2-base", "gpt2-medium"]:
        parser.add_argument(
            f"--{model}",
            help=f"name of the {model} model in huggingface hub or path if local",
        )

    models = vars(parser.parse_args())
    number = models.pop("number")
    output = models.pop("output")

    if not output:
        print(main(models, number))
    else:
        with open(output, "w") as f:
            dump(main(models, number), f)

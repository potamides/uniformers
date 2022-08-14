#!/usr/bin/env -S torchrun --nproc_per_node gpu
from argparse import ArgumentParser
from os.path import join
from os.path import basename

from transformers.models.auto.modeling_auto import (
    AutoModelForCausalLM as AutoLM,
    AutoModelForSeq2SeqLM as AutoS2S,
)
from transformers.models.auto.tokenization_auto import AutoTokenizer
from transformers.trainer_utils import set_seed
from transformers.utils.logging import (
    enable_explicit_format,
    get_logger,
    set_verbosity_debug,
    set_verbosity_info,
)

from uniformers.models.bygpt5 import ByGPT5LMHeadModel, ByGPT5Tokenizer
from uniformers.trainers import PoetryLMTrainer

set_verbosity_info()
enable_explicit_format()
logger = get_logger("transformers")
set_seed(0)

def try_load(model_name):
    try:
        model, tokenizer = AutoLM.from_pretrained(model_name), AutoTokenizer.from_pretrained(model_name)
    except (EnvironmentError, KeyError, ValueError):
        try:
            model, tokenizer = AutoS2S.from_pretrained(model_name), AutoTokenizer.from_pretrained(model_name)
        except (EnvironmentError, KeyError, ValueError):
            model, tokenizer = ByGPT5LMHeadModel.from_pretrained(model_name), ByGPT5Tokenizer.from_pretrained(model_name)

    return model, tokenizer


def train(
    base_model,
    output_dir,
    lang="en",
    meter_model_name_or_path="nllg/clf-canine-m",
    rhyme_model_name_or_path="nllg/clf-canine-r",
    gradient_accumulation_steps=8,
    gradient_checkpointing=False,
    test_run=False,
):
    try:
        model, tokenizer = try_load(output_dir)
        logger.info(f"Model already trained. Skipping.")
    except EnvironmentError:
        model, tokenizer = try_load(base_model)
        trainer = PoetryLMTrainer(
            model,
            tokenizer,
            output_dir,
            lang=lang,
            meter_model_name_or_path=meter_model_name_or_path,
            rhyme_model_name_or_path=rhyme_model_name_or_path,
            gradient_accumulation_steps=gradient_accumulation_steps,
            gradient_checkpointing=gradient_checkpointing,
            test_run=test_run,
        )
        trainer.train()
        trainer.save_model()
        trainer.save_state()
        trainer.test()
        model = trainer.model
    return model, tokenizer


if __name__ == "__main__":
    argument_parser = ArgumentParser(
        description="Fine-tune language models for poetry generation"
    )
    argument_parser.add_argument(
        "--model_name_or_path",
        default="google/byt5-small",
        help="name of the base model in huggingface hub or path if local",
    )
    argument_parser.add_argument(
        "--meter_model_name_or_path",
        default="nllg/clf-canine-m",
        help="name or path of the meter classification model",
    )
    argument_parser.add_argument(
        "--rhyme_model_name_or_path",
        default="nllg/clf-canine-r",
        help="name or path of the rhyme classification model",
    )
    argument_parser.add_argument(
        "--out_dir",
        default="models",
        help="directory where to write the model files",
    )
    argument_parser.add_argument(
        "--out_name",
        help="name for the model directory in out_dir, when not given defaults to basename(model_name_or_path)",
    )
    argument_parser.add_argument(
        "--lang",
        choices=["en", "de"],
        default="en",
        help="specify which language to train on",
    )
    argument_parser.add_argument(
        "--grad_acc_steps",
        default=8,
        type=int,
        help="number of gradient accumulation steps",
    )
    argument_parser.add_argument(
        "--debug",
        action="store_true",
        help="perform a test run on debug verbosity",
    )
    args = argument_parser.parse_args()

    if args.debug:
        set_verbosity_debug()
        args.out_dir = join(args.out_dir, "debug")

    train(
        args.model_name_or_path,
        join(args.out_dir, args.out_name or basename(args.model_name_or_path), args.lang),
        meter_model_name_or_path=args.meter_model_name_or_path,
        rhyme_model_name_or_path=args.rhyme_model_name_or_path,
        lang=args.lang,
        gradient_accumulation_steps=args.grad_acc_steps,
        test_run=args.debug,
    )

# examples/training/poetry_training.py --model_name_or_path=models/bygpt5-base/de --out_dir=models/poetry --out_name=bygpt5-base --meter_model_name_or_path=models/canine-c/clf-canine-m --rhyme_model_name_or_path=models/canine-c/clf-canine-r --lang=de --grad_acc_steps=8

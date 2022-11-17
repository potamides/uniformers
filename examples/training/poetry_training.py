#!/usr/bin/env -S torchrun --nproc_per_node gpu
from argparse import ArgumentParser
from functools import partial
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
from uniformers.trainers import PoetryEmotionLMTrainer, PoetryLMTrainer

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
    emotion_model_name_or_path="nllg/clf-bert-e",
    coherence_model_name_or_path="bert-base-multilingual-cased",
    gradient_accumulation_steps=8,
    gradient_checkpointing=False,
    test_run=False,
    emotion=False,
    do_test=True,
    low_resource=False
):
    poetry_models = {
        "meter_model_name_or_path": meter_model_name_or_path,
        "rhyme_model_name_or_path": rhyme_model_name_or_path,
        "coherence_model_name_or_path": coherence_model_name_or_path
    }
    emotion_models = {
        "emotion_model_name_or_path": emotion_model_name_or_path,
    }
    Trainer = partial(
        PoetryEmotionLMTrainer if emotion else PoetryLMTrainer,
        output_dir=output_dir,
        lang=lang,
        gradient_accumulation_steps=gradient_accumulation_steps,
        gradient_checkpointing=gradient_checkpointing,
        test_run=test_run,
        low_resource=low_resource,
        **(emotion_models if emotion else poetry_models),
    )
    try:
        model, tokenizer = try_load(output_dir)
        trainer = Trainer(model, tokenizer)
        logger.info(f"Model already trained. Skipping training.")
    except EnvironmentError:
        model, tokenizer = try_load(base_model)
        trainer = Trainer(model, tokenizer)
        trainer.train()
        trainer.save_model()
        trainer.save_state()
        model = trainer.model

    if do_test:
        trainer.test()
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
        "--emotion_model_name_or_path",
        default="nllg/clf-bert-e",
        help="name or path of the emotion classification model",
    )
    argument_parser.add_argument(
        "--coherence_model_name_or_path",
        default="bert-base-multilingual-cased",
        help="name or path of the coherence model (for BERT NSP)",
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
    argument_parser.add_argument(
        "--emotion",
        action="store_true",
        help="train conditioned on emotion, not poetic devices",
    )
    argument_parser.add_argument(
        "--low_resource",
        action="store_true",
        help="perform a low resource training run",
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
        emotion_model_name_or_path=args.emotion_model_name_or_path,
        coherence_model_name_or_path=args.coherence_model_name_or_path,
        lang=args.lang,
        gradient_accumulation_steps=args.grad_acc_steps,
        emotion=args.emotion,
        test_run=args.debug,
        low_resource=args.low_resource
    )

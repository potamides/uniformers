#!/usr/bin/env -S torchrun --nproc_per_node gpu
from argparse import ArgumentParser
from os.path import join
from os.path import basename

from transformers.trainer_utils import set_seed
from transformers.utils.logging import (
    enable_explicit_format,
    get_logger,
    set_verbosity_info,
    set_verbosity_debug,
)

from uniformers.models.bygpt5 import ByGPT5Config, ByGPT5LMHeadModel, ByGPT5Tokenizer
from uniformers.trainers import LMTrainer

set_verbosity_info()
enable_explicit_format()
logger = get_logger("transformers")
set_seed(0)


def train(
    base_model,
    output_dir,
    lang="en",
    from_scratch=False,
    gradient_accumulation_steps=8,
    gradient_checkpointing=False,
    test_run=False,
):
    logger.info(
        f"Training {basename(base_model)}-style model from {'scratch' if from_scratch else 'checkpoint'}."
    )
    try:
        model, tokenizer = ByGPT5LMHeadModel.from_pretrained(
            output_dir
        ), ByGPT5Tokenizer.from_pretrained(output_dir)
        logger.info(f"Model already trained. Skipping.")
    except EnvironmentError:
        if from_scratch:
            config = ByGPT5Config.from_pretrained(base_model)
            model = ByGPT5LMHeadModel(config)
        else:
            model = ByGPT5LMHeadModel.from_pretrained(base_model)
        # T5 doesn't support both cache and grad checkpointing at the same time
        model.config.use_cache = not gradient_checkpointing  # pyright: ignore
        tokenizer = ByGPT5Tokenizer.from_pretrained(base_model)
        trainer = LMTrainer(
            model,
            tokenizer,
            output_dir,
            lang=lang,
            gradient_accumulation_steps=gradient_accumulation_steps,
            gradient_checkpointing=gradient_checkpointing,
            test_run=test_run,
            transfer=True,
        )
        trainer.train()
        trainer.save_model()
        trainer.save_state()
    return model, tokenizer


def byt5_size_to_bygpt5_size(size):
    """
    Since we only use the decoder our models are a lot smaller which is why we
    use more appropriate names.
    """
    match size:
        case "large":
            return "medium"
        case "xl":
            return "large"
        case "xxl":
            return "xl"
        case _:
            return size


if __name__ == "__main__":
    argument_parser = ArgumentParser(
        description="Extract the decoder of byT5 and pre-train it for language modeling."
    )
    argument_parser.add_argument(
        "--model_size",
        choices=["small", "base", "large", "xl", "xxl"],
        default="small",
        help="specify which byt5 model size to extract the decoder from",
    )
    argument_parser.add_argument(
        "--out_dir",
        default="models",
        help="directory where to write the model files",
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

    base_model = f"google/byt5-{args.model_size}"
    output_dir = join(
        args.out_dir, f"bygpt5-{byt5_size_to_bygpt5_size(args.model_size)}", args.lang
    )

    train(
        base_model,
        output_dir,
        lang=args.lang,
        gradient_accumulation_steps=args.grad_acc_steps,
        test_run=args.debug,
    )

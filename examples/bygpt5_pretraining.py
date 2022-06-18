#!/usr/bin/env -S python -m torch.distributed.launch --nproc_per_node gpu
from os import getenv
from os.path import join

from transformers.utils.logging import (
    enable_explicit_format,
    set_verbosity_info,
    get_logger,
)

from uniformers.models.bygpt5 import ByGPT5Config, ByGPT5LMHeadModel, ByGPT5Tokenizer
from uniformers.trainer import LMTrainer

from os.path import basename

set_verbosity_info()
enable_explicit_format()
logger = get_logger("transformers")

size = getenv("UNIFORMERS_MODEL_SIZE", "small")
output_dir = join(getenv("UNIFORMERS_DATA", ""), f"models/bygpt5-{size}")
base_model = f"google/byt5-{size}"


def train(from_scratch=False, gradient_accumulation_steps=8, gradient_checkpointing=False):
    directory = f"{output_dir}-scratch" if from_scratch else output_dir
    logger.info(
        f"Training {basename(base_model)}-style model from {'scratch' if from_scratch else 'checkpoint'}."
    )
    try:
        model, tokenizer = ByGPT5LMHeadModel.from_pretrained(
            directory
        ), ByGPT5Tokenizer.from_pretrained(directory)
    except EnvironmentError:
        if from_scratch:
            config = ByGPT5Config.from_pretrained(base_model)
            model = ByGPT5LMHeadModel(config)
        else:
            model = ByGPT5LMHeadModel.from_pretrained(base_model)
        # T5 doesn't support both cache and grad checkpointing at the same time
        model.config.use_cache = not gradient_checkpointing # pyright: ignore
        tokenizer = ByGPT5Tokenizer.from_pretrained(base_model)
        trainer = LMTrainer(
            model,
            tokenizer,
            directory,
            gradient_accumulation_steps=gradient_accumulation_steps,
            gradient_checkpointing=gradient_checkpointing,
            #test_run=True,
            transfer=True,
        )
        trainer.train()
        trainer.save_model()
        trainer.save_state()
    return model, tokenizer


if __name__ == "__main__":
    train()
    if size == "small":
        # contrast reusing weights with training from scratch
        train(from_scratch=True)

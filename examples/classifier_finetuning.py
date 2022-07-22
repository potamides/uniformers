#!/usr/bin/env -S torchrun --nproc_per_node gpu
from argparse import ArgumentParser
from os.path import join
from os.path import basename

from transformers.trainer_utils import set_seed
from transformers.utils.logging import (
    enable_explicit_format,
    get_logger,
    set_verbosity_debug,
    set_verbosity_info,
)

from transformers.models.auto.modeling_auto import AutoModelForSequenceClassification as AutoMFSC
from transformers.models.auto.tokenization_auto import AutoTokenizer
from uniformers.trainers import PoetryClassificationTrainer
from uniformers.utils import METERS, RHYME_LABELS

set_verbosity_info()
enable_explicit_format()
logger = get_logger("transformers")
set_seed(0)

def model_init(name, task):
    num_labels = len(METERS) if task == "meter" else len(RHYME_LABELS)
    return lambda: AutoMFSC.from_pretrained(name, num_labels=num_labels)

def train(
    base_model,
    output_dir,
    task,
    lang="en",
    gradient_accumulation_steps=1,
    gradient_checkpointing=False,
    test_run=False,
):
    try:
        model, tokenizer = AutoMFSC.from_pretrained(output_dir), AutoTokenizer.from_pretrained(output_dir)
        logger.info(f"Model already trained. Skipping.")
    except EnvironmentError:
        tokenizer = AutoTokenizer.from_pretrained(base_model)
        trainer = PoetryClassificationTrainer(
            model_init(base_model, task),
            tokenizer,
            output_dir,
            lang=lang,
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
        description="Fine-tune an encoder like CANINE-c"
    )
    argument_parser.add_argument(
        "--model_name_or_path",
        default="google/canine-c",
        help="name of the base model in huggingface hub or path if local",
    )
    argument_parser.add_argument(
        "--out_dir",
        default="models",
        help="directory where to write the model files",
    )
    argument_parser.add_argument(
        "--task",
        choices=["rhyme", "meter"],
        default="meter",
        help="specify which task to train on",
    )
    argument_parser.add_argument(
        "--grad_acc_steps",
        default=1,
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
        join(args.out_dir, basename(args.model_name_or_path), args.task),
        args.task,
        lang=args.lang,
        gradient_accumulation_steps=args.grad_acc_steps,
        test_run=args.debug,
    )

from functools import cached_property
from itertools import chain
from os.path import isdir, isfile, join
from random import randrange
from textwrap import shorten

from datasets import Features, Sequence, Value
from datasets.fingerprint import update_fingerprint
from datasets.load import load_dataset
from transformers.data.data_collator import DataCollatorForLanguageModeling
from transformers.trainer import Trainer
from transformers.trainer_utils import get_last_checkpoint
from transformers.training_args import TrainingArguments
from transformers.utils import logging

logger = logging.get_logger("transformers")

LM_LEARNING_RATES = {
    "base": 6e-4,
    "medium": 3e-4,
    "large": 2.5e-4,
    "xl": 2e-4,
}


class LMTrainingArguments(TrainingArguments):
    """
    TrainingArguments class which evenly distributes batch_size on available
    GPUs under distributed training (DistributedDataParallel). Normal
    TrainingArguments use same batch_size on each GPU. (see
    https://discuss.pytorch.org/t/should-we-split-batch-size-according-to-ngpu-per-node-when-distributeddataparallel/72769/15)
    This should also work for DataParallel which does splitting on its own (see
    https://discuss.pytorch.org/t/a-question-concerning-batchsize-and-multiple-gpus-in-pytorch/33767).
    Additionally, batch_size is scaled according to gradient accumulation
    steps.
    """

    def __init__(
        self,
        global_train_batch_size=8,
        global_eval_batch_size=8,
        eval_samples=1000,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.global_train_batch_size = global_train_batch_size
        self.global_eval_batch_size = global_eval_batch_size
        self.eval_samples = eval_samples
        self.per_device_train_batch_size = self._scale_batch_size(
            global_train_batch_size
        )
        self.per_device_eval_batch_size = self._scale_batch_size(global_eval_batch_size)
        if self.world_size > 1:
            logger.info(f"Dividing batches equally on {self.world_size} processes.")

    def _scale_batch_size(self, batch_size) -> int:
        scaled_batch_size, remainder = divmod(
            batch_size,
            self.world_size * self.gradient_accumulation_steps,
        )
        if remainder != 0:
            raise ValueError(
                "`batch_size` must be divisible by number of processes times gradient accumulation steps."
            )
        return scaled_batch_size


class LMTrainer(Trainer):
    def __init__(
        self,
        model,
        tokenizer,
        output_dir,
        lang="en",
        sequence_length=1024,
        # https://github.com/huggingface/transformers/issues/14608#issuecomment-1004390803
        fp16=False,
        bf16=True,
        tf32=False,
        test_run=False,
        transfer=False,
        overwrite_output_dir=False,
        # only change below stuff when model doesn't fit into memory (see
        # https://huggingface.co/docs/transformers/performance)
        gradient_accumulation_steps=1,
        gradient_checkpointing=False,
        **kwargs,
    ):

        self.model = model
        self.tokenizer = tokenizer
        self.sequence_length = sequence_length
        data_collator = DataCollatorForLanguageModeling(self.tokenizer, mlm=False)

        if self.parameters < 280 * 10**6:
            learning_rate = "base"
        elif self.parameters < 770 * 10**6:
            learning_rate = "medium"
        elif self.parameters < 1550 * 10**6:
            learning_rate = "large"
        else:
            learning_rate = "xl"
        logger.info(
            f"Using learning rate for training {learning_rate} language models."
        )

        # interesting resource: https://huggingface.co/course/chapter7/6?fw=pt
        self.args = LMTrainingArguments(
            optim="adamw_torch",
            lr_scheduler_type="cosine",
            learning_rate=LM_LEARNING_RATES[learning_rate],
            max_steps=1000 if test_run else (50_000 if transfer else 600_000), # ~25Gb of text (for transfer)
            eval_samples=200 if test_run else 102_400, # 100Mb of text
            weight_decay=0.1,
            warmup_ratio=0.01,
            global_train_batch_size=512,
            global_eval_batch_size=512,
            fp16=fp16,
            bf16=bf16,
            tf32=tf32,
            save_total_limit=2,
            overwrite_output_dir=overwrite_output_dir,
            gradient_accumulation_steps=gradient_accumulation_steps,
            gradient_checkpointing=gradient_checkpointing,
            ddp_find_unused_parameters=False,
            evaluation_strategy="no" if test_run else "steps",
            eval_steps=2500,
            logging_steps=20 if test_run else 250,
            logging_first_step=True,
            save_steps=2500,
            output_dir=output_dir,
        )

        if test_run:
            split_dataset = self.load_dataset("stas/openwebtext-10k")
        elif lang == "de":
            split_dataset = self.load_dataset("cc100", lang=lang)
        else:
            split_dataset = self.load_dataset(
                "the_pile_openwebtext2",
                # column types need special care for this dataset
                features=Features(
                    {
                        "title": Value("string"),
                        "text": Value("string"),
                        "reddit_scores": Sequence(Value("int32")),
                    }
                ),
            )

        super().__init__(
            model=self.model,
            tokenizer=self.tokenizer,
            args=self.args,
            data_collator=data_collator,
            train_dataset=split_dataset["train"],  # pyright: ignore
            eval_dataset=split_dataset["test"],  # pyright: ignore
            **kwargs,
        )

    def load_dataset(self, dataset_name, **kwargs):
        # the minimum amount of samples we need for training
        self.min_num_samples = (
            self.args.max_steps * self.args.global_train_batch_size
            + self.args.eval_samples
        )
        self._num_samples = 0
        raw_dataset = load_dataset(dataset_name, split="train", **kwargs)
        # internal hash computation is not the same across sessions so we have
        # to compute it ourselves
        new_fingerprint = update_fingerprint(
            raw_dataset._fingerprint,  # pyright: ignore
            self.__class__.tokenize,
            {
                "tokenizer": self.tokenizer,
                "sequence_length": self.sequence_length,
                "split": self.__class__.split,
                "min_num_samples": self.min_num_samples,
            },
        )
        # we only load the number of samples we need to save some disk space
        logger.info("Tokenizing dataset, progress reports may be inaccurate (upper bound).")
        tokenized_dataset = raw_dataset.map(
            self.tokenize,
            batched=True,
            new_fingerprint=new_fingerprint,  # pyright: ignore
            remove_columns=raw_dataset.column_names,  # pyright: ignore
        )
        split_dataset = tokenized_dataset.train_test_split(
            test_size=self.args.eval_samples
        )
        logger.info(f"Original dataset length: {len(raw_dataset)}")  # pyright: ignore
        logger.info(
            f"Tokenized dataset length: {len(split_dataset['train'])} (train)"
            f", {len(split_dataset['test'])} (test)"
        )
        index = randrange(len(split_dataset["train"]))
        logger.info(
            shorten(
                f"Sample {index} of the training set: {split_dataset['train'][index]}.",
                width=120,
            )
        )
        return split_dataset

    def train(self, **kwargs):
        if not self.args.overwrite_output_dir and isdir(self.args.output_dir):
            last_checkpoint = get_last_checkpoint(self.args.output_dir)

            if isfile(join(self.args.output_dir, "config.json")):
                logger.warning(
                    f"Output directory ({self.args.output_dir}) exists already and is not empty. Skipping training."
                )
            elif last_checkpoint:
                logger.info(
                    f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                    "the `output_dir` or add `overwrite_output_dir` to train from scratch."
                )
        else:
            last_checkpoint = None
        return super().train(resume_from_checkpoint=last_checkpoint, **kwargs)

    def tokenize(self, element):
        input_batch = []
        if self._num_samples < self.min_num_samples:
            outputs = self.tokenizer(element["text"])

            for input_ids in chain.from_iterable(
                self.split(ids, self.sequence_length)
                for ids in outputs["input_ids"]  # pyright: ignore
            ):
                if len(input_ids) == self.sequence_length:
                    input_batch.append(input_ids)
            self._num_samples = self._num_samples + len(input_batch)
        return {"input_ids": input_batch}

    def split(self, lst, n):
        """Yield successive n-sized chunks from lst."""
        for i in range(0, len(lst), n):
            yield lst[i : i + n]

    @cached_property
    def parameters(self):
        return sum(t.numel() for t in self.model.parameters())

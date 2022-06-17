from functools import cached_property
from itertools import chain
from os.path import isdir, isfile, join
from random import randrange
from textwrap import shorten

from datasets.load import load_dataset
from datasets.fingerprint import update_fingerprint
from torch.cuda import is_available as has_cuda
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
    https://discuss.pytorch.org/t/a-question-concerning-batchsize-and-multiple-gpus-in-pytorch/33767)
    """

    def __init__(self, global_train_batch_size=8, global_eval_batch_size=8, **kwargs):
        kwargs["per_device_train_batch_size"] = global_train_batch_size
        kwargs["per_device_eval_batch_size"] = global_eval_batch_size
        super().__init__(**kwargs)
        if self.world_size > 1:
            logger.info(f"Dividing batches equally on {self.world_size} processes.")

    @property
    def train_batch_size(self) -> int:
        train_batch_size, remainder = divmod(
            self.per_device_train_batch_size, self.world_size
        )
        if remainder != 0:
            raise ValueError("`batch_size` must be divisible by number of processes.")
        return train_batch_size

    @property
    def eval_batch_size(self) -> int:
        eval_batch_size, remainder = divmod(
            self.per_device_eval_batch_size, self.world_size
        )
        if remainder != 0:
            raise ValueError("`batch_size` must be divisible by number of processes.")
        return eval_batch_size


class LMTrainer(Trainer):
    def __init__(
        self,
        model,
        tokenizer,
        output_dir,
        sequence_length=1024,
        fp16=has_cuda(),
        tf32=has_cuda(),
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

        dataset_name = "stas/openwebtext-10k" if test_run else "the_pile_openwebtext2"
        raw_dataset = load_dataset(dataset_name, split="train")
        # internal hash computation is not the same across sessions so we have
        # to compute it ourselves
        new_fingerprint = update_fingerprint(
            raw_dataset._fingerprint, # pyright: ignore
            self.__class__.tokenize,
            {
                "tokenizer": tokenizer,
                "sequence_length": self.sequence_length,
                "split": self.__class__.split,
            },
        )
        tokenized_dataset = raw_dataset.map(
            self.tokenize,
            batched=True,
            new_fingerprint=new_fingerprint,  # pyright: ignore
            remove_columns=raw_dataset.column_names,  # pyright: ignore
        )
        logger.info(f"Original dataset length: {len(raw_dataset)}")  # pyright: ignore
        logger.info(f"Tokenized dataset length: {len(tokenized_dataset)}")
        index = randrange(len(tokenized_dataset))
        logger.info(shorten(f"Sample {index} of the training set: {tokenized_dataset[index]}.", width=120))

        split_dataset = tokenized_dataset.train_test_split(test_size=0.005)
        data_collator = DataCollatorForLanguageModeling(self.tokenizer, mlm=False)

        if self.parameters < 350 * 10**6:
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
        args = LMTrainingArguments(
            optim="adamw_hf",
            lr_scheduler_type="cosine",
            learning_rate=LM_LEARNING_RATES[learning_rate],
            num_train_epochs=1,
            max_steps=1000 if test_run else (100_000 if transfer else 600_000),
            weight_decay=0.1,
            warmup_ratio=0.01,
            global_train_batch_size=512,
            global_eval_batch_size=512,
            fp16=fp16,
            tf32=tf32,
            save_total_limit=2,
            overwrite_output_dir=overwrite_output_dir,
            gradient_accumulation_steps=gradient_accumulation_steps,
            gradient_checkpointing=gradient_checkpointing,
            evaluation_strategy="no" if test_run else "steps",
            eval_steps=50000,
            logging_steps=test_run and 20 or 1000,
            save_steps=10000,
            output_dir=output_dir,
        )
        super().__init__(
            model=self.model,
            tokenizer=self.tokenizer,
            args=args,
            data_collator=data_collator,
            train_dataset=split_dataset["train"],  # pyright: ignore
            eval_dataset=split_dataset["test"],  # pyright: ignore
            **kwargs,
        )

    def train(self, **kwargs):
        if not self.args.overwrite_output_dir and isdir(self.args.output_dir):
            last_checkpoint = get_last_checkpoint(self.args.output_dir)

            if isfile(join(self.args.output_dir, 'config.json')):
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
        outputs = self.tokenizer(element["text"])
        input_batch = []

        for input_ids in chain.from_iterable(
            self.split(ids, self.sequence_length)
            for ids in outputs["input_ids"]  # pyright: ignore
        ):
            if len(input_ids) == self.sequence_length:
                input_batch.append(input_ids)
        return {"input_ids": input_batch}

    def split(self, lst, n):
        """Yield successive n-sized chunks from lst."""
        for i in range(0, len(lst), n):
            yield lst[i : i + n]

    @cached_property
    def parameters(self):
        return sum(t.numel() for t in self.model.parameters())

    #@classmethod
    #def from_pretrained(cls, output_dir):
    #    if isfile(join(output_dir, "config.json")):  # pyright: ignore
    #        logger.info(
    #            f"Output directory ({output_dir}) exists and is not empty. "
    #            "Returning pretrained models."
    #        )

    #        config = AutoConfig.from_pretrained(
    #            output_dir,
    #        )
    #        tokenizer = AutoTokenizer.from_pretrained(
    #            output_dir,
    #        )
    #        model = AutoModelForSeq2SeqLM.from_pretrained(
    #            output_dir,
    #            config=config,
    #        )
    #        return model, tokenizer
    #    else:
    #        raise FileNotFoundError

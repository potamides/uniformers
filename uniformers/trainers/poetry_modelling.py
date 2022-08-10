from collections import ChainMap
from copy import deepcopy
from functools import cached_property
from os.path import isdir, isfile, join
from random import randrange
from typing import Any, Dict, List, Optional, Tuple, Union

from datasets.arrow_dataset import Dataset
from datasets.load import load_metric
from numpy import argmax, asarray, split
from torch import Tensor, nn
from transformers.data.data_collator import DataCollatorForLanguageModeling
from transformers.trainer import Trainer
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import logging

from uniformers.datasets import load_dataset
from uniformers.models.bygpt5 import ByGPT5Tokenizer
from uniformers.utils import (
    ALLITERATION_LEVELS,
    METERS,
    Poetry2Tokens,
    QUATRAIN_RHYME_SCHEMES,
    QuatrainProcessing,
    clean_sentence,
)

from . import LM_LEARNING_RATES
from .training_args import GlobalBatchTrainingArguments
logger = logging.get_logger("transformers")


def _tokenize(examples, p2t, lang, medium, high):
    texts, cats = list(), ("text", "rhyme", "meter", "alliteration")
    for text, rhyme, meter, score in zip(*[examples[cat] for cat in cats]):  # pyright: ignore
        if score < medium:
            alliteration = "low"
        if score < high:
            alliteration = "medium"
        else:
            alliteration = "high"

        rhyme_token = p2t.rhymes2tokens[rhyme]
        meter_token = p2t.meters2tokens[meter]
        allit_token = p2t.alliterations2tokens[alliteration]
        # remove = from sentences (English corpus uses it weirdly)
        clean_text = clean_sentence(text, lang, remove_punct=["="])
        texts.append(f"{rhyme_token}{meter_token}{allit_token}{clean_text}")

    return p2t.tokenizer(texts)

class PoetryLMTrainingArguments(GlobalBatchTrainingArguments):
    def __init__(
        self,
        eval_multiplier=10,
        max_length=1024,
        num_beams=5,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.eval_multiplier = eval_multiplier
        self.max_length = max_length
        self.num_beams = num_beams

class PoetryLMTrainer(Trainer):
    def __init__(
        self,
        model,
        tokenizer,
        output_dir,
        meter_model_name_or_path="nllg/clf-canine-m",
        rhyme_model_name_or_path="nllg/clf-canine-r",
        lang="en",
        # https://github.com/huggingface/transformers/issues/14608#issuecomment-1004390803
        fp16=False,
        bf16=True,
        tf32=False,
        batch_size=512,
        overwrite_output_dir=False,
        # only change below stuff when model doesn't fit into memory (see
        # https://huggingface.co/docs/transformers/performance)
        gradient_accumulation_steps=1,
        gradient_checkpointing=False,
        test_run=False,
        **kwargs,
    ):

        self.model = model
        self.tokenizer = tokenizer
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

        self.metrics = {
            # TODO
        }

        # interesting resource: https://huggingface.co/course/chapter7/6?fw=pt
        self.args = PoetryLMTrainingArguments(
            optim="adamw_torch",
            lr_scheduler_type="cosine",
            learning_rate=LM_LEARNING_RATES[learning_rate],
            num_train_epochs=1 if test_run else 10,
            weight_decay=0.1,
            warmup_ratio=0.01,
            global_train_batch_size=batch_size,
            global_eval_batch_size=batch_size,
            fp16=fp16,
            bf16=bf16,
            tf32=tf32,
            save_total_limit=1,
            overwrite_output_dir=overwrite_output_dir,
            gradient_accumulation_steps=gradient_accumulation_steps,
            gradient_checkpointing=gradient_checkpointing,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            logging_steps=20 if test_run else 250,
            logging_first_step=True,
            output_dir=output_dir,
        )

        train_data, eval_data = self.load_dataset(
            lang,
            meter_model_name_or_path,
            rhyme_model_name_or_path,
            batch_size,
            test_run,
        )

        super().__init__(
            model=self.model,
            tokenizer=self.tokenizer,
            args=self.args,
            data_collator=data_collator,
            train_dataset=train_data, eval_data,  # pyright: ignore
            eval_dataset=eval_data,
            compute_metrics=self.compute_metrics,
            **kwargs,
        )

    def compute_metrics(self, p):
        # TODO
        preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        preds = argmax(preds, axis=1)
        return dict(
            ChainMap(
                *[
                    func(predictions=preds, references=p.label_ids)
                    for func in self.metrics.values()
                ]
            )
        )

    def load_dataset(self, lang, meter_model, rhyme_model, bs, test):
        # deepcopy tokenizer sincd we change some tokenizer attributes
        # without deepcopying this might have side effects
        tokenizer = deepcopy(self.tokenizer)
        if isinstance(tokenizer, ByGPT5Tokenizer):
            tokenizer.add_bos_token = True
            tokenizer.add_eos_token = True
            tokenizer.bos_token = tokenizer.eos_token

        raw_dataset = load_dataset(
            "quatrain", lang=lang, split="train" + ("[:2000]" if test else "")
        )
        dataset = raw_dataset.map(
            QuatrainProcessing(
                lang=lang,
                meter_model_name=meter_model,
                rhyme_model_name=rhyme_model,
                batch_size=bs,
            ),
            batched=True,
        )

        scores = asarray(dataset["alliteration"])  # pyright: ignore
        _, allit_medium, allit_high = split(
            sorted(scores), [round(len(scores) * 0.6), round(len(scores) * 0.9)]
        )
        tokenized_dataset = dataset.map(
            _tokenize,
            batched=True,
            fn_kwargs={  # pyright: ignore
                "lang": lang,
                "p2t": (p2t := Poetry2Tokens(tokenizer)),
                "medium": allit_medium[0],
                "high": allit_high[0],
            },
        )

        index = randrange(len(tokenized_dataset))
        sample = tokenized_dataset[index := randrange(len(tokenized_dataset))][
            "input_ids"
        ]
        detokenized = tokenizer.decode(sample)
        logger.info(f"Sample {index} of the training set: {sample}")
        logger.info(f"Sample {index} of the training set (detokenized): {detokenized}")

        eval_dataset = list()
        # all combinations of rhymes, meters and alliteration levels
        for rhyme in [p2t.rhymes2tokens[rhyme] for rhyme in QUATRAIN_RHYME_SCHEMES]:
            for meter in [p2t.meters2tokens[meter] for meter in METERS]:
                for alliteration in [p2t.alliterations2tokens[allit] for allit in ALLITERATION_LEVELS]:
                    bos = tokenizer.bos_token
                    eval_dataset.extend([bos + rhyme + meter + alliteration] * 1 if test else self.args.eval_multiplier)

        # TODO: check tokenization of eval_dataset
        return tokenized_dataset, Dataset.from_dict(tokenizer(eval_dataset, add_special_tokens=False)) # pyright: ignore

    def train(self, **kwargs):
        if not self.args.overwrite_output_dir and isdir(
            output_dir := self.args.output_dir
        ):
            last_checkpoint = get_last_checkpoint(output_dir)

            if isfile(join(output_dir, "config.json")):
                logger.info(
                    f"Output directory ({output_dir}) exists already and is not empty. Skipping training."
                )
                last_checkpoint = output_dir
            elif last_checkpoint:
                logger.info(
                    f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                    "the `output_dir` or add `overwrite_output_dir` to train from scratch."
                )
        else:
            last_checkpoint = None

        kwargs["resume_from_checkpoint"] = last_checkpoint
        return super().train(**kwargs)

    def prediction_step(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[Tensor], Optional[Tensor], Optional[Tensor]]:

        if prediction_loss_only:
            return super().prediction_step(
                model, inputs, prediction_loss_only=prediction_loss_only, ignore_keys=ignore_keys
            )

        inputs = self._prepare_inputs(inputs)

        gen_kwargs = {
            "max_length": self.args.max_length if self.args.max_length is not None else self.model.config.max_length,
            "num_beams": self.args.num_beams if self.args.num_beams is not None else self.model.config.num_beams,
        }

        if "attention_mask" in inputs:
            gen_kwargs["attention_mask"] = inputs.get("attention_mask")
        if "global_attention_mask" in inputs:
            gen_kwargs["global_attention_mask"] = inputs.get("global_attention_mask")

        # prepare generation inputs
        # some encoder-decoder models can have varying encoder's and thus
        # varying model input names
        if hasattr(self.model, "encoder") and self.model.encoder.main_input_name != self.model.main_input_name:
            generation_inputs = inputs[self.model.encoder.main_input_name]
        else:
            generation_inputs = inputs[self.model.main_input_name]

        generated_tokens = self.model.generate(
            generation_inputs,
            **gen_kwargs,
        )

        return (None, None if self.args.prediction_loss_only else generated_tokens, None)

    @cached_property
    def parameters(self):
        if hasattr(self, "model"):
            return sum(t.numel() for t in self.model.parameters())
        else:
            raise ValueError

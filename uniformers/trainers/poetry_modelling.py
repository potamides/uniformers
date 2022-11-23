from abc import ABC, abstractmethod
from functools import partial
from functools import cached_property
from itertools import combinations
from os.path import isdir, isfile, join
from random import randrange
from re import sub
from typing import Any, Dict, List, Optional, Tuple, Union

from datasets.arrow_dataset import Dataset
from numpy import where
from torch import Tensor, cat, full, nn, tensor
from transformers.data.data_collator import (
    DataCollatorForLanguageModeling,
    DataCollatorForSeq2Seq,
)
from transformers.models.gpt2.tokenization_gpt2 import GPT2Tokenizer
from transformers.models.gpt2.tokenization_gpt2_fast import GPT2TokenizerFast
from transformers.models.t5.tokenization_t5 import T5Tokenizer
from transformers.models.t5.tokenization_t5_fast import T5TokenizerFast
from transformers.trainer import Trainer
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import logging

from uniformers.datasets import load_dataset
from uniformers.metrics import load_metric
from uniformers.models.bygpt5 import ByGPT5Tokenizer
from uniformers.utils import (
    ALLITERATION_LEVELS,
    EMOTIONS,
    EmotionProcessing,
    METERS,
    Poetry2Tokens,
    QUATRAIN_RHYME_SCHEMES,
    QuatrainProcessing,
    clean_sentence,
)

from . import LM_LEARNING_RATES
from .training_args import GlobalBatchTrainingArguments

logger = logging.get_logger("transformers")

def _add_special_tokens(tokenizer, texts):
    bos, eos = tokenizer.bos_token, tokenizer.eos_token
    return [bos + text + eos for text in texts]


def _tokenize(examples, p2t, lang, medium, high, is_encoder_decoder=False):
    inputs, labels, cats = list(), list(), ("text", "rhyme", "meter", "alliteration")
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
        clean_text = "\n".join(clean_sentence(verse, lang, remove_punct=["="]) for verse in text)  # pyright: ignore

        inputs.append(rhyme_token + meter_token + allit_token)
        labels.append(clean_text)

    if is_encoder_decoder:
        model_inputs = p2t.tokenizer(inputs, add_special_tokens=False)
        labels = p2t.tokenizer(labels)
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs
    else:
        return p2t.tokenizer(_add_special_tokens(p2t.tokenizer, [i + l for i, l in zip(inputs, labels)]))


def _emotion_tokenize(examples, lang, tokenizer, e2t, is_encoder_decoder=False):
    inputs, labels = list(), list()
    for text, emotion in zip(examples["text"], examples["emotion"]):
        clean_text = "\n".join(clean_sentence(verse, lang, remove_punct=["="]) for verse in text)  # pyright: ignore
        inputs.append("".join(e2t[e] for e in sorted(emotion)))
        labels.append(clean_text)

    if is_encoder_decoder:
        model_inputs = tokenizer(inputs, add_special_tokens=False)
        labels = tokenizer(labels)
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs
    else:
        return tokenizer(_add_special_tokens(tokenizer, [i + l for i, l in zip(inputs, labels)]))


class PoetryLMTrainingArguments(GlobalBatchTrainingArguments):
    def __init__(
        self,
        eval_multiplier=75,
        max_length=384,
        num_beams=1,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.eval_multiplier = eval_multiplier
        self.max_length = max_length
        self.num_beams = num_beams


class AbstractPoetryLMTrainer(Trainer, ABC):
    @abstractmethod
    def __init__(
        self,
        model,
        tokenizer,
        eval_multiplier,
        output_dir,
        # https://github.com/huggingface/transformers/issues/14608#issuecomment-1004390803
        fp16=False,
        bf16=True,
        tf32=False,
        batch_size=128,
        overwrite_output_dir=False,
        # only change below stuff when model doesn't fit into memory (see
        # https://huggingface.co/docs/transformers/performance)
        gradient_accumulation_steps=1,
        gradient_checkpointing=False,
        test_run=False,
        **trainer_args
    ):

        self.model = model
        self.tokenizer = tokenizer
        self.trainer_args = trainer_args
        self.patch_tokenizer()

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
        self.args = PoetryLMTrainingArguments(
            optim="adamw_torch",
            lr_scheduler_type="cosine",
            learning_rate=LM_LEARNING_RATES[learning_rate],
            num_train_epochs=1 if test_run else 10,
            weight_decay=0.1,
            warmup_ratio=0.01,
            eval_multiplier=eval_multiplier,
            global_train_batch_size=batch_size,
            global_eval_batch_size=batch_size,
            fp16=fp16,
            bf16=bf16,
            tf32=tf32,
            save_total_limit=1,
            overwrite_output_dir=overwrite_output_dir,
            gradient_accumulation_steps=gradient_accumulation_steps,
            gradient_checkpointing=gradient_checkpointing,
            ddp_find_unused_parameters=False,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            logging_steps=20 if test_run else 250,
            logging_first_step=True,
            output_dir=output_dir,
        )

    @abstractmethod
    def compute_metrics(self, p):
        preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        # replace dynamic padding index with true padding index
        preds = self.decode(where(preds == -100, self.tokenizer.pad_token_id, preds), batch=True)
        # remove text after eos token (if token exists)
        preds = [pred[:pred.find(self.tokenizer.eos_token, len(self.tokenizer.eos_token))] for pred in preds]

        return preds

    @abstractmethod
    def patch_tokenizer(self):
        if isinstance(self.tokenizer, ByGPT5Tokenizer):
            self.tokenizer.add_bos_token = False
            self.tokenizer.add_eos_token = False
            self.tokenizer.bos_token = self.tokenizer.eos_token
        elif isinstance(self.tokenizer, (GPT2Tokenizer, GPT2TokenizerFast)):
            self.tokenizer.add_bos_token = False # pyright: ignore
        elif isinstance(self.tokenizer, (T5Tokenizer, T5TokenizerFast)):
            # sentencepiece replaces \n, so we need our own symbol (e.g., as sep_token)
            self.tokenizer.add_special_tokens({"sep_token": "\n"}) # pyright: ignore
            self.model.resize_token_embeddings(len(self.tokenizer))

    @abstractmethod
    def load_dataset(self):
        pass

    def decode(self, ids, batch=False):
        decoded = self.tokenizer.batch_decode(ids) if batch else [self.tokenizer.decode(ids)]
        # remove spaces sentencepiece adds around newline
        for idx, sent in enumerate(decoded):
            decoded[idx] = sub(r"\s*\n\s*", "\n", sent)
        return decoded if batch else decoded[0]

    def train(self, **kwargs):
        if not self.args.overwrite_output_dir and isdir(output_dir := self.args.output_dir):
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

    def test(self, save_metrics=True):
        logger.info("Testing model.")
        ds = self.eval_dataset
        metrics = self.evaluate(eval_dataset=ds)

        metrics = {key.replace("eval", "test"): value for key, value in metrics.items()}
        self.log_metrics('test', metrics)
        if save_metrics:
            self.save_metrics('test', metrics, False)

    def prediction_step(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[Tensor], Optional[Tensor], Optional[Tensor]]:

        if prediction_loss_only or self.args.prediction_loss_only:
            return super().prediction_step(
                model,
                inputs,
                prediction_loss_only=prediction_loss_only,
                ignore_keys=ignore_keys,
            )

        inputs = self._prepare_inputs(inputs)

        gen_kwargs = {
            "bad_words_ids": [[id_] for id_ in self.tokenizer.additional_special_tokens_ids],
            "max_length": self.args.max_length
            if self.args.max_length is not None
            else self.model.config.max_length,
            "num_beams": self.args.num_beams
            if self.args.num_beams is not None
            else self.model.config.num_beams,
            "do_sample": True,
            "temperature": 0.7,
            "top_p": 0.9,
            "top_k": 0
        }

        if "attention_mask" in inputs:
            gen_kwargs["attention_mask"] = inputs.get("attention_mask")
        if "global_attention_mask" in inputs:
            gen_kwargs["global_attention_mask"] = inputs.get("global_attention_mask")

        # prepare generation inputs
        # some encoder-decoder models can have varying encoder's and thus
        # varying model input names
        if (
            hasattr(self.model, "encoder")
            and self.model.encoder.main_input_name != self.model.main_input_name
        ):
            generation_inputs = inputs[self.model.encoder.main_input_name]
        else:
            generation_inputs = inputs[self.model.main_input_name]

        generated_tokens = self.model.generate(
            generation_inputs,
            **gen_kwargs,
        )

        if self.model.config.is_encoder_decoder:
            # add a start tensor so that eval code works (it expects a bos_token)
            start_tensor = full((len(generation_inputs), 1), self.model.config.decoder_start_token_id, device=self.args.device)
            # and remove start-of-sequence token
            generated_tokens = cat((start_tensor, generation_inputs, generated_tokens[:, 1:]), dim=1)
            pass

        return (None, generated_tokens, tensor([], device=self.args.device))

    @cached_property
    def parameters(self):
        if hasattr(self, "model"):
            return sum(t.numel() for t in self.model.parameters())
        else:
            raise ValueError


class PoetryLMTrainer(AbstractPoetryLMTrainer):
    def __init__(
        self,
        model,
        meter_model_name_or_path="nllg/clf-canine-m",
        rhyme_model_name_or_path="nllg/clf-canine-r",
        coherence_model_name_or_path="bert-base-multilingual-cased",
        lang="en",
        batch_size=128,
        test_run=False,
        low_resource=False,
        **kwargs,
    ):
        super().__init__(
            model=model,
            batch_size=batch_size,
            test_run=test_run,
            eval_multiplier=5 if test_run else 75,
            **kwargs,
        )

        if model.config.is_encoder_decoder:
            data_collator = DataCollatorForSeq2Seq(self.tokenizer, self.model)
        else:
            data_collator = DataCollatorForLanguageModeling(self.tokenizer, mlm=False)

        train_data, tokenized_data, eval_data, medium, high = self.load_dataset(
            lang,
            meter_model_name_or_path,
            rhyme_model_name_or_path,
            batch_size,
            test_run,
            low_resource
        )

        super(AbstractPoetryLMTrainer, self).__init__(
            model=self.model,
            tokenizer=self.tokenizer,
            args=self.args,
            data_collator=data_collator,
            train_dataset=tokenized_data,  # pyright: ignore
            eval_dataset=eval_data,  # pyright: ignore
            compute_metrics=partial(
                self.compute_metrics,
                train_data,
                lang,
                medium,
                high,
                meter_model_name_or_path,
                rhyme_model_name_or_path,
                coherence_model_name_or_path,
                batch_size,
            ),
            **self.trainer_args,
        )

    def compute_metrics(self, train_data, lang, medium, high, meter_model, rhyme_model, coherence_model, bs, p):
        preds = super().compute_metrics(p)
        p2t = Poetry2Tokens(self.tokenizer)
        rhymes, meters, allits = list(), list(), list()

        for idx, pred in enumerate(preds):
            tokenized = self.tokenizer.tokenize(pred)
            rhymes.append(p2t.tokens2forms[tokenized[1]])
            meters.append(p2t.tokens2forms[tokenized[2]])
            allits.append(p2t.tokens2forms[tokenized[3]])
            preds[idx] = self.tokenizer.convert_tokens_to_string(tokenized[4:])

        rhyme_scores = load_metric(
            "rhyme", language=lang, batch_size=bs, model_name=rhyme_model
        ).compute(quatrains=preds, schemes=rhymes)
        meter_scores = load_metric(
            "meter", language=lang, batch_size=bs, model_name=meter_model
        ).compute(quatrains=preds, meters=meters)
        allit_scores = load_metric(
            "alliteration", language=lang, batch_size=bs, medium=medium, high=high
        ).compute(quatrains=preds, levels=allits)
        coherence_scores = load_metric(
            "coherence", batch_size=bs, model_name=coherence_model
        ).compute(quatrains=preds)
        copying_scores = load_metric(
            "copying", train_data=train_data,
        ).compute(quatrains=preds, schemes=rhymes, meters=meters, levels=allits)

        return rhyme_scores | meter_scores | allit_scores | coherence_scores  | copying_scores # pyright: ignore

    def patch_tokenizer(self):
        super().patch_tokenizer()
        if not self.tokenizer.additional_special_tokens:
            num_special = len(ALLITERATION_LEVELS + METERS + QUATRAIN_RHYME_SCHEMES)
            special = {
                "additional_special_tokens": [f"<extra_id_{idx}>" for idx in range(num_special)],
                'pad_token': '<pad>'
            }
            self.tokenizer.add_special_tokens(special) # pyright: ignore
            self.model.resize_token_embeddings(len(self.tokenizer))

    def load_dataset(self, lang, meter_model, rhyme_model, bs, test, low_res):
        raw_dataset = load_dataset(
            "quatrain", lang=lang, split="train" + ("[:20000]" if test else "")
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

        tokenized_dataset = dataset.map(
            _tokenize,
            batched=True,
            fn_kwargs={  # pyright: ignore
                "lang": lang,
                "p2t": (p2t := Poetry2Tokens(self.tokenizer)),
                "medium": (medium := 0.05),
                "high": (high := 0.1),
                "is_encoder_decoder": self.model.config.is_encoder_decoder,
            },
        )

        if low_res: # use only 1/20 of training dataset
            tokenized_dataset, _ = tokenized_dataset.train_test_split(test_size=0.95).values() # pyright: ignore

        index = randrange(len(tokenized_dataset))
        sample = tokenized_dataset[index]
        detokenized = self.decode(sample["input_ids"])
        logger.info(f"Input sample {index} of the training set: {sample['input_ids']}")
        logger.info(f"Input sample {index} of the training set (detokenized): {detokenized}")
        if "labels" in sample: # pyright: ignore
            detokenized = self.decode(sample["labels"])
            logger.info(f"Label sample {index} of the training set: {sample['labels']}")
            logger.info(f"Label sample {index} of the training set (detokenized): {detokenized}")

        eval_dataset = list()
        # evaluate on common combinations rhymes, meters and alliteration levels
        for rhyme in [p2t.rhymes2tokens[rhyme] for rhyme in ["AABB", "ABCB", "ABAB", "ABBA"]]:
            meters = ["alexandrine", "iambus", "trochee"] if lang == "de" else ["anapaest", "dactyl", "iambus", "trochee"]
            for meter in [p2t.meters2tokens[meter] for meter in meters]:
                for alliteration in [p2t.alliterations2tokens[allit] for allit in ALLITERATION_LEVELS ]:
                    bos = "" if self.model.config.is_encoder_decoder else self.tokenizer.bos_token
                    eval_dataset.extend(
                        [bos + rhyme + meter + alliteration] * self.args.eval_multiplier
                    )
            if test:
                break
        tokenized_eval_dataset = Dataset.from_dict(self.tokenizer(eval_dataset, add_special_tokens=False))  # pyright: ignore

        return dataset, tokenized_dataset, tokenized_eval_dataset, medium, high


class PoetryEmotionLMTrainer(AbstractPoetryLMTrainer):
    def __init__(
        self,
        model,
        emotion_model_name_or_path="nllg/clf-bert-e", # German only!
        lang="de",
        batch_size=128,
        test_run=False,
        low_resource=False,
        **kwargs,
    ):

        super().__init__(
            model=model,
            batch_size=batch_size,
            test_run=test_run,
            eval_multiplier=5 if test_run else 100,
            **kwargs,
        )

        if model.config.is_encoder_decoder:
            data_collator = DataCollatorForSeq2Seq(self.tokenizer, self.model)
        else:
            data_collator = DataCollatorForLanguageModeling(self.tokenizer, mlm=False)

        self.emotions2tokens = dict(zip(EMOTIONS, self.tokenizer.additional_special_tokens))
        self.tokens2emotions = {v: k for k, v in self.emotions2tokens.items()}

        train_data, eval_data = self.load_dataset(
            lang,
            emotion_model_name_or_path,
            batch_size,
            test_run,
            low_resource
        )

        super(AbstractPoetryLMTrainer, self).__init__(
            model=self.model,
            tokenizer=self.tokenizer,
            args=self.args,
            data_collator=data_collator,
            train_dataset=train_data,  # pyright: ignore
            eval_dataset=eval_data,  # pyright: ignore
            compute_metrics=partial(
                self.compute_metrics,
                lang,
                emotion_model_name_or_path,
                batch_size,
            ),
            **self.trainer_args,
        )

    def compute_metrics(self, lang, emotion_model, bs, p):
        preds = super().compute_metrics(p)

        emotions = list()
        for idx, pred in enumerate(preds):
            tokenized, emotion = self.tokenizer.tokenize(pred)[1:], list()
            for token in tokenized:
                if token in self.tokens2emotions:
                    emotion.append(self.tokens2emotions[token])
                else:
                    break
            emotions.append(emotion)
            preds[idx] = self.tokenizer.convert_tokens_to_string(tokenized[len(emotion):])

        emotion_scores = load_metric(
            "emotion", language=lang, batch_size=bs, model_name=emotion_model
        ).compute(quatrains=preds, emotions=emotions)

        return emotion_scores or {}

    def patch_tokenizer(self):
        super().patch_tokenizer()
        if not self.tokenizer.additional_special_tokens:
            special = {
                "additional_special_tokens": [f"<extra_id_{idx}>" for idx in range(len(EMOTIONS))],
                'pad_token': '<pad>'
            }
            self.tokenizer.add_special_tokens(special) # pyright: ignore
            self.model.resize_token_embeddings(len(self.tokenizer))

    def load_dataset(self, lang, emotion_model, bs, test, low_res):
        raw_dataset = load_dataset(
            "quatrain", lang=lang, split="train" + ("[:20000]" if test else "")
        )
        dataset = raw_dataset.map(
            EmotionProcessing(
                lang=lang,
                emotion_model_name=emotion_model,
                batch_size=bs,
            ),
            batched=True,
        )

        tokenized_dataset = dataset.map(
            _emotion_tokenize,
            batched=True,
            fn_kwargs={  # pyright: ignore
                "lang": lang,
                "tokenizer": self.tokenizer,
                "e2t": self.emotions2tokens,
                "is_encoder_decoder": self.model.config.is_encoder_decoder,
            },
        )

        if low_res: # use only 1/20 of training dataset
            tokenized_dataset, _ = tokenized_dataset.train_test_split(test_size=0.95).values() # pyright: ignore

        index = randrange(len(tokenized_dataset))
        sample = tokenized_dataset[index]
        detokenized = self.decode(sample["input_ids"])
        logger.info(f"Input sample {index} of the training set: {sample['input_ids']}")
        logger.info(f"Input sample {index} of the training set (detokenized): {detokenized}")
        if "labels" in sample: # pyright: ignore
            detokenized = self.decode(sample["labels"])
            logger.info(f"Label sample {index} of the training set: {sample['labels']}")
            logger.info(f"Label sample {index} of the training set (detokenized): {detokenized}")

        eval_dataset = list()
        for emotions in sorted(set(combinations(EMOTIONS, 2))):
            bos = "" if self.model.config.is_encoder_decoder else self.tokenizer.bos_token
            eval_dataset.extend([bos + "".join(self.emotions2tokens[emo] for emo in sorted(emotions))] * self.args.eval_multiplier)

        tokenized_eval_dataset = Dataset.from_dict(self.tokenizer(eval_dataset, add_special_tokens=False))  # pyright: ignore

        return tokenized_dataset, tokenized_eval_dataset

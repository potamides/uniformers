from collections import ChainMap
from functools import cached_property, partial
from os.path import isdir, isfile, join
from random import randrange

from datasets import Value
from datasets.load import load_metric
from numpy import argmax, where, zeros
from optuna.samplers import GridSampler
from torch import Tensor, sigmoid
from transformers.models.auto.configuration_auto import AutoConfig
from transformers.models.auto.modeling_auto import AutoModelForSequenceClassification
from transformers.trainer import Trainer
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import logging

from uniformers.datasets import load_dataset
from uniformers.utils import clean_sentence

from .training_args import GlobalBatchTrainingArguments

logger = logging.get_logger("transformers")


def _preprocess_data(examples, tokenizer, label_info):
    if type(examples['text'][0]) == str:
        logger.debug("Tokenizing single sentences.")
        tokenized = tokenizer([clean_sentence(t, l) for t, l in zip(examples['text'], examples['language'])], truncation=True)
    else:
        logger.debug("Tokenizing sentence pairs.")
        sentences1, sentences2, languages = *zip(*examples["text"]), examples['language']
        sentences1 = [clean_sentence(s, l) for s, l in zip(sentences1, languages)]
        sentences2 = [clean_sentence(s, l) for s, l in zip(sentences2, languages)]
        tokenized = tokenizer(sentences1, sentences2, truncation=True)

    if isinstance(examples["labels"][0], list):
        logger.debug("Transforming labels for multi-label classification.")
        labels_matrix = zeros((len(examples['labels']), label_info.feature.num_classes))
        for idx, labels in enumerate(examples['labels']):
          labels_matrix[idx, labels] = 1
        tokenized["labels"] = labels_matrix.tolist()

    return tokenized


class PoetryClassificationTrainer(Trainer):
    def __init__(
        self,
        model_init,
        tokenizer,
        output_dir,
        task="meter",
        # https://github.com/huggingface/transformers/issues/14608#issuecomment-1004390803
        fp16=True,
        bf16=False,
        tf32=False,
        overwrite_output_dir=False,
        # only change below stuff when model doesn't fit into memory (see
        # https://huggingface.co/docs/transformers/performance)
        gradient_accumulation_steps=1,
        gradient_checkpointing=False,
        test_run=False,
        **kwargs,
    ):
        assert task in ["meter", "rhyme", "emotion"]
        self.tokenizer = tokenizer
        self.metrics = {
            # FIXME: uses _compute instead of compute since the latter doesn't
            # work with multi-label classification. should use sklearn directly
            # instead
            "precision": partial(load_metric("precision")._compute, average="macro", zero_division=0),
            "recall": partial(load_metric("recall")._compute, average="macro", zero_division=0),
            "f1": partial(load_metric("f1")._compute, average="macro"),
            "accuracy": load_metric("accuracy")._compute
        }

        # interesting resource: https://huggingface.co/course/chapter7/6?fw=pt
        self.args = GlobalBatchTrainingArguments(
            optim="adamw_torch",
            lr_scheduler_type="cosine",
            learning_rate=10e-6,
            num_train_epochs=1 if test_run else 100 if task == "meter" else 10,
            weight_decay=0.001,
            warmup_ratio=0.1,
            global_train_batch_size=8,
            global_eval_batch_size=8,
            fp16=fp16,
            bf16=bf16,
            tf32=tf32,
            save_total_limit=1,
            overwrite_output_dir=overwrite_output_dir,
            gradient_accumulation_steps=gradient_accumulation_steps,
            gradient_checkpointing=gradient_checkpointing,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            logging_steps=250,
            logging_first_step=True,
            output_dir=output_dir,
        )

        train_dataset, eval_dataset, self.test_dataset = self.load_dataset(task)

        super().__init__(
            model_init=model_init,
            tokenizer=self.tokenizer,
            args=self.args,
            train_dataset=train_dataset,  # pyright: ignore
            eval_dataset=eval_dataset,  # pyright: ignore
            compute_metrics=self.compute_metrics,
            **kwargs,
        )

    def compute_metrics(self, p, threshold=0.5):
        probs = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        if self.model.config.problem_type == "multi_label_classification":
            probs = sigmoid(Tensor(probs))
            preds = zeros(probs.shape)
            preds[where(probs >= threshold)] = 1
        else:
            preds = argmax(probs, axis=1)
        return dict(ChainMap(*[func(predictions=preds, references=p.label_ids) for func in self.metrics.values()]))

    def load_dataset(self, task):
        self._num_samples = 0
        if task == "emotion":
            # FIXME: proper class option for single or multilingual training
            raw_dataset = load_dataset("poemo", lang="de", split="train")
        else:
            raw_dataset = load_dataset("poetrain", task, split="train")

        tokenized_dataset = raw_dataset.map(
            _preprocess_data,
            batched=True,
            #remove_columns=raw_dataset.column_names,  # pyright: ignore
            fn_kwargs = {  # pyright: ignore
                "tokenizer": self.tokenizer,
                "label_info": raw_dataset.features['labels'] # pyright: ignore
                }
        )

        if task == "emotion": # multi-label classification needs float labels
            new_features = tokenized_dataset.features.copy() # pyright: ignore
            new_features['labels'].feature = Value("float64")
            tokenized_dataset = tokenized_dataset.cast(new_features)

        train_dataset, tmp_dataset = tokenized_dataset.train_test_split( # pyright: ignore
            test_size=0.1, stratify_by_column=None if task == "emotion" else "labels"
        ).values()
        eval_dataset, test_dataset = tmp_dataset.train_test_split(
            test_size=0.5, stratify_by_column=None if task == "emotion" else "labels"
        ).values()

        index = randrange(len(train_dataset))
        sample = train_dataset[index := randrange(len(train_dataset))]['input_ids'] # pyright: ignore
        detokenized = self.tokenizer.decode(sample)
        logger.info(f"Sample {index} of the training set: {sample}")
        logger.info(f"Sample {index} of the training set (detokenized): {detokenized}")
        return train_dataset, eval_dataset, test_dataset

    def grid_search(
        self, search_space={"learning_rate": [1e-6, 5e-6, 1e-5, 5e-5, 1e-4]}
    ):
        sampler = GridSampler(search_space)
        hp_space = lambda t: {k: t.suggest_float(k, min(v), max(v)) for k, v in search_space.items() }
        best_run = self.hyperparameter_search(
            direction="maximize",
            hp_space=hp_space,
            sampler=sampler,
            compute_objective=lambda m: m["eval_f1"], # pyright: ignore
        )
        self._load_run(best_run)

    def train(self, **kwargs):
        if "trial" in kwargs:
            output_dir = join(self.args.output_dir, f"run-{kwargs['trial'].number}")
        else:
            output_dir = self.args.output_dir
        if not self.args.overwrite_output_dir and isdir(output_dir):
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

        kwargs['resume_from_checkpoint'] = last_checkpoint
        return super().train(**kwargs)

    def _load_run(self, run):
        self.state.trial_params = run.hyperparameters
        path = get_last_checkpoint(join(self.args.output_dir, f"run-{run.run_id}"))
        config = AutoConfig.from_pretrained(path)
        self.model = AutoModelForSequenceClassification.from_pretrained(path, config=config).to(self.args.device)

    def test(self, save_metrics=True):
        logger.info("Testing model.")
        ds = self.test_dataset
        all_metrics = self.evaluate(eval_dataset=ds)

        lang_metrics = dict()
        if len(langs := sorted(set(ds['language']))) > 1:
            for lang in langs:
                lang_metrics[lang] = self.evaluate(eval_dataset=ds.filter(lambda example: example["language"] == lang))

        for name, metrics in zip(["test"] + [f"test-{l}" for l in lang_metrics.keys()], [all_metrics] + list(lang_metrics.values())):
            metrics = {key.replace("eval", "test"): value for key, value in metrics.items()}
            self.log_metrics(name, metrics)
            if save_metrics:
                self.save_metrics(name, metrics, False)

    @cached_property
    def parameters(self):
        if hasattr(self, "model"):
            return sum(t.numel() for t in self.model.parameters())

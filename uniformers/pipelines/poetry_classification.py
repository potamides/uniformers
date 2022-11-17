#!/usr/bin/env python

from abc import ABC, abstractmethod
from typing import List, Tuple

from torch import device
from torch.cuda import is_available as has_cuda
from transformers.models.auto.modeling_auto import AutoModelForSequenceClassification
from transformers.models.auto.tokenization_auto import AutoTokenizer
from transformers.pipelines.text_classification import TextClassificationPipeline

from uniformers.utils.poetry import METERS, RHYME_LABELS, EMOTIONS
from uniformers.utils.clean import clean_sentence


class _AbstractPoetryClassificationPipeline(TextClassificationPipeline, ABC):
    @abstractmethod
    def __init__(
        self,
        model_name,
        lang,
        labels,
        device=device("cuda:0" if has_cuda() else "cpu"),
        batch_size=1,
        **kwargs,
    ):
        self.lang = lang
        id2label = {idx: label for idx, label in enumerate(labels)}
        label2id = dict(zip(id2label.values(), id2label.keys()))
        super().__init__(
            model=AutoModelForSequenceClassification.from_pretrained(
                model_name, label2id=label2id, id2label=id2label
            ),
            tokenizer=AutoTokenizer.from_pretrained(model_name),
            batch_size=batch_size,
            device=device,
            truncation=True,
            padding=True,
            **kwargs,
        )

    def __call__(self, *args, **kwargs):
        return self.classify(*args, **kwargs)

    @abstractmethod
    def classify(self, *args, **kwargs):
        self.call_count = 0 # hotfix for false-positive warning
        return super().__call__(*args, **kwargs)


class RhymeClassificationPipeline(_AbstractPoetryClassificationPipeline):
    def __init__(self, lang, model_name="nllg/clf-canine-r", **kwargs):
        super().__init__(
            lang=lang, model_name=model_name, labels=RHYME_LABELS, **kwargs
        )

    def classify(self, pairs: Tuple[str, str] | List[Tuple[str, str]]):
        args = list()
        for verse1, verse2 in (
            [pairs] if len(pairs) and isinstance(pairs[0], str) else pairs
        ):
            # during training tuples where sorted so we do it here as well but
            # it probably isn't that important
            verse1, verse2 = sorted(
                (
                    clean_sentence(verse1, lang=self.lang),
                    clean_sentence(verse2, lang=self.lang),
                )
            )
            args.append({"text": verse1, "text_pair": verse2})

        return super().classify(args)


class MeterClassificationPipeline(_AbstractPoetryClassificationPipeline):
    def __init__(self, lang, model_name="nllg/clf-canine-m", **kwargs):
        super().__init__(lang=lang, model_name=model_name, labels=METERS, **kwargs)

    def classify(self, sents: str | List[str]):
        cleaned = [
            clean_sentence(verse, lang=self.lang)
            for verse in ([sents] if isinstance(sents, str) else sents)
        ]
        return super().classify(cleaned)

class EmotionClassificationPipeline(_AbstractPoetryClassificationPipeline):
    def __init__(self, lang="de", model_name="nllg/clf-bert-e", threshold=0.5, **kwargs):
        super().__init__(lang=lang, model_name=model_name, labels=EMOTIONS, top_k=None, **kwargs)
        self.threshold = threshold

    def classify(self, sents: str | List[str]):
        cleaned = [
            clean_sentence(verse, lang=self.lang)
            for verse in ([sents] if isinstance(sents, str) else sents)
        ]
        classified_sents = super().classify(cleaned)
        for classified_sent in classified_sents:
            for label in classified_sent:
                label['predicted'] = label['score'] >= self.threshold

        return classified_sents

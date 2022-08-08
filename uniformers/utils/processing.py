from collections import Counter
from functools import cached_property
from itertools import chain, combinations
from statistics import mean

from uniformers.pipelines import (
    MeterClassificationPipeline,
    RhymeClassificationPipeline,
)
from uniformers.utils import Phonemizer
from uniformers.utils import alliteration_score, meter_to_label, scheme_to_label


def process_alliterations(examples, phonemizer):
    unique = list(set(chain.from_iterable(examples["text"])))
    verse2phoneme = dict(zip(unique, phonemizer(unique)))

    scores = list()
    for quatrain in examples["text"]:
        phonemes = [verse2phoneme[verse] for verse in quatrain]
        scores.append(mean(alliteration_score(verse) for verse in phonemes))

    examples["alliteration"] = scores
    return examples


def process_meters(examples, clf_meter):
    unique = list(set(chain.from_iterable(examples["text"])))
    verse2meter = dict(zip(unique, clf_meter(unique)))

    all_meters = list()
    for quatrain in examples["text"]:
        meters = Counter(verse2meter[verse]["label"] for verse in quatrain)
        meter, count = meters.most_common(1)[0]
        # at least half of the verses must match, else we map quatrain to 'other'
        if count >= 2:
            all_meters.append(meter_to_label(meter))
        else:
            all_meters.append(meter_to_label("other"))

    examples["meter"] = all_meters
    return examples


def process_rhymes(examples, clf_rhyme):
    verse_pairs = [list(combinations(quatrain, r=2)) for quatrain in examples["text"]]
    unique = list(set(chain.from_iterable(verse_pairs)))
    pair2rhyme = dict(zip(unique, clf_rhyme(unique)))

    schemes = list()
    for idx, quatrain in reversed(list(enumerate(examples["text"]))):
        scheme = list("ABCD")
        for (idx1, verse1), (idx2, verse2) in combinations(enumerate(quatrain), r=2):
            if pair2rhyme[(verse1, verse2)]["label"] == "rhyme":
                # when transitivity doesn't hold (e.g. A rhymes with B, B
                # rhymes with C but A does not rhyme with C), skip
                if scheme[idx1] != "ABCD"[idx1] and scheme[idx2] != scheme[idx1]:
                    for value in examples.values():
                        value.pop(idx)
                    break
                scheme[idx2] = scheme[idx1]
        else:
            schemes.insert(0, scheme_to_label(*scheme))
            pass

    examples["rhyme"] = schemes
    return examples


class QuatrainProcessing:
    def __init__(self, lang, meter_model_name, rhyme_model_name, batch_size=1):
        self.lang = lang
        self.meter = meter_model_name
        self.rhyme = rhyme_model_name
        self.bs = batch_size

    @cached_property
    def clf_meter(self):
        return MeterClassificationPipeline(
            lang=self.lang, batch_size=self.bs, model_name=self.meter
        )

    @cached_property
    def clf_rhyme(self):
        return RhymeClassificationPipeline(
            lang=self.lang, batch_size=self.bs, model_name=self.rhyme
        )

    @cached_property
    def phonemizer(self):
        return Phonemizer(lang=self.lang, batch_size=self.bs)

    def __call__(self, examples):
        examples = process_rhymes(examples, self.clf_rhyme)
        examples = process_meters(examples, self.clf_meter)
        examples = process_alliterations(examples, self.phonemizer)
        return examples

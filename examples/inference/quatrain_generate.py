#!/usr/bin/env python

from argparse import ArgumentParser
from enum import Enum
from itertools import combinations
from statistics import mean

from termcolor import colored
from torch import device
from transformers.models.auto.configuration_auto import CONFIG_MAPPING
from transformers.models.auto.modeling_auto import AutoModelForCausalLM
from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING
from transformers.models.auto.tokenization_auto import TOKENIZER_MAPPING
from transformers.models.auto.tokenization_auto import AutoTokenizer
from transformers.models.t5.configuration_t5 import T5Config
from transformers.pipelines.text_generation import TextGenerationPipeline
from transformers.utils.logging import log_levels, set_verbosity

from uniformers.models.bygpt5 import ByGPT5Config, ByGPT5LMHeadModel, ByGPT5Tokenizer
from uniformers.pipelines import (
    MeterClassificationPipeline,
    RhymeClassificationPipeline,
)
from uniformers.utils import (
    ALLITERATION_LEVELS,
    METERS,
    Phonemizer,
    Poetry2Tokens,
    QUATRAIN_RHYME_SCHEMES,
    alliteration_score,
    scheme_to_label,
)

# we need to add this to to be able to use ByGPT5 with AutoModel
CONFIG_MAPPING.register(ByGPT5Config.model_type, ByGPT5Config)
TOKENIZER_MAPPING.register(ByGPT5Config, (ByGPT5Tokenizer, None))
MODEL_FOR_CAUSAL_LM_MAPPING.register(ByGPT5Config, ByGPT5LMHeadModel)
MODEL_FOR_CAUSAL_LM_MAPPING.register(T5Config, ByGPT5LMHeadModel)


class Match(str, Enum):
    GOOD = "blue"
    OK = "yellow"
    BAD = "red"

def identify(prompt, model_name, lang):
    # quickfix for this annoyance https://github.com/huggingface/transformers/pull/12024#pullrequestreview-694970383
    set_verbosity(log_levels['critical'])
    clf_meter = MeterClassificationPipeline(lang=lang, model_name=model_name)
    set_verbosity(log_levels['warning'])
    return clf_meter(prompt)[0]['label']

def generate(model_name, rhyme, meter, alliteration, dev="cpu", verse=None, num_samples=5, top_p=0.9, temperature=0.6, max_length=384):
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    p2t = Poetry2Tokens(tokenizer)

    pipeline = TextGenerationPipeline(
        model=model,
        tokenizer=tokenizer,
        device=device(dev),
    )
    prompt = (
        tokenizer.bos_token
        + p2t.rhymes2tokens[rhyme]
        + p2t.meters2tokens[meter]
        + p2t.alliterations2tokens[alliteration]
        + (f"{verse}\n" if verse else "")
    )

    return [(f"{verse}\n" if verse else "") + quatrain['generated_text'] for quatrain in pipeline(
        prompt,
        return_full_text=False,
        bad_words_ids=[[id_] for id_ in tokenizer.additional_special_tokens_ids],
        max_length=max_length,
        num_return_sequences=num_samples,
        do_sample=True,
        top_k=0,
        temperature=temperature,
        top_p=top_p,
    )]

def classify(quatrains, rhyme, meter, level, rhyme_model_name, meter_model_name, lang):
    schemes, meters, scores = list(), list(), list()
    set_verbosity(log_levels['critical'])
    clf_meter = MeterClassificationPipeline(lang=lang, model_name=meter_model_name, top_k=None)
    clf_rhyme = RhymeClassificationPipeline(lang=lang, model_name=rhyme_model_name)
    phonemizer = Phonemizer(lang=lang)
    set_verbosity(log_levels['warning'])

    for quatrain in quatrains:
        scheme = list("ABCD")
        combs = list(combinations(range(len(quatrain)), r=2))
        # evaluate pairs that should rhyme first, this helps when transitivity doesn't hold
        rhyme_combs = filter(lambda comb: rhyme[comb[0]] == rhyme[comb[1]], combs)
        norhyme_combs = filter(lambda comb: rhyme[comb[0]] != rhyme[comb[1]], combs)

        for idx1, idx2 in list(rhyme_combs) + list(reversed(list(norhyme_combs))):
            verse1, verses2 = quatrain[idx1], [verse for idx, verse in enumerate(quatrain) if scheme[idx] == scheme[idx2]]
            if scheme[idx1] == "ABCD"[idx1] and all(clf_rhyme((verse1, verse2))[0]["label"] == "rhyme" for verse2 in verses2):
                scheme = [scheme[idx1] if val == scheme[idx2] else val for val in scheme]

        meters_ = list()
        for prob_meter in clf_meter(quatrain):
            if prob_meter[0]['label'] == meter:
                meters_.append((meter, Match.GOOD))
            # if probability is higher than for uniform distribution we still count it
            elif list(filter(lambda res: res['label'] == meter, prob_meter))[0]['score'] >= 1/len(METERS):
                meters_.append((meter, Match.OK))
            else:
                meters_.append((prob_meter[0]['label'], Match.BAD))

        score = mean([alliteration_score(verse) for verse in phonemizer(quatrain)])
        if score < 0.05:
            scores.append(("low", {"low": Match.GOOD, "medium": Match.OK, "high": Match.BAD}[level]))
        elif score < 0.1:
            scores.append(("medium", {"low": Match.OK, "medium": Match.GOOD, "high": Match.OK}[level]))
        else:
            scores.append(("high", {"low": Match.BAD, "medium": Match.OK, "high": Match.GOOD}[level]))


        schemes.append([(b, Match.GOOD if a == b else Match.BAD) for a, b in zip(rhyme, scheme_to_label(*scheme))])
        meters.append(meters_)

    return schemes, meters, scores


if __name__ == "__main__":
    parser = ArgumentParser(
        description="Generate quatrains using a pretrained poetry model."
    )
    parser.add_argument(
        "--rhyme",
        help="rhyme scheme to use",
        choices=QUATRAIN_RHYME_SCHEMES,
        required=True
    )
    parser.add_argument(
        "--prompt",
        help="initialize quatrain with this text as first verse (as meter is inferred from this, --meter is not used)",
        type=lambda p: p.strip().replace('\n', ' '),
    )
    parser.add_argument(
        "--meter",
        help="meter to use",
        choices=[meter for meter in METERS if meter != "other"],
        default="iambus"
    )
    parser.add_argument(
        "--alliteration",
        help="alliteration level",
        choices=ALLITERATION_LEVELS,
        default="medium"
    )
    parser.add_argument(
        "--model_name_or_path",
        default="nllg/poetry-bygpt5-base-en",
        help="name of the model in huggingface hub or path if local",
    )
    parser.add_argument(
        "--meter_model_name_or_path",
        default="nllg/clf-canine-m",
        help="name or path of the meter classification model",
    )
    parser.add_argument(
        "--rhyme_model_name_or_path",
        default="nllg/clf-canine-r",
        help="name or path of the rhyme classification model",
    )
    parser.add_argument(
        "--lang",
        choices=["en", "de"],
        default="en",
        help="specify which language to use for classification of generated quatrains",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=5,
        help="number of quatrains to generate",
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=0.9,
        help="float to define the tokens that are within the sample operation of text generation",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="the temperature of the sampling operation",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="device for computation",
    )

    args = parser.parse_args()

    if args.prompt:
        args.meter = identify(
            model_name=args.meter_model_name_or_path,
            prompt=args.prompt,
            lang=args.lang
        )
        print(f"Detected meter {args.meter}.")

    quatrains = generate(
        model_name=args.model_name_or_path,
        verse=args.prompt,
        rhyme=args.rhyme,
        meter=args.meter,
        alliteration=args.alliteration,
        dev=args.device,
        num_samples=args.num_samples,
        top_p=args.top_p,
        temperature=args.temperature)

    quatrains = [splitted for quatrain in quatrains if len((splitted := quatrain.split("\n"))) == 4]

    if len(quatrains) < args.num_samples:
        print(f"Filtered {args.num_samples - len(quatrains)} garbage quatrains.")

    rhymes, meters, levels = classify(
        quatrains=quatrains,
        rhyme=args.rhyme,
        meter=args.meter,
        level=args.alliteration,
        meter_model_name=args.meter_model_name_or_path,
        rhyme_model_name=args.rhyme_model_name_or_path,
        lang=args.lang
    )

    for quatrain, rhyme, meter, level in zip(quatrains, rhymes, meters, levels):
        quatrain_width = max([len(verse) for verse in quatrain])
        meter_width = max([len(label) for label, _ in meter])

        print("Alliteration level", f"{colored(*level)}:")
        for verse, scheme, (meter_label, meter_enum) in zip(quatrain, rhyme, meter):
            print(verse.ljust(quatrain_width), colored(*scheme), colored(meter_label.ljust(meter_width), meter_enum))
        print()


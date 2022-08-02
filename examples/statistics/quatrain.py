#!/usr/bin/env python
from argparse import ArgumentParser
from json import dump
from os import makedirs
from os.path import join

from numpy import asarray, split

from uniformers.datasets import load_dataset
from uniformers.utils import METERS, QUATRAIN_RHYME_SCHEMES, QuatrainProcessing

def quatrain_stats(lang, args):
    dataset = load_dataset("quatrain", lang=lang, split="train")
    dataset = dataset.map(
        QuatrainProcessing(
            lang=lang,
            meter_model_name=args.meter_model_name_or_path,
            rhyme_model_name=args.rhyme_model_name_or_path,
            batch_size=args.batch_size,
        ),
        batched=True,
        remove_columns=dataset.column_names,  # pyright: ignore
    )

    stats = {"length": len(dataset), "meter": {}, "rhyme": {}} # pyright: ignore
    for scheme in QUATRAIN_RHYME_SCHEMES:
        stats['rhyme'][scheme] = len(dataset.filter(lambda example: example["rhyme"] == scheme)) # pyright: ignore
    for meter in METERS:
        stats['meter'][meter] = len(dataset.filter(lambda example: example["meter"] == meter)) # pyright: ignore
    scores = asarray(dataset['aliteration']) # pyright: ignore
    alit_low, alit_medium, alit_high = split(sorted(scores), [round(len(scores)*0.5), round(len(scores)*0.8)])
    stats['aliteration'] = {
        "low": [alit_low[0], alit_low[-1]],
        "medium": [alit_medium[0], alit_medium[-1]],
        "high": [alit_high[0], alit_high[-1]]
    }

    print(f"QuaTrain statistics ({lang}):")
    print(f"  #quatrains: {stats['length']}")
    for scheme, count in stats['rhyme'].items():
        print(f"  #{scheme}: {count}")
    for meter, count in stats['meter'].items():
        print(f"  #{meter}: {count}")
    for alit_level, range_ in stats['aliteration'].items():
        print(f"  Aliteration ({alit_level}) range: {range_}")

    return stats


if __name__ == "__main__":
    argument_parser = ArgumentParser(
        description="Compute statistics of QuaTrain dataset."
    )
    argument_parser.add_argument(
        "--out_dir",
        default="data",
        help="directory where to write the json file with results",
    )
    argument_parser.add_argument(
        "--meter_model_name_or_path",
        default="nllg/clf-canine-m",
        help="name of the model in huggingface hub or path if local",
    )
    argument_parser.add_argument(
        "--rhyme_model_name_or_path",
        default="nllg/clf-canine-r",
        help="name of the model in huggingface hub or path if local",
    )
    argument_parser.add_argument(
        "--batch_size",
        default=512,
        type=int,
        help="batch size to use for dataset processing",
    )

    args = argument_parser.parse_args()
    makedirs(args.out_dir, exist_ok=True)
    with open(
        join(args.out_dir, f"quatrain-statistics.json"), "w"
    ) as fp:
        dump({"de": quatrain_stats("de", args), "en": quatrain_stats("en", args)}, fp)

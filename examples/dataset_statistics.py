#!/usr/bin/env python

from argparse import ArgumentParser
from collections import Counter
from json import dump
from os import makedirs
from os.path import join

from uniformers.datasets import load_dataset

def count_labels(type_, column="labels", lang="all"):
    dataset = load_dataset("poetrain", type_, split="train")
    labels = dataset.features[column]  # pyright: ignore
    counts = Counter()

    if lang != "all":
        dataset = dataset.filter(lambda example: example["language"] == lang)

    for label_class in range(labels.num_classes):
        num_labels = len([label for label in dataset[column] if label == label_class])  # pyright: ignore
        counts[labels.int2str(label_class)] = num_labels

    return counts


def poetrain_stats():
    data = {"all": {}, "en": {}, "de": {}}

    for lang in data.keys():
        data[lang]["meter"] = count_labels("meter", lang=lang)
        data[lang]["original"] = count_labels("meter", column="original", lang=lang)
        data[lang]["rhyme"] = count_labels("rhyme", lang=lang)

    for lang in data.keys():
        print("{} PoeTrain statistics:".format({"all": "All", "en": " English", "de": "German"}[lang]))
        for key, value in zip(["Meter", "Original meter", "Rhyme"], data[lang].values()):
            print(f"  {key}:")
            for label, count in value.most_common():
                print(f"    {label}:", count)

    return data

def quatrain_stats():
    stats = {
        "de": {"length": len(load_dataset("quatrain", lang="de", split="train"))}, # pyright: ignore
        "en": {"length": len(load_dataset("quatrain", lang="en", split="train"))}, # pyright: ignore
    }

    print("QuaTrain statistics:")
    print(f"  Number of English quatrains: {stats['en']['length']}")
    print(f"  Number of German quatrains: {stats['de']['length']}")
    return stats

if __name__ == "__main__":
    argument_parser = ArgumentParser(
        description="Compute statistics of Poetrain dataset."
    )
    argument_parser.add_argument(
        "--out_dir",
        default="data",
        help="directory where to write the json file with results",
    )
    args = argument_parser.parse_args()
    makedirs(args.out_dir, exist_ok=True)
    with open(
        join(args.out_dir, f"dataset-statistics.json"), "w"
    ) as fp:
        dump({"poetrain": poetrain_stats(), "quatrain": quatrain_stats()}, fp)

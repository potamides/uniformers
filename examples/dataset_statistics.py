#!/usr/bin/env python

from argparse import ArgumentParser
from collections import Counter
from json import dump
from os import makedirs
from os.path import join

from uniformers.datasets import load_dataset

dataset = load_dataset("poetrain", "meter", split="train")


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


def gen_stats():
    data = {"all": {}, "en": {}, "de": {}}

    for lang in data.keys():
        data[lang]["meter"] = count_labels("meter", lang=lang)
        data[lang]["original"] = count_labels("meter", column="original", lang=lang)
        data[lang]["rhyme"] = count_labels("rhyme", lang=lang)

    for lang in data.keys():
        print("{} Poetrain statics:".format({"all": "All", "en": " English", "de": "German"}[lang]))
        for key, value in zip(["Meter", "Original meter", "Rhyme"], data[lang].values()):
            print(f"  {key}:")
            for label, count in value.most_common():
                print(f"    {label}:", count)

    return data


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
    stats = gen_stats()
    with open(
        join(args.out_dir, f"statistics-poetrain.json"), "w"
    ) as fp:
        dump(stats, fp)

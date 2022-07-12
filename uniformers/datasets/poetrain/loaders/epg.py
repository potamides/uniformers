from glob import glob
from itertools import chain, islice
from os.path import join
from pathlib import Path
from random import Random

from lxml import etree
from transformers.utils import logging

from uniformers.utils import align_syllables, find_rhymes, meter_to_label

logger = logging.get_logger("transformers")
random = Random(0)
NS = "{http://www.tei-c.org/ns/1.0}"

def _parse_xml(xml_file):
    poems = dict()
    tree = etree.parse(str(xml_file))
    root = tree.getroot()
    tei_url = NS.replace("{", "").replace("}", "")

    if not any(ns == tei_url for ns in root.nsmap.values()):
        return

    title = root.find(f".//{NS}title").text
    author = root.find(f".//{NS}author")
    author_text = "unknown"

    if author is not None:
        author_text = author.text

    poems.update({
        "poem_title": title,
        "author": author_text,
    })
    stanza_list = list()
    for poem in root.findall(f".//{NS}body/{NS}lg"):
        assert poem.get("type") in ["poem", "part"]
        for stanza in poem.findall(f"{NS}lg"):
            assert stanza.get("type") == "stanza"
            rhyme, line_list = stanza.get("rhyme"), list()
            for line in stanza.findall(f"{NS}l"):
                foot, metre = align_syllables(line.get("met"))
                line_list.append({
                    "line_text": line.text,
                    "metrical_pattern": line.get("met"),
                    "foot": foot,
                    "metre": metre,
                })
            stanza_list.append({
                "stanza_type": stanza.get("type"),
                "rhyme": rhyme,
                "lines": line_list,
            })

    poems.update({
        "stanzas": stanza_list,
    })
    return poems

def _get_features(path):
    ecpa_folder = glob(join(path, "epg64-english-poetry-annotated-*"))[0]
    for filename in (Path(ecpa_folder) / "poems_xml").rglob("*.xml"):
        if filename.stem != "EdwardEstlinCummings_1894": # seems to contain errors
            result = _parse_xml(filename)
            if result is not None:
                yield result

def epg_loader(path, config):
    if config.name == "meter":
        for i, poem in enumerate(_get_features(path)):
            for j, stanza in enumerate(poem['stanzas']):
                for k, line in enumerate(stanza['lines']):
                    if line["foot"]:
                        yield f"epg64-{i}-{j}-{k}", {
                            "text": line["line_text"].strip(),
                            "language": "en",
                            "labels": meter_to_label(line["foot"]),
                            "original": meter_to_label(line["foot"], group_rare=False),
                        }
                    else:
                        logger.debug(f"A verse from {poem['author']} doesn't contain foot, skipping.")
    elif config.name == "rhyme":
        rhyme_pairs, dissonance_pairs = dict(), dict()
        for i, poem in enumerate(_get_features(path)):
            for j, stanza in enumerate(poem['stanzas']):
                if stanza['rhyme'] and len(lines := stanza['lines']) > 1:
                    if len(lines[0]['line_text']) <= 3:
                        lines = lines[1:]
                        #stanza['rhyme'] = stanza['rhyme'][1:]
                        logger.debug(f"A verse from {poem['author']} is too short, removing.")
                    if len(lines) < len(stanza['rhyme']):
                        stanza['rhyme'] = stanza['rhyme'][:len(lines)]
                        logger.debug(f"Rhyme scheme for stanza of {poem['author']} too long, shortening.")
                    elif len(lines) > len(stanza['rhyme']):
                        logger.debug(f"Rhyme scheme and stanza length don't match for {poem['author']}, skipping.")
                        continue

                    r_pairs, d_pairs = find_rhymes([l['line_text'].strip() for l in lines], stanza['rhyme'])
                    for pair, (v1, v2, label) in r_pairs.items():
                        rhyme_pairs[tuple(sorted(pair))] = (f"prosodic-{i}-{j}-{v1}-{v2}", label)
                    for pair, (v1, v2, label) in d_pairs.items():
                        dissonance_pairs[tuple(sorted(pair))] = (f"prosodic-{i}-{j}-{v1}-{v2}", label)

        dissonance_pairs = dict(random.sample(list(dissonance_pairs.items()), len(dissonance_pairs)))
        for pair, (_id, label) in chain(rhyme_pairs.items(), islice(dissonance_pairs.items(), len(rhyme_pairs))):
            yield _id, {
                "text": pair,
                "language": "en",
                "labels": label,
            }

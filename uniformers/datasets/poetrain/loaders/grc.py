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
        forename = author.find(f".//{NS}forename")
        surname = author.find(f".//{NS}surname")

        if forename is not None:
            author_text = f"{forename.text} {surname.text}"
        else:
            author_text = surname.text

    poems.update({
        "poem_title": title,
        "author": author_text,
    })
    stanza_list = list()
    for poem in root.findall(f".//{NS}body/{NS}div/{NS}lg"):
        assert poem.get("type") in ["poem", "part"]
        for stanza in poem.findall(f"{NS}lg"):
            if (stan_typ := stanza.get("type")) == "stanza":
                rhyme, line_list = stanza.get("rhyme"), list()
                for line in stanza.findall(f"{NS}l"):
                    etree.strip_tags(line, "*")
                    if line.text:
                        line_list.append(line.text)
                    else:
                        logger.debug("Stanza contains empty verses, skipping.")
                        break
                else:
                    stanza_list.append({
                        "stanza_type": stanza.get("type"),
                        "rhyme": rhyme,
                        "lines": line_list,
                    })
            else:
                logger.debug(f"Not a stanza (has type {stan_typ}), skipping.")

    poems.update({
        "stanzas": stanza_list,
    })
    return poems

def _get_features(path):
    ecpa_folder = glob(join(path, "german-rhyme-corpus-*"))[0]
    for filename in (Path(ecpa_folder) / "Diachron_Sample_DTA_DTR_Rhyme_Annotated").glob("*.xml"):
        result = _parse_xml(filename)
        if result is not None:
            yield result

def grc_loader(path, _):
    rhyme_pairs, dissonance_pairs = dict(), dict()
    for i, poem in enumerate(_get_features(path)):
        for j, stanza in enumerate(poem['stanzas']):
            if stanza['rhyme'] and len(lines := stanza['lines']) > 1:
                if any("\n" in line.strip() or len(line) <=3 for line in lines):
                    logger.debug(f"A stanza from {poem['author']} has strange formatting, skipping.")
                    continue
                if len(lines) < len(stanza['rhyme']):
                    logger.debug(f"Rhyme scheme for stanza of {poem['author']} too long, shortening.")
                    stanza['rhyme'] = stanza['rhyme'][:len(lines)]
                elif len(lines) > len(stanza['rhyme']):
                    logger.debug(f"Rhyme scheme and stanza length don't match for {poem['author']}, skipping.")
                    continue

                r_pairs, d_pairs = find_rhymes([l.strip() for l in lines], stanza['rhyme'])
                for pair, (v1, v2, label) in r_pairs.items():
                    rhyme_pairs[tuple(sorted(pair))] = (f"grc-{i}-{j}-{v1}-{v2}", label)
                for pair, (v1, v2, label) in d_pairs.items():
                    dissonance_pairs[tuple(sorted(pair))] = (f"grc-{i}-{j}-{v1}-{v2}", label)

    dissonance_pairs = dict(random.sample(list(dissonance_pairs.items()), len(dissonance_pairs)))
    for pair, (_id, label) in chain(rhyme_pairs.items(), islice(dissonance_pairs.items(), len(rhyme_pairs))):
        yield _id, {
            "text": pair,
            "language": "de",
            "labels": label,
        }

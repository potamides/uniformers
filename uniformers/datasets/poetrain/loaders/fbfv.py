from glob import glob
from itertools import chain, islice
from os.path import basename, join
from pathlib import Path
from random import Random
import re

from lxml import etree

from uniformers.utils import meter_to_label, find_rhymes, align_syllables

NS = "{http://www.tei-c.org/ns/1.0}"
random = Random(0)

# based from https://github.com/linhd-postdata/averell/blob/develop/src/averell/readers/forbetter4verse.py
def _parse_xml(xml_file):
    """Poem parser for 'For better for verse' corpus.
    We read the data and find elements like title, author, year, etc. Then
    we iterate over the poem text and we look for each stanza and  line data.
    :param xml_file: Path for the poem xml file
    :return: Dict with the data obtained from the poem
    :rtype: dict
    """
    poem = {}
    tree = etree.parse(str(xml_file))
    corpus_name = xml_file.parts[-4]
    root = tree.getroot()
    nsmap = root.nsmap
    tei_url = NS.replace("{", "").replace("}", "")
    if not any(ns == tei_url for ns in nsmap.values()):
        # no TEI declaration in input file, cannot parse file
        return
    title = root.find(f".//{NS}title").text
    author = root.find(f".//{NS}author")
    author_text = "unknown"
    if author is not None:
        author_text = author.text
    date = root.find(f".//{NS}date")
    year = None
    if date is not None:
        year = root.find(f".//{NS}date").text
    poem.update({
        "poem_title": title,
        "author": author_text,
        "year": year
    })
    line_group_list = root.findall(f".//{NS}lg")
    stanza_list = []
    line_number = 0
    for stanza_number, line_group in enumerate(line_group_list):
        stanza_type = line_group.get("type")
        rhyme = line_group.get("rhyme")
        line_list = []
        stanza_text = []
        for line in line_group.findall(f"{NS}l"):
            line_dict = {}
            line_length = None
            met = line.get("met")
            foot, metre = align_syllables(met)
            seg_list = [re.sub(r"[\n ]+", " ", seg.xpath("string()")) for seg in
                        line.findall(f"{NS}seg")]
            line_text = "".join(seg_list)
            line_dict.update({
                "line_number": line_number + 1,
                "line_text": line_text,
                "metrical_pattern": met,
                "line_length": line_length,
                "foot": foot,
                "metre": metre,
            })
            line_list.append(line_dict)
            stanza_text.append(line_text)
            line_number += 1
        stanza_list.append({
            "stanza_number": stanza_number + 1,
            "stanza_type": stanza_type,
            "rhyme": rhyme,
            "lines": line_list,
            "stanza_text": "\n".join(stanza_text),
        })
    poem.update({
        "stanzas": stanza_list,
        "corpus": corpus_name,
    })
    return poem


def _get_features(path):
    """Function to find each poem file and parse it
    :param path: Corpus Path
    :return: List of poem dicts
    :rtype: list
    """

    ecpa_folder = basename(glob(join(path, "for_better_for_verse-*"))[0])
    path = Path(path)
    xml_folders = [
        path / ecpa_folder / "poems",
        #path / ecpa_folder / "poems2" # seem to be only repetitions
    ]
    for folder in xml_folders:
        for filename in folder.rglob("*.xml"):
            result = _parse_xml(filename)
            if result is not None:
                yield result

def fbfv_loader(path, config):
    if config.name == "meter":
        for i, poem in enumerate(_get_features(path)):
            for stanza in poem['stanzas']:
                for line in stanza['lines']:
                    yield f"prosodic-{i}- {stanza['stanza_number']}-{line['line_number']}", {
                        "text": line["line_text"].strip(),
                        "language": "en",
                        "labels": meter_to_label(line["foot"]),
                        "original": meter_to_label(line["foot"], group_rare=False),
                    }
    elif config.name == "rhyme":
        rhyme_pairs, dissonance_pairs = dict(), dict()
        for i, poem in enumerate(_get_features(path)):
            for stanza in poem['stanzas']:
                # there is one poem with one line.. doesn't make much sense
                if stanza['rhyme'] and len(lines := stanza['lines']) > 1:
                    stanza_nr = stanza['stanza_number']
                    r_pairs, d_pairs = find_rhymes([l['line_text'].strip() for l in lines], stanza['rhyme'])
                    for pair, (v1, v2, label) in r_pairs.items():
                        rhyme_pairs[tuple(sorted(pair))] = (f"prosodic-{i}-{stanza_nr}-{v1}-{v2}", label)
                    for pair, (v1, v2, label) in d_pairs.items():
                        dissonance_pairs[tuple(sorted(pair))] = (f"prosodic-{i}-{stanza_nr}-{v1}-{v2}", label)

        # deterministically shuffle dissonance_pairs to minimize repeated verses
        dissonance_pairs = dict(random.sample(list(dissonance_pairs.items()), len(dissonance_pairs)))
        # also add equal amount dissonance_pairs
        for pair, (_id, label) in chain(rhyme_pairs.items(), islice(dissonance_pairs.items(), len(rhyme_pairs))):
            yield _id, {
                "text": pair,
                "language": "en",
                "labels": label,
            }

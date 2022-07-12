from csv import DictReader
from itertools import chain, islice
from random import Random

from transformers.utils import logging

from uniformers.utils import find_rhymes, meter_to_label

random = Random(0)
logger = logging.get_logger("transformers")

def wild_loader(filepath, config):
    with open(filepath, newline="") as csvfile:
        reader = DictReader(csvfile, delimiter="\t")
        if config.name == "meter":
            for idx, row in enumerate(reader):
                if row["s_measure"] not in ["unknown", "single"]:
                    yield f"prosodic-{idx}", {
                        "text": row["line_text"].strip(),
                        "language": "de",
                        "labels": meter_to_label(row["s_measure"]),
                        "original": meter_to_label(row["s_measure"], group_rare=False),
                    }
                else:
                    logger.debug(f"Skipping line with meter '{row['s_measure']}'.")
        elif config.name == "rhyme":
            lines = [line for line in reader]
            rhyme_pairs, dissonance_pairs = dict(), dict()
            num_stanzas = int(lines[-1]['stanza_id'])
            for stanza_id in range(1, num_stanzas + 1):
                stanza = [line for line in lines if int(line['stanza_id']) == stanza_id]
                assert all(int(verse['total_lines_in_stanza']) == len(stanza) for verse in stanza)
                assert all(verse['rhyme_schema'] == stanza[0]['rhyme_schema'] for verse in stanza)

                if len(stanza) == 0:
                    logger.debug(f"Stanza is empty, skipping.")
                    continue
                elif len(stanza) != len(scheme := stanza[0]['rhyme_schema']):
                    logger.debug(f"Rhyme scheme and verses don't add up, skipping.")
                    continue
                elif any(verse['s_measure'] == "single" for verse in stanza):
                    logger.debug(f"Skipping stanza with meter 'single'.")
                    continue

                r_pairs, d_pairs = find_rhymes([verse['line_text'].strip() for verse in stanza], "".join(scheme))
                for pair, (v1, v2, label) in r_pairs.items():
                    rhyme_pairs[tuple(sorted(pair))] = (f"chicago-{stanza_id}-{v1}-{v2}", label)
                for pair, (v1, v2, label) in d_pairs.items():
                    dissonance_pairs[tuple(sorted(pair))] = (f"chicago-{stanza_id}-{v1}-{v2}", label)

            dissonance_pairs = dict(random.sample(list(dissonance_pairs.items()), len(dissonance_pairs)))
            for pair, (_id, label) in chain(rhyme_pairs.items(), islice(dissonance_pairs.items(), len(rhyme_pairs))):
                yield _id, {
                    "text": pair,
                    "language": "de",
                    "labels": label,
                }

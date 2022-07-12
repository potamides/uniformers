from collections import Counter, defaultdict
from csv import DictReader
from glob import glob
from os.path import join

from uniformers.utils import meter_to_label


def prosodic_loader(filepath, _):
    globpattern = join(
        filepath, "prosodic-*/tagged_samples/tagged-sample-litlab-2016.txt"
    )
    with open(glob(globpattern)[0], newline="") as csvfile:
        reader = DictReader(csvfile, delimiter="\t")
        stanzas = defaultdict(list)
        for row in reader:
            meter, _id, stanza_nr, verse = (
                row["Meter Scheme"],
                row["PoemID"],
                eval(row["line_num"])[1],
                row["line"].strip(),
            )
            stanzas[(_id, stanza_nr)].append((verse, meter))

        for (_id, stanza_nr), stanza in stanzas.items():
            if len(stanza) < 4:
                continue
            for i in range(len(stanza) - 3):
                stanza_win = stanza[i : i + 4]

                # Meter for the stanza is the most common meter of the verses.
                # Usually they are all the same.
                stanza_meter = Counter(
                    [meter for (_, meter) in stanza_win]
                ).most_common(1)[0][0]
                yield f"prosodic-{_id}-{stanza_nr}-{i}", {
                    "text": "\n".join([verse for (verse, _) in stanza_win]),
                    "language": "en",
                    "labels": meter_to_label(stanza_meter),
                }

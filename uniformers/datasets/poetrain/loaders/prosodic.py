from csv import DictReader

from uniformers.utils import meter_to_label


def prosodic_loader(filepath, _):
    with open(filepath, newline="") as csvfile:
        reader = DictReader(csvfile, delimiter="\t")
        for row in reader:
            _id, verse_nr = row["PoemID"], eval(row["line_num"])[0]
            yield f"prosodic-{_id}-{verse_nr}", {
                "text": row["line"].strip(),
                "language": "en",
                "labels": meter_to_label(row["Meter Scheme"]),
                "original": meter_to_label(row["Meter Scheme"], group_rare=False),
            }

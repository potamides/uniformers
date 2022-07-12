from itertools import chain
from re import escape, finditer

QUATRAIN_RHYME_SCHEMES = (
    "AAAA",  # Monorhyme
    "AAAB",
    "AABA",
    "AABB",  # Couplet
    "AABC",
    "ABAA",
    "ABAB",  # Alternating Rhyme
    "ABAC",
    "ABBA",  # Enclosed rhyme
    "ABBB",
    "ABBC",
    "ABCA",
    "ABCB",  # Simple 4-line
    "ABCC",
    "ABCD",
)

RHYME_LABELS = ("rhyme", "dissonance")

ALLITERATION_LEVELS = ("high", "medium", "low")

METERS = (
    "alexandrine",
    "amphibrach",
    "anapaest",
    "dactyl",
    "iambus",
    "other",
    "trochee",
)

RARE_METERS = (
    "asklepiade",
    "diphilius",
    "glykoneus",
    "hexameter",
    "pherekrateus",
    "prosodiakos",
    "spondee",
    "zehnsilber",
)

ALL_METERS = METERS + RARE_METERS


def scheme_to_label(a, b, c, d):
    """Converts schemes in the form CCAD to AABC"""
    label = f"{a}\0{b}\0{c}\0{d}"
    for char, replacement in zip(dict.fromkeys([a, b, c, d]), ["A", "B", "C", "D"]):
        label = label.replace(char, replacement)
    return label.replace("\0", "")


def meter_to_label(meter, group_rare=True):
    match meter:
        case "anapestic": meter = "anapaest"
        case "dactylic": meter = "dactyl"
        case "trochaic": meter = "trochee"
        case "spondeus": meter = "spondee"
        case "iamb" | "iambic": meter = "iambus"

    # group rare meters to label "other"
    if group_rare and meter in [
        "asklepiade",
        "pherekrateus",
        "glykoneus",
        "prosodiakos",
        "hexameter",
        "zehnsilber",
        "spondee",
        "diphilius",
    ]:
        meter = "other"

    if meter in METERS if group_rare else ALL_METERS:
        return meter
    else:
        raise ValueError(f"Meter called '{meter}' can't be processed or doesn't exist.")


def find_rhymes(stanza, rhyme):
    assert len(stanza) == len(rhyme)
    rhyme_pairs, dissonance_pairs = dict(), dict()
    for j, letter in enumerate(rhyme):
        for k in chain(
            finditer(escape(letter), rhyme), finditer(f"[^{escape(letter)}]", rhyme)
        ):
            if j != k.start():
                verse1, verse2 = stanza[j], stanza[k.start()]
                pair = (verse1, verse2)
                if k.re.pattern == letter:
                    rhyme_pairs[tuple(sorted(pair))] = (j, k.start(), RHYME_LABELS[0])
                else:
                    dissonance_pairs[tuple(sorted(pair))] = (
                        j,
                        k.start(),
                        RHYME_LABELS[1],
                    )

    return rhyme_pairs, dissonance_pairs

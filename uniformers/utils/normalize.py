from re import sub

# based on https://github.com/tnhaider/xml-poetry-reader/blob/master/utils/helper.py#L100
def normalize_characters(text):
    corrections = {
        "<[^>]*>": "",
        "ſ": "s",
        r"\[": "",
        r"\]": "",
        "’": "'",
        "&#223;": "ß",
        "&#383;": "s",
        "u&#868;": "ü",
        "a&#868;": "ä",
        "o&#868;": "ö",
        "&#246;": "ö",
        "&#224;": "a",
        "&#772;": "m",
        "&#8217;": "'",
        "&#42843;": "r",
        "&#244;": "o",
        "&#230;": "ae",
        "&#8229;": ".",
        "Jch": "Ich",
        "Jhr": "Ihr",
        "Jst": "Ist",
        "JCh": "Ich",
        "jch": "ich",
        "Jn": "In",
        "DJe": "Die",
        "Wje": "Wie",
        "¬": "-",
        " ’ ": "'",
        "’": "'",
        "´": "'",
        "''": '"',
        "—": "--",
        " - ": "-",
        "”": '"',
    }
    utf_corrections = {
        b"o\xcd\xa4": b"\xc3\xb6",
        b"u\xcd\xa4": b"\xc3\xbc",
        b"a\xcd\xa4": b"\xc3\xa4",
        b"&#771;": b"\xcc\x83",
        b"&#8222;": b"\xe2\x80\x9d",
        b"\xea\x9d\x9b": b"r",
        b"\xea\x9d\x9a": b"R",
    }
    for src, tgt in corrections.items():
        text = sub(src, tgt, text)
    text = text.encode("utf-8", "replace")
    for src, tgt in utf_corrections.items():
        text = sub(src, tgt, text)
    text = text.decode("utf-8")
    if text.startswith("b'"):
        text = text[2:-1]
    # check if first two letters are capitalized
    if len(text) >= 3 and text[:2].isupper() and not text[2].isupper():
        text = text[0] + text[1].lower() + text[2:]
    return text

"""Text normalisation for business names and addresses.

The noise in this dataset comes from a generator that (a) abbreviates / expands
legal forms and street types, (b) injects titles, alias phrases, filler words and
junk symbols, (c) swaps look-alike characters (i/l/1, o/0, s/5), (d) adds accents,
(e) transliterates Indian records into native scripts, and (f) drops, reorders or
pads address components.  Each function below undoes one family of that noise so
that downstream similarity features compare like with like.

Only hand-written, general-purpose abbreviation tables are used (no external data).
"""

import re
import unicodedata

from indic_transliteration import sanscript
from indic_transliteration.sanscript import transliterate
from unidecode import unidecode

# --------------------------------------------------------------------------------------
# Indic script -> Latin
# --------------------------------------------------------------------------------------
# (start, end, scheme); scheme None means unidecode gives a closer English spelling.
_INDIC_BLOCKS = [
    (0x0900, 0x097F, sanscript.DEVANAGARI),
    (0x0980, 0x09FF, sanscript.BENGALI),
    (0x0A00, 0x0A7F, sanscript.GURMUKHI),
    (0x0A80, 0x0AFF, sanscript.GUJARATI),
    (0x0B00, 0x0B7F, sanscript.ORIYA),
    (0x0B80, 0x0BFF, None),  # Tamil
    (0x0C00, 0x0C7F, sanscript.TELUGU),
    (0x0C80, 0x0CFF, sanscript.KANNADA),
    (0x0D00, 0x0D7F, sanscript.MALAYALAM),
]
_INDIC_RUN = re.compile(r"[ऀ-ൿ][ऀ-ൿ‌‍]*")
_HAS_INDIC = re.compile(r"[ऀ-ൿ]")


def _script_of(ch):
    cp = ord(ch)
    for lo, hi, scheme in _INDIC_BLOCKS:
        if lo <= cp <= hi:
            return scheme
    return None


def _translit_word(word):
    """Transliterate one run of Indic characters to plain ASCII."""
    word = word.replace("‌", "").replace("‍", "")
    scheme = _script_of(word[0])
    if scheme is None:
        out = unidecode(word)
    else:
        out = unidecode(transliterate(word, scheme, sanscript.IAST))
    out = out.lower()
    # Hindi-style spellings of English loanwords: drop the inherent final 'a'
    # (limiteda -> limited), anusvara before g -> n (marketimga -> marketinga).
    out = re.sub(r"m(?=[gk])", "n", out)
    if len(out) > 3 and out.endswith("a") and out[-2] not in "aeiou":
        out = out[:-1]
    out = out.replace("ph", "f")
    return out


_LEARNED = None  # native word -> English word / state code, learned from training pairs


def _learned():
    """Load the dictionary produced by indic_dict.py (empty if it has not been built)."""
    global _LEARNED
    if _LEARNED is None:
        import json
        import os
        from pathlib import Path
        path = Path(os.environ.get("BER_INDIC_DICT", Path(__file__).resolve().parents[3] /
                                   "work" / "cache" / "indic_dict.json"))
        try:
            from config import CACHE_DIR
            path = Path(os.environ.get("BER_INDIC_DICT", CACHE_DIR / "indic_dict.json"))
        except ImportError:
            pass
        _LEARNED = json.loads(path.read_text(encoding="utf-8")) if path.exists() else             {"name": {}, "addr": {}}
    return _LEARNED


def translit_indic(text, kind="name"):
    """Replace every Indic-script run in ``text`` with English.

    Words seen in training use the learned dictionary (exact); other words fall back to
    rule-based transliteration.  ``kind`` selects the name or address dictionary.
    """
    if not _HAS_INDIC.search(text):
        return text
    table = _learned().get(kind, {})

    def repl(m):
        w = m.group(0)
        return " " + (table.get(w) or _translit_word(w)) + " "

    return _INDIC_RUN.sub(repl, text)


def has_indic(text):
    return bool(_HAS_INDIC.search(text or ""))


# --------------------------------------------------------------------------------------
# Shared helpers
# --------------------------------------------------------------------------------------
_NULLS = re.compile(r"<\s*null\s*>|\bnull\b|\bn/a\b|\bnone\b|\bnil\b", re.I)


def to_ascii(text, kind="name"):
    """Transliterate Indic scripts, strip accents, lowercase."""
    if not text:
        return ""
    text = translit_indic(text, kind)
    text = text.replace("°", " ")  # N° -> N
    text = unicodedata.normalize("NFKC", text)
    return unidecode(text).lower()


def _fold_confusables(tok):
    """Map look-alike characters to one representative (applied to names only).

    The generator swaps i<->l, l<->1, o<->0, s<->5.  Folding every token the same
    way makes 'We1fare', 'Welfare', 'lndia' and 'India' compare equal.
    """
    if not tok.isdigit():
        tok = tok.replace("0", "o").replace("1", "l").replace("5", "s")
    return tok.replace("i", "l")


# --------------------------------------------------------------------------------------
# Business names
# --------------------------------------------------------------------------------------
# Canonical legal forms (written unfolded; folded at import time).
_LEGAL_MAP_RAW = {
    "pvt": "pvt", "private": "pvt", "pte": "pvt", "prv": "pvt", "pra": "pvt", "praivet": "pvt",
    "ltd": "ltd", "limited": "ltd", "lt": "ltd", "li": "ltd", "limitet": "ltd", "limitd": "ltd",
    # native-script spellings after transliteration (Hindi, Tamil, Telugu, ...)
    "praivett": "pvt", "piraiveett": "pvt", "piraivet": "pvt", "praiveta": "pvt",
    "limittett": "ltd", "limitett": "ltd", "limitted": "ltd", "limitedd": "ltd",
    "llc": "llc", "llp": "llp", "lp": "lp", "pllc": "pllc", "plc": "plc", "pc": "pc",
    "inc": "inc", "incorporated": "inc", "corp": "corp", "corporation": "corp",
    "co": "co", "company": "co", "cie": "co", "compagnie": "co",
    "sarl": "sarl", "sas": "sas", "sasu": "sasu", "sa": "sa", "eurl": "eurl", "sci": "sci",
    "snc": "snc", "ei": "ei", "scop": "scop", "selarl": "selarl", "gie": "gie", "gmbh": "gmbh",
    "ag": "ag", "bv": "bv", "nv": "nv", "srl": "srl", "spa": "spa", "pty": "pty",
}
_TITLES_RAW = {"mr", "mrs", "ms", "messrs", "dr", "smt", "shri", "sri", "shree", "the", "m"}
_ALIAS_RAW = {"aka", "fka", "dba", "nee", "formerly", "doing", "business", "known", "as", "ta"}
_FILLER_RAW = {"center", "centre", "services", "service", "partners", "and", "trading",
               "enterprises", "labs", "sys", "one", "india", "france", "usa", "us", "www", "com"}

_LEGAL_MAP = {_fold_confusables(k): v for k, v in _LEGAL_MAP_RAW.items()}
_TITLES = {_fold_confusables(t) for t in _TITLES_RAW}
_ALIAS = {_fold_confusables(t) for t in _ALIAS_RAW}
_FILLER = {_fold_confusables(t) for t in _FILLER_RAW}
_LEGAL_VALUES = set(_LEGAL_MAP_RAW.values())
# Single letters that join into a legal form: "L.L.C." -> "l l c" -> "llc".
_JOINABLE = {_fold_confusables(k) for k in ("llc", "llp", "lp", "pllc", "pc", "plc", "sa",
                                            "sas", "sarl", "inc", "ltd", "pvt", "co", "sci")}
_NAME_SPLIT = re.compile(r"[^a-z0-9]+")


def _join_initials(tokens):
    """Merge runs of single-letter tokens that spell a legal form (l l c -> llc)."""
    out, i = [], 0
    while i < len(tokens):
        if len(tokens[i]) == 1:
            j = i
            while j < len(tokens) and len(tokens[j]) == 1:
                j += 1
            run = "".join(tokens[i:j])
            if j - i >= 2 and run in _JOINABLE:
                out.append(run)
            else:
                out.extend(tokens[i:j])
            i = j
        else:
            out.append(tokens[i])
            i += 1
    return out


def normalize_name(raw):
    """Return (tokens, core_tokens, flags) for a business name.

    tokens       all cleaned words, legal forms canonicalised (pvt, ltd, llc, ...)
    core_tokens  tokens without legal forms, titles, alias markers or filler words
    flags        dict: is_domain (website-style name), has_alias (aka/dba/formerly)
    """
    text = to_ascii(raw)
    text = text.replace("&", " and ").replace("'", "").replace("m/s", " ")
    is_domain = bool(re.search(r"\.com\b|\bwww\.", text))
    toks = [t for t in _NAME_SPLIT.split(text) if t]
    toks = [_fold_confusables(t) for t in toks]
    toks = _join_initials(toks)
    # website-style single token without the dot: "lucknowlndlacom"
    content = [t for t in toks if t not in _TITLES]
    if len(content) == 1 and len(content[0]) >= 9 and content[0].endswith("com"):
        is_domain = True
        toks = [t[:-3] if t is content[0] else t for t in toks]
    has_alias = any(t in ("aka", "fka", "dba", "nee", "formerly") for t in toks) or \
        "doing business as" in " ".join(toks)
    toks = [_LEGAL_MAP.get(t, t) for t in toks]
    core = [t for t in toks if t not in _LEGAL_VALUES and t not in _TITLES
            and t not in _ALIAS and t not in _FILLER]
    return toks, core, {"is_domain": is_domain, "has_alias": has_alias}


# --------------------------------------------------------------------------------------
# Addresses
# --------------------------------------------------------------------------------------
_STREET_MAP = {
    "street": "st", "str": "st", "st": "st", "saint": "st",
    "road": "rd", "rd": "rd", "drive": "dr", "dr": "dr", "avenue": "ave", "ave": "ave",
    "av": "ave", "lane": "ln", "ln": "ln", "court": "ct", "ct": "ct", "place": "pl",
    "pl": "pl", "circle": "cir", "cir": "cir", "boulevard": "blvd", "blvd": "blvd",
    "bd": "blvd", "boul": "blvd", "highway": "hwy", "hwy": "hwy", "parkway": "pkwy",
    "pkwy": "pkwy", "terrace": "ter", "terr": "ter", "ter": "ter", "trail": "trl",
    "trl": "trl", "square": "sq", "sq": "sq", "way": "way", "loop": "loop",
    "north": "n", "south": "s", "east": "e", "west": "w", "northeast": "ne",
    "northwest": "nw", "southeast": "se", "southwest": "sw",
    "apartment": "apt", "apt": "apt", "appartement": "apt", "suite": "ste", "ste": "ste",
    "floor": "fl", "fl": "fl", "flr": "fl", "building": "bldg", "bldg": "bldg",
    "near": "nr", "nr": "nr", "opposite": "opp", "opp": "opp", "sector": "sec", "sec": "sec",
    "nagar": "ngr", "ngr": "ngr", "colony": "col", "col": "col",
    "chemin": "chemin", "ch": "chemin", "chem": "chemin", "impasse": "imp", "imp": "imp",
    "allee": "allee", "all": "allee", "route": "rte", "rte": "rte", "faubourg": "fbg",
    "fbg": "fbg", "rue": "rue",
    "bombay": "mumbai", "bengaluru": "bangalore", "gurugram": "gurgaon",
    "calcutta": "kolkata", "madras": "chennai",
}
# Tokens that carry no information (injected by the generator or pure boilerplate).
_ADDR_DROP = {"no", "number", "num", "h", "hn", "hno", "door", "null", "na", "none",
              "house", "unit", "po", "box", "pmb", "cdp", "city", "of", "township", "county",
              "the", "de", "du", "des", "la", "le", "les", "d", "l", "region"}
_ORDINAL_RE = re.compile(r"\b(\d+)\s?(?:st|nd|rd|th)\b")  # 2nd -> 2, 1st floor -> 1 floor

_US_STATES = {
    "alabama": "al", "alaska": "ak", "arizona": "az", "arkansas": "ar", "california": "ca",
    "colorado": "co", "connecticut": "ct", "delaware": "de", "florida": "fl", "georgia": "ga",
    "hawaii": "hi", "idaho": "id", "illinois": "il", "indiana": "in", "iowa": "ia",
    "kansas": "ks", "kentucky": "ky", "louisiana": "la", "maine": "me", "maryland": "md",
    "massachusetts": "ma", "michigan": "mi", "minnesota": "mn", "mississippi": "ms",
    "missouri": "mo", "montana": "mt", "nebraska": "ne", "nevada": "nv",
    "new hampshire": "nh", "new jersey": "nj", "new mexico": "nm", "new york": "ny",
    "north carolina": "nc", "north dakota": "nd", "ohio": "oh", "oklahoma": "ok",
    "oregon": "or", "pennsylvania": "pa", "rhode island": "ri", "south carolina": "sc",
    "south dakota": "sd", "tennessee": "tn", "texas": "tx", "utah": "ut", "vermont": "vt",
    "virginia": "va", "washington": "wa", "west virginia": "wv", "wisconsin": "wi",
    "wyoming": "wy", "district of columbia": "dc", "puerto rico": "pr",
}
_IN_STATES = {
    "maharashtra": "mh", "delhi": "dl", "uttar pradesh": "up", "karnataka": "ka",
    "tamil nadu": "tn", "west bengal": "wb", "gujarat": "gj", "telangana": "tg",
    "haryana": "hr", "rajasthan": "rj", "kerala": "kl", "keralam": "kl", "bihar": "br",
    "madhya pradesh": "mp", "andhra pradesh": "ap", "punjab": "pb", "odisha": "od",
    "orissa": "od", "assam": "as", "jharkhand": "jh", "chhattisgarh": "cg",
    "uttarakhand": "uk", "himachal pradesh": "hp", "goa": "ga", "jammu and kashmir": "jk",
    "chandigarh": "ch", "puducherry": "py", "pondicherry": "py", "tripura": "tr",
    "meghalaya": "ml", "manipur": "mn", "nagaland": "nl", "mizoram": "mz", "sikkim": "sk",
    "arunachal pradesh": "ar",
}
_STATE_PHRASES = {}
for _table, _ctry in ((_US_STATES, "us"), (_IN_STATES, "india")):
    for _full, _code in _table.items():
        _STATE_PHRASES[(_ctry, tuple(_full.split()))] = _code
_MAX_STATE_WORDS = 3

_ADDR_SPLIT = re.compile(r"[^a-z0-9]+")
_LETTER_DIGIT = re.compile(r"(?<=[a-z])(?=\d)|(?<=\d)(?=[a-z])")


def _country_key(country):
    c = (country or "").strip().lower()
    return "india" if c == "india" else ("us" if c in ("us", "usa", "united states") else c)


def normalize_address(raw, country=""):
    """Return (tokens, numbers) for an address.

    tokens   cleaned words; street types/directions abbreviated to one canonical form,
             state names mapped to their postal code, junk/injected tokens removed
    numbers  numeric tokens with leading zeros stripped (house / plot / unit numbers)
    """
    text = to_ascii(raw, "addr")
    if not text:
        return [], []
    text = _NULLS.sub(" ", text)
    text = text.replace("'", " ")
    text = _ORDINAL_RE.sub(r"\1", text)
    text = _LETTER_DIGIT.sub(" ", text)
    toks = [t for t in _ADDR_SPLIT.split(text) if t]
    ck = _country_key(country)
    out, i = [], 0
    while i < len(toks):
        # multi-word state names -> code
        matched = False
        for n in range(_MAX_STATE_WORDS, 0, -1):
            key = (ck, tuple(toks[i:i + n]))
            if key in _STATE_PHRASES:
                out.append(_STATE_PHRASES[key])
                i += n
                matched = True
                break
        if matched:
            continue
        t = toks[i]
        i += 1
        if t.isdigit():
            t = t.lstrip("0") or "0"
            out.append(t)
            continue
        if ck == "france" and t == "r":
            t = "rue"
        if ck == "france" and t == "n" and i < len(toks) and toks[i].isdigit():
            continue  # N° 162 -> 162
        t = _STREET_MAP.get(t, t)
        if t in _ADDR_DROP:
            continue
        out.append(t)
    nums = [t for t in out if t.isdigit()]
    return out, nums

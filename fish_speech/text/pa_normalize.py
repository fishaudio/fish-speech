"""Punjabi (Gurmukhi) to diacritic-aware Latin transliteration.

This is an OPT-IN preprocessing pass.
See PUNJABI_TRANSLITERATION.md in this directory for the
full mapping table and rationale.

Scheme summary (based on ISO 15919):
    - Long vowels get a macron:              ā ī ū ē ō
    - Retroflex consonants get a dot below:   ṭ ṭh ḍ ḍh ṇ
    - Gemination (addak, U+0A71) doubles the *following* consonant, not the
    preceding one: ਗੱਲ (ਗ + ੱ + ਲ) -> "gall", not "ggal".
    addak doubles the next consonant, not the one before it.
    - Nasalization (bindi U+0A02 / tippi U+0A70) is marked with a
      trailing 'ṁ' attached to the syllable it modifies. Tippi is used in gemination for nasal consonants ਙ, ਞ, ਨ and ਮ.
    - Tippi before a non-nasal consonant (ਜ, ਬ, etc.) → nasalize the vowel before it.
    ਪੰਜਾਬ → paṁjāb.
    - The inherent vowel ("schwa") that bare consonant letters imply
      is dropped or kept using the rule in `_apply_schwa_deletion`
    - consonants mapped are from official Unicode Standard, Gurmukhi block chart.
    - when a consonant is immediately followed by virama, mark its schwa as definitively cancelled (no need to guess, since virama explicitly says so), consume the virama, and let the next consonant start fresh right after it.
    - for schwa deletion - we delete the schwa at the literal end of a word, and within a run of adjacent bare consonants, alternate delete/keep starting from the rightmost one — which exactly reproduces the one full worked example from the issue.

LIMITATION:
    - The transliteration does not handle all possible combinations of consonants and vowels, especially in borrowed words or names. Some combinations may not be accurately represented in the Latin script.
    - The transliteration does not account for regional variations in pronunciation (out of scope), which may affect how certain letters are pronounced and thus how they should be transliterated.
    - the unmarked/implicit schwa deletion rule is not always accurate since its not a direct part of ISO 15919, and may result in incorrect transliterations for some words. This is a known limitation of the current implementation.
    - The ISO 15919 standard is not fully implemented, and some diacritic marks may not be applied correctly in all cases. Users should be aware of this limitation when using the transliteration function.
    - strict ISO 15919 has no mechanism to silently drop schwas — it either writes "a" literally, or relies on a virama being present in the source text.
    - the schwa rule for deletion was derived from a single sentence, not a full phonological model, so it will get some real words wrong — there's no code fix that gets this to 100%, since even published Hindi schwa-deletion research doesn't reach that. Closing the gap means testing against a much bigger set of real words and native-speaker corrections over time, not a one-time patch.
"""

import unicodedata

INDEPENDENT_VOWELS = {
    "ਅ": "a",
    "ਆ": "ā",
    "ਇ": "i",
    "ਈ": "ī",
    "ਉ": "u",
    "ਊ": "ū",
    "ਏ": "ē",
    "ਐ": "ai",
    "ਓ": "ō",
    "ਔ": "au",
}

VOWEL_SIGNS = {
    "ਾ": "ā",
    "ਿ": "i",
    "ੀ": "ī",
    "ੁ": "u",
    "ੂ": "ū",
    "ੇ": "ē",
    "ੈ": "ai",
    "ੋ": "ō",
    "ੌ": "au",
}

CONSONANTS = {
    "ਕ": "k",
    "ਖ": "kh",
    "ਗ": "g",
    "ਘ": "gh",
    "ਙ": "ṅ",
    "ਚ": "c",
    "ਛ": "ch",
    "ਜ": "j",
    "ਝ": "jh",
    "ਞ": "ñ",
    "ਟ": "ṭ",
    "ਠ": "ṭh",
    "ਡ": "ḍ",
    "ਢ": "ḍh",
    "ਣ": "ṇ",
    "ੜ": "ṛ",  # retroflex consonants
    "ਤ": "t",
    "ਥ": "th",
    "ਦ": "d",
    "ਧ": "dh",
    "ਨ": "n",
    "ਪ": "p",
    "ਫ": "ph",
    "ਬ": "b",
    "ਭ": "bh",
    "ਮ": "m",
    "ਯ": "y",
    "ਰ": "r",
    "ਲ": "l",
    "ਵ": "v",
    "ਸ": "s",
    "ਹ": "h",
}

# Consonants formed with a nukta (਼) for borrowed sounds.
NUKTA_CONSONANTS = {
    "ਸ਼": "ś",
    "ਜ਼": "z",
    "ਖ਼": "x",
    "ਗ਼": "ġ",
    "ਫ਼": "f",
    "ਲ਼": "ḷ",
}

ADDAK = "\u0a71"  # ੱ gemination marker
NUKTA = "\u0a3c"  # ਼ combining nukta
BINDI = "\u0a02"  # ਂ nasalization
TIPPI = "\u0a70"  # ੰ nasalization
VIRAMA = "\u0a4d"  # ੍ explicitly cancels the schwa, forms a cluster

PUNCTUATION = {"।": ".", "॥": "."}


def _is_consonant(ch: str) -> bool:
    return ch in CONSONANTS


def _tokenize_word(word: str) -> list[dict]:
    """Break one Gurmukhi word into syllable units.

    Each unit is a dict describing one syllable:
        kind: 'V' (independent vowel), 'CV' (consonant + explicit
              vowel), 'C' (bare consonant, a schwa-deletion
              candidate) or 'CO' (bare consonant with virama, schwa definitely cancelled)
        latin: the consonant/vowel's Latin form (already geminated
               if an addak applied)
        vowel: the explicit vowel's Latin form, only for kind 'CV'
        nasal: True if a bindi/tippi attaches to this syllable
    """
    units = []
    i, n = 0, len(word)
    pending_gem = False

    while i < n:
        ch = word[i]

        if ch == ADDAK:
            pending_gem = True
            i += 1
            continue

        if ch in INDEPENDENT_VOWELS:
            units.append({"kind": "V", "latin": INDEPENDENT_VOWELS[ch]})
            pending_gem = False  # an orphan addak must not bleed past a vowel
            i += 1
            continue

        if _is_consonant(ch):
            j = i + 1
            if j < n and word[j] == NUKTA and (ch + NUKTA) in NUKTA_CONSONANTS:
                latin = NUKTA_CONSONANTS[ch + NUKTA]
                j += 1
            else:
                latin = CONSONANTS[ch]

            if pending_gem:
                latin = latin * 2
                pending_gem = False

            has_virama = False
            if j < n and word[j] == VIRAMA:
                has_virama = True
                j += 1

            vowel = None
            if not has_virama and j < n and word[j] in VOWEL_SIGNS:
                vowel = VOWEL_SIGNS[word[j]]
                j += 1

            nasal = False
            if j < n and word[j] in (BINDI, TIPPI):
                nasal = True
                j += 1

            units.append(
                {
                    "kind": (
                        "CO" if has_virama else ("CV" if vowel is not None else "C")
                    ),
                    "latin": latin,
                    "vowel": vowel,
                    "nasal": nasal,
                }
            )
            i = j
            continue

        # Unrecognized character inside a Gurmukhi run — pass through.
        units.append({"kind": "?", "latin": ch})
        i += 1

    return units


def _apply_schwa_deletion(units: list[dict]) -> None:
    """Sets `keep_schwa` on each bare ('C') unit, in place.

    Rule (derived from, and verified exactly against, the issue's
    worked example):
      - Bare consonants form runs of consecutive 'C' units (an addak
        does not break a run, since it carries no vowel of its own).
      - A run of length 1 that is NOT at the literal end of the word
        keeps its schwa (e.g. ਵ in ਵਧੀਆ -> "va").
      - Every other run (length 1 at word-end, or length >= 2
        anywhere) is resolved right-to-left, alternating starting
        with DELETE at the rightmost unit (e.g. [ਕ,ਰ] in ਕਰਕੇ ->
        ਰ deletes, ਕ keeps -> "kar"; ਲ in ਗੱਲ deletes -> "gall").
      - KNOWN LIMITATION: for runs of length >= 3, this alternation
      produces linguistically incorrect output. e.g. ਕਮਲ ("kamal")
      -> "kmal", ਕਲਮ ("kalam") -> "klam".
      Fixing this requires a
      real schwa-deletion model rather than a one-line tweak; see
      PUNJABI_TRANSLITERATION.md "Known limitations" for context.
    """
    n = len(units)
    i = 0
    while i < n:
        if units[i]["kind"] == "C":
            j = i
            while j < n and units[j]["kind"] == "C":
                j += 1
            run = units[i:j]
            is_word_final_run = j == n

            if len(run) == 1 and not is_word_final_run:
                run[0]["keep_schwa"] = True
            else:
                for idx, unit in enumerate(reversed(run)):
                    unit["keep_schwa"] = idx % 2 == 1
            i = j
        else:
            i += 1


def _render_word(units: list[dict]) -> str:
    _apply_schwa_deletion(units)
    out = []
    for u in units:
        out.append(u["latin"])
        if u["kind"] == "CV":
            out.append(u["vowel"] + ("ṁ" if u["nasal"] else ""))
        elif u["kind"] == "C":
            if u.get("keep_schwa"):
                out.append("a" + ("ṁ" if u["nasal"] else ""))
            elif u["nasal"]:
                out.append("ṁ")
        elif u["kind"] == "CO":
            if u["nasal"]:
                out.append("ṁ")
    return "".join(out)


def normalize_punjabi(text: str) -> str:
    """Transliterate Gurmukhi text into a diacritic-aware Latin form.

    Non-Gurmukhi characters (spaces, Latin text, digits, other
    punctuation) pass through unchanged, so this is safe to run on
    mixed-script input — only Gurmukhi runs are rewritten.
    """
    text = unicodedata.normalize("NFC", text)
    # this ensures every one-character nukta letter becomes base letter plus nukta before any of the code looks at it.

    out = []
    i, n = 0, len(text)
    while i < n:
        ch = text[i]

        if ch in PUNCTUATION:
            out.append(PUNCTUATION[ch])
            i += 1
            continue

        if ch in INDEPENDENT_VOWELS or _is_consonant(ch):
            j = i
            while j < n and (
                text[j] in INDEPENDENT_VOWELS
                or _is_consonant(text[j])
                or text[j] in VOWEL_SIGNS
                or text[j] in (ADDAK, NUKTA, BINDI, TIPPI, VIRAMA)
            ):
                j += 1
            word = text[i:j]
            out.append(_render_word(_tokenize_word(word)))
            i = j
            continue

        out.append(ch)
        i += 1

    return "".join(out)


"""the above loop that decides how far a single "Gurmukhi word" extends before handing it off to _tokenize_word()."""

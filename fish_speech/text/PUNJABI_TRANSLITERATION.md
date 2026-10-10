# Punjabi (Gurmukhi) → Latin Transliteration Scheme

This document specifies the mapping used in `fish_speech/text/pa_normalize.py`
to convert Gurmukhi text into a diacritic-aware Latin transliteration before
it's passed to the model, when the `--punjabi-normalize` flag of
`fish_speech/models/text2semantic/inference.py` is set.

## Why

Raw Gurmukhi text and a plain, ambiguous Latin romanization both produce
noticeably worse pronunciation than an explicit, diacritic-marked Latin
spelling. This scheme makes that explicit spelling automatic, so users can
type normally in Gurmukhi and get the better pronunciation without knowing
this conversion happens.

The vowel and consonant symbols follow ISO 15919/IAST convention. The
schwa-deletion and addak-gemination rules below are **not** part of ISO
15919 — that standard always writes the inherent vowel as `a` literally and
has no mechanism to drop it without an explicit virama in the source text.
These are a separate, deliberate phonetic layer built on top of the ISO
symbol set, since the goal here is pronunciation, not lossless, reversible
transliteration.

## Vowels

| Gurmukhi | Independent (word-initial) | As a vowel sign (matra) | Latin |
|---|---|---|---|
| ਅ / — | ਅ | (none — inherent vowel) | a |
| ਆ / ਾ | ਆ | ਾ | ā |
| ਇ / ਿ | ਇ | ਿ | i |
| ਈ / ੀ | ਈ | ੀ | ī |
| ਉ / ੁ | ਉ | ੁ | u |
| ਊ / ੂ | ਊ | ੂ | ū |
| ਏ / ੇ | ਏ | ੇ | ē |
| ਐ / ੈ | ਐ | ੈ | ai |
| ਓ / ੋ | ਓ | ੋ | ō |
| ਔ / ੌ | ਔ | ੌ | au |

Long vowels get a macron (ā, ī, ū, ē, ō); short vowels don't. This is the
single biggest source of pronunciation improvement over plain Latin
spelling — e.g. `tuhāḍē` vs. the ambiguous `tuhade`. Gurmukhi always writes
these vowel signs explicitly (unlike, say, Arabic), so there's no ambiguity
between a bare consonant and one with a long vowel — `ਕ` is always short
`a`, `ਕਾ` is always long `ā`.

## Consonants

| Gurmukhi | Latin | Gurmukhi | Latin | Gurmukhi | Latin |
|---|---|---|---|---|---|
| ਕ | k | ਟ | ṭ | ਤ | t |
| ਖ | kh | ਠ | ṭh | ਥ | th |
| ਗ | g | ਡ | ḍ | ਦ | d |
| ਘ | gh | ਢ | ḍh | ਧ | dh |
| ਙ | ṅ | ਣ | ṇ | ਨ | n |
| ਚ | c | ਪ | p | ਯ | y |
| ਛ | ch | ਫ | ph | ਰ | r |
| ਜ | j | ਬ | b | ਲ | l |
| ਝ | jh | ਭ | bh | ਵ | v |
| ਞ | ñ | ਮ | m | ਸ | s |
| ੜ | ṛ |   |   | ਹ | h |

Retroflex consonants (ਟ ਠ ਡ ਢ ਣ ੜ) get a dot below (ṭ ṭh ḍ ḍh ṇ ṛ) to
distinguish them from their dental counterparts (ਤ ਥ ਦ ਧ ਨ) — this
distinction is completely lost in plain Latin spelling (both map to
`t`/`d`/`n`).

### Nukta-modified consonants (borrowed sounds)

| Gurmukhi | Latin |
|---|---|
| ਸ਼ | ś |
| ਜ਼ | z |
| ਖ਼ | x |
| ਗ਼ | ġ |
| ਫ਼ | f |
| ਲ਼ | ḷ |

## Gemination (addak, `ੱ` U+0A71)

Addak doubles the consonant that **follows** it, not the one it's visually
adjacent to on the left. `ਗੱਲ` (ਗ + ੱ + ਲ) → `gall`, not `ggal`.

## Nasalization (bindi `ਂ` U+0A02 / tippi `ੰ` U+0A70)

Both are rendered as a trailing `ṁ` attached to the syllable they modify —
whether that syllable has an explicit vowel (`ਸਿੰਘ` → `siṁgh`) or is a bare
consonant carrying its own (kept) inherent vowel (`ਪੰਜਾਬ` → `paṁjāb`).

**Known gap:** tippi has a second function this scheme doesn't yet
implement. Before the nasal consonants ਙ, ਞ, ਨ, or ਮ specifically, tippi
can signal *gemination* of that consonant instead of nasalizing the
preceding syllable — e.g. `ਕੰਮ` ("work") is pronounced `kamm` (doubled m),
not `kaṁm`. The current code always treats tippi as nasalization, so this
case is currently handled incorrectly. Fixing it requires a rule to
disambiguate the two readings and hasn't been validated against enough
real examples yet — flagged here rather than guessed at.

## Explicit schwa cancellation (virama, `੍` U+0A4D)

Virama explicitly and unambiguously kills a consonant's inherent vowel and
forms a direct cluster with the next consonant — no inference needed,
unlike the heuristic below. `ਸ੍ਰੀ` (a common name/loanword pattern) →
`srī`, not `sarī` or `s(virama)rī`.

## Inherent vowel ("schwa") deletion

Every Gurmukhi consonant letter with no explicit vowel sign and no virama
carries an implicit "a" sound (traditionally called a schwa) that is often
— but not always — dropped in speech. Gurmukhi orthography essentially
never marks this in writing, so it has to be inferred. Getting this right
is the hardest part of the scheme. The rule implemented here:

1. Find runs of consecutive bare consonants (an addak doesn't break a run,
   since it carries no vowel of its own; a virama-cancelled consonant is
   excluded from these runs entirely, since its schwa is already known,
   not inferred).
2. A run of exactly one bare consonant that is **not** at the literal end
   of the word keeps its schwa.
   - `ਵਧੀਆ` → `vadhīā` (ਵ is isolated and not at the end of the word → keeps "a")
3. Every other run — length 1 at the end of a word, or length 2+ anywhere
   — is resolved right-to-left, alternating starting with **delete** at
   the rightmost consonant of the run:
   - `ਦਿਲ` → dil (ਲ is a length-1 run at word end → deletes)
   - `ਮੌਸਮ` → mausam (ਸ,ਮ run: ਮ deletes, ਸ keeps → "sam")
   - `ਕਰਕੇ` → karkē (ਕ,ਰ run: ਰ deletes, ਕ keeps → "kar")

**This is a heuristic, not a full phonological analysis.** It was derived
directly from, and exactly reproduces, the example sentence in the
originating issue. It has not been validated against a broader vocabulary
— see Known Limitations below.

## Punctuation

`।` and `॥` (danda / double danda) map to `.`. Everything else — Latin
text, digits, whitespace, other punctuation — passes through unchanged,
so this function is safe to run on mixed-script input.

## Open question for native-speaker review

The issue's own example sentence is internally inconsistent on one point:
the vowel sign `ੇ` is rendered with a macron in some words (`tē`,
`tuhāḍē`) and without one in others (`karke`, `merā`), even though it's
the same underlying Gurmukhi character each time. This scheme applies the
macron **consistently** (`ē` everywhere), which means it renders those two
words as `karkē` / `mērā` rather than the example's `karke` / `merā`.
Worth confirming with a native speaker whether that inconsistency reflects
a real pronunciation distinction being missed, or was a typo in the
hand-typed example.

## Known limitations

- **Tippi's nasal-gemination role (see above)** is a real, confirmed gap —
  not yet implemented.
- **Regional/dialectal pronunciation variation** (Majhi, Malwai, Doabi,
  etc.) is out of scope — no grapheme-level rule set can capture dialect
  choice; that would need separate rule profiles per dialect.
- **The schwa-deletion rule was derived from a single worked example**, not
  a full phonological model. It will get some real words wrong. There's no
  code fix that gets this to 100% — even published Hindi schwa-deletion
  research doesn't reach that. Closing the gap means testing against a
  much larger set of real words and incorporating native-speaker
  corrections over time, not a one-time patch.
- **Very rare or unusual Unicode sequences** outside the categories
  documented above (e.g. uncommon combining-mark stacking not seen in any
  test example) may not be handled correctly — the tokenizer is
  compositional, so all common consonant/vowel/mark combinations are
  covered systematically, but it hasn't been fuzzed against exhaustive
  edge cases.
- **Runs of three or more consecutive bare consonants are resolved incorrectly.** 
  The right-to-left alternation rule (delete rightmost,
  alternate) matches the issue's worked example but does not generalize:
  `ਕਮਲ` ("kamal") is rendered `kmal`, `ਕਲਮ` ("kalam") as `klam`. A correct
  rule requires a real schwa-deletion model, which is out of scope for this
  first pass. This behaviour is pinned by a unit test so a future fix is a
  visible change.

## Testing status

Verified exactly against the worked example from the originating issue,
with the caveat that this scheme applies the macron consistently to `ੇ`
(`karkē`/`mērā`) where the issue text renders it inconsistently
(`karke`/`merā`) — see "Open question for native-speaker review" above.
Targeted unit tests also cover each rule above (addak, tippi/bindi
nasalization, retroflex consonants, schwa-deletion alternation, virama,
nukta consonants). 

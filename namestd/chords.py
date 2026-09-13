"""Chord symbols: parsing, transposition, and chroma templates.

The corpus writes chords in iReal Pro's shorthand (``Bbmaj7``, ``Em7b5``,
``B07``, ``Bb6/F``).  We reduce every symbol to a root pitch class plus one of
six *quality classes*.  Six is deliberate: it is the coarsest split that still
separates chords a listener would never confuse, and finer distinctions (``7b9``
vs ``7#11``) are exactly the ones improvisers alter freely, so encoding them
would model the chart rather than the performance.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np

PITCH_CLASSES = ["C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "B"]

_NOTE_BASE = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}

# The six quality classes.
MAJ, MIN, DOM, HDIM, DIM, SUS = "maj", "min", "dom", "hdim", "dim", "sus"
QUALITIES = [MAJ, MIN, DOM, HDIM, DIM, SUS]
QUALITY_INDEX = {q: i for i, q in enumerate(QUALITIES)}

# Every quality suffix observed in the corpus, mapped to its class.  Suffixes are
# matched longest-first, so ``m7b5`` wins over ``m7`` and ``maj7`` over ``m``.
_SUFFIX_TO_QUALITY = {
    # major family
    "": MAJ, "6": MAJ, "69": MAJ, "add9": MAJ,
    "maj7": MAJ, "maj9": MAJ, "maj13": MAJ,
    "maj7#11": MAJ, "maj9#11": MAJ, "maj7#5": MAJ,
    # minor family
    "m": MIN, "m6": MIN, "m7": MIN, "m9": MIN, "m11": MIN, "m69": MIN,
    "mb6": MIN, "m7b9": MIN, "maug": MIN,
    # dominant family
    "7": DOM, "9": DOM, "13": DOM, "aug": DOM,
    "7b9": DOM, "7#9": DOM, "7#5": DOM, "7#11": DOM, "7b13": DOM, "7b5": DOM,
    "7alt": DOM, "13b9": DOM, "13#11": DOM, "9#11": DOM, "9#5": DOM, "9b5": DOM,
    "7#5#9": DOM, "7b9#5": DOM, "7b5b9": DOM, "7b9b13": DOM, "7b5#9": DOM,
    "7#9#11": DOM, "7b9#11": DOM, "7b9#9": DOM,
    # half-diminished / diminished
    "m7b5": HDIM,
    "0": DIM, "07": DIM,
    # suspended
    "sus": SUS, "7sus": SUS, "9sus": SUS, "13sus": SUS,
    "7b9sus": SUS, "7b13sus": SUS,
}

_SUFFIXES_BY_LENGTH = sorted(_SUFFIX_TO_QUALITY, key=len, reverse=True)

_ROOT_RE = re.compile(r"^([A-G][b#]?)")
# Parenthesised chords are iReal's "alternate changes" annotation.  They are one
# player's substitution, not what the band agreed on, so we drop them.
_PARENS_RE = re.compile(r"\([^)]*\)")


def note_to_pc(name: str) -> int:
    """``"Bb"`` -> 10.  Raises ValueError on anything that isn't a note name."""
    m = _ROOT_RE.match(name)
    if not m or m.end() != len(name):
        raise ValueError(f"not a note name: {name!r}")
    text = m.group(1)
    pc = _NOTE_BASE[text[0]]
    if len(text) > 1:
        pc += 1 if text[1] == "#" else -1
    return pc % 12


@dataclass(frozen=True)
class Chord:
    """A root pitch class, a quality class, and an optional slash bass."""

    root: int
    quality: str
    bass: int | None = None

    def transpose(self, semitones: int) -> "Chord":
        return Chord(
            (self.root + semitones) % 12,
            self.quality,
            None if self.bass is None else (self.bass + semitones) % 12,
        )

    @property
    def bass_pc(self) -> int:
        """Pitch class the bassist is most likely to be sitting on."""
        return self.root if self.bass is None else self.bass

    def __str__(self) -> str:
        text = PITCH_CLASSES[self.root] + {
            MAJ: "maj", MIN: "m", DOM: "7", HDIM: "m7b5", DIM: "dim", SUS: "sus",
        }[self.quality]
        if self.bass is not None:
            text += "/" + PITCH_CLASSES[self.bass]
        return text


def parse_chord(symbol: str) -> Chord | None:
    """Parse one iReal-style chord symbol.  Returns None for empty/unparseable."""
    text = _PARENS_RE.sub("", symbol).strip()
    if not text:
        return None

    bass = None
    if "/" in text:
        text, _, bass_text = text.partition("/")
        try:
            bass = note_to_pc(bass_text.strip())
        except (ValueError, KeyError):
            bass = None

    m = _ROOT_RE.match(text)
    if not m:
        return None
    try:
        root = note_to_pc(m.group(1))
    except KeyError:
        return None

    suffix = text[m.end():].strip()
    quality = _SUFFIX_TO_QUALITY.get(suffix)
    if quality is None:
        # Unknown extension: fall back on the leading characters, which carry the
        # family.  "m<something>" is minor, "maj..." major, otherwise dominant.
        if suffix.startswith("maj"):
            quality = MAJ
        elif suffix.startswith("m"):
            quality = MIN
        elif suffix.startswith("sus"):
            quality = SUS
        else:
            quality = DOM

    if bass is not None and bass == root:
        bass = None
    return Chord(root, quality, bass)


# --- chroma templates -------------------------------------------------------
#
# Weights are deliberately uneven.  The third and seventh are what distinguish
# qualities; the fifth is shared by nearly every chord and so carries almost no
# discriminative information.  The root is weighted lower in the treble template
# than you might expect because jazz pianists play rootless voicings - the root
# is the bass player's job, which is why we keep a separate bass stream.

_TREBLE_INTERVALS: dict[str, dict[int, float]] = {
    MAJ:  {0: 0.7, 4: 1.0, 7: 0.4, 11: 0.9, 9: 0.5, 2: 0.3},
    MIN:  {0: 0.7, 3: 1.0, 7: 0.4, 10: 0.8, 2: 0.3, 5: 0.2},
    DOM:  {0: 0.7, 4: 1.0, 7: 0.3, 10: 0.9, 2: 0.3, 9: 0.2},
    HDIM: {0: 0.7, 3: 1.0, 6: 0.8, 10: 0.8, 1: 0.2},
    DIM:  {0: 0.8, 3: 0.9, 6: 0.9, 9: 0.9},
    SUS:  {0: 0.8, 5: 1.0, 7: 0.5, 10: 0.8, 2: 0.4},
}


def treble_template(chord: Chord) -> np.ndarray:
    """Unit-norm 12-vector of expected chroma energy for a chord's upper voices."""
    vec = np.zeros(12, dtype=np.float32)
    for interval, weight in _TREBLE_INTERVALS[chord.quality].items():
        vec[(chord.root + interval) % 12] += weight
    norm = np.linalg.norm(vec)
    return vec / norm if norm else vec


def bass_template(chord: Chord) -> np.ndarray:
    """Unit-norm 12-vector for the bass register.

    A walking bassline hits the root hardest but spends real time on the fifth
    and the approach tone below the next root, so the template is peaked rather
    than one-hot - a one-hot template would punish ordinary walking.
    """
    vec = np.zeros(12, dtype=np.float32)
    bass = chord.bass_pc
    vec[bass] += 1.0
    vec[(chord.root + 7) % 12] += 0.35
    third = {MAJ: 4, DOM: 4, MIN: 3, HDIM: 3, DIM: 3, SUS: 5}[chord.quality]
    vec[(chord.root + third) % 12] += 0.25
    if chord.bass is not None:
        vec[chord.root] += 0.4
    norm = np.linalg.norm(vec)
    return vec / norm if norm else vec


def template_matrix(chords: list[Chord]) -> np.ndarray:
    """Stack bass+treble templates into an (n, 24) matrix."""
    if not chords:
        return np.zeros((0, 24), dtype=np.float32)
    return np.stack(
        [np.concatenate([bass_template(c), treble_template(c)]) for c in chords]
    ).astype(np.float32)


def all_chords() -> list[Chord]:
    """The 72-chord alphabet: 12 roots x 6 qualities, no slash bass."""
    return [Chord(r, q) for q in QUALITIES for r in range(12)]

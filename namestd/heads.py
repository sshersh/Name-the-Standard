"""Generate a distinct head for each tune, for evaluating the melody stage.

These are **not transcriptions.**  Real jazz heads are copyrighted, and the
chord corpus carries no melodies at all, so there is nothing to load.  What this
module provides is a deterministic bebop-style line over a tune's own changes -
different for every tune, playable, and shaped like the real thing: mostly
eighth notes, chord tones on the strong beats, chromatic approaches into the
next chord.

That is enough to answer the engineering question the melody stage exists to
answer - *given* a melody index, how reliably does the rerank separate tunes
that share changes, under realistic pitch-extraction noise? - without shipping
protected material.  A production system would build its index from licensed
lead sheets or by transcribing canonical recordings, storing only the derived
interval sequences (see melody.py).
"""

from __future__ import annotations

import hashlib

import numpy as np

from .chords import DIM, DOM, HDIM, MAJ, MIN, SUS, Chord
from .corpus import Song

# Chord tones and the tensions a bebop line actually lands on.
_TONES: dict[str, list[int]] = {
    MAJ:  [0, 4, 7, 9, 11],
    MIN:  [0, 3, 5, 7, 10],
    DOM:  [0, 4, 7, 10, 2],
    HDIM: [0, 3, 6, 10],
    DIM:  [0, 3, 6, 9],
    SUS:  [0, 5, 7, 10, 2],
}

LOW, HIGH = 60, 84       # C4 .. C6, a horn's comfortable range


def _seed_for(title: str) -> int:
    digest = hashlib.sha256(title.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


def generate_head(song: Song, notes_per_beat: int = 2) -> tuple[np.ndarray, np.ndarray]:
    """A line over one chorus.  Returns (beat positions, MIDI pitches).

    Deterministic in the tune's title, so the same tune always gets the same
    head and two tunes on identical changes still get different melodies - which
    is exactly the case the rerank has to resolve.
    """
    rng = np.random.default_rng(_seed_for(song.title))
    chords = song.chords

    positions: list[float] = []
    pitches: list[int] = []
    current = int(rng.integers(LOW + 4, HIGH - 8))

    for beat_index, chord in enumerate(chords):
        next_chord = chords[(beat_index + 1) % len(chords)]
        for step in range(notes_per_beat):
            strong = (step == 0)
            if strong or rng.random() < 0.55:
                tone = int(rng.choice(_TONES[chord.quality]))
                target_pc = (chord.root + tone) % 12
            elif next_chord != chord and step == notes_per_beat - 1:
                # Chromatic approach into the next chord's root.
                target_pc = (next_chord.root + int(rng.choice([-1, 1]))) % 12
            else:
                # Passing tone: step away from where we are.
                target_pc = (current + int(rng.choice([-2, -1, 1, 2]))) % 12

            # Move to the nearest instance of that pitch class, so the line is
            # stepwise rather than leaping around the register.
            delta = (target_pc - current % 12 + 6) % 12 - 6
            candidate = current + delta
            if candidate < LOW:
                candidate += 12
            elif candidate > HIGH:
                candidate -= 12
            # Occasional leap, the way a bebop line breaks up its running eighths.
            if rng.random() < 0.10:
                candidate += 12 * int(rng.choice([-1, 1]))
                candidate = int(np.clip(candidate, LOW, HIGH))

            current = int(candidate)
            # Rests: a real head breathes at phrase ends.
            if not (step and rng.random() < 0.18):
                positions.append(beat_index + step / notes_per_beat)
                pitches.append(current)

    return np.asarray(positions, dtype=float), np.asarray(pitches, dtype=int)


def build_index(songs: list[Song], titles: list[str] | None = None):
    """A MelodyIndex over generated heads, for evaluation."""
    from .melody import MelodyIndex

    index = MelodyIndex()
    wanted = set(titles) if titles is not None else None
    for song in songs:
        if wanted is None or song.title in wanted:
            _, pitches = generate_head(song)
            index.add(song.title, list(pitches))
    return index

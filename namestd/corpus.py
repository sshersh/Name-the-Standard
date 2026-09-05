"""Load the chord corpus and expand each chart into a flat, beat-level form.

Charts are written the way a human reads them - sections, repeats, first/second
endings.  A matcher wants the opposite: one flat array with a chord for every
beat of one full chorus, so that a query can be slid against it.  This module
does that expansion once, at load time.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .chords import Chord, QUALITY_INDEX, QUALITIES, parse_chord

DEFAULT_DATA = Path(__file__).resolve().parent.parent / "data" / "raw" / "JazzStandards.json"


@dataclass
class Song:
    """One tune, expanded to a chord per beat over exactly one chorus."""

    title: str
    composer: str
    key: str | None
    rhythm: str
    beats_per_bar: int
    chords: list[Chord]

    @property
    def n_beats(self) -> int:
        return len(self.chords)

    @property
    def n_bars(self) -> int:
        return len(self.chords) // self.beats_per_bar

    def codes(self) -> np.ndarray:
        """Beat-level ``root * 6 + quality`` codes, for fast comparison."""
        return np.array(
            [c.root * len(QUALITIES) + QUALITY_INDEX[c.quality] for c in self.chords],
            dtype=np.int16,
        )

    def bar_strings(self) -> list[str]:
        """Human-readable bars, for debugging and for explaining a match."""
        bars = []
        for start in range(0, self.n_beats, self.beats_per_bar):
            beat_chords = self.chords[start:start + self.beats_per_bar]
            uniq: list[Chord] = []
            for c in beat_chords:
                if not uniq or c != uniq[-1]:
                    uniq.append(c)
            bars.append(" ".join(str(c) for c in uniq))
        return bars


def _split_bar_to_beats(bar: str, beats_per_bar: int, carry: Chord | None
                        ) -> tuple[list[Chord | None], Chord | None]:
    """Distribute the chords written in one bar across that bar's beats.

    An empty slot (the corpus writes trailing commas, e.g. ``"G7,Ab7,G7,"``)
    means "hold the previous chord", so it extends whatever came before rather
    than introducing a rest.
    """
    slots = [parse_chord(tok) for tok in bar.split(",")]
    # Resolve empty slots against the running chord.
    resolved: list[Chord | None] = []
    for chord in slots:
        if chord is None:
            chord = resolved[-1] if resolved else carry
        resolved.append(chord)
    resolved = [c for c in resolved] or [carry]

    n = len(resolved)
    if n >= beats_per_bar:
        beats = resolved[:beats_per_bar]
    else:
        # Spread evenly, giving the surplus beats to the earlier chords - the
        # convention for three chords in a 4/4 bar is 2 + 1 + 1.
        base, extra = divmod(beats_per_bar, n)
        beats = []
        for i, chord in enumerate(resolved):
            beats.extend([chord] * (base + (1 if i < extra else 0)))

    last = next((c for c in reversed(beats) if c is not None), carry)
    return beats, last


def _expand_segment(text: str, beats_per_bar: int, carry: Chord | None
                    ) -> tuple[list[Chord | None], Chord | None]:
    beats: list[Chord | None] = []
    for bar in text.split("|"):
        if not bar.strip() and not beats:
            continue
        bar_beats, carry = _split_bar_to_beats(bar, beats_per_bar, carry)
        beats.extend(bar_beats)
    return beats, carry


def _expand_song(raw: dict) -> Song | None:
    time_sig = raw.get("TimeSignature") or "4/4"
    try:
        beats_per_bar = int(time_sig.split("/")[0])
    except (ValueError, IndexError):
        beats_per_bar = 4
    if beats_per_bar < 2:
        return None

    beats: list[Chord | None] = []
    carry: Chord | None = None
    for section in raw.get("Sections", []):
        main = (section.get("MainSegment") or {}).get("Chords", "")
        endings = section.get("Endings") or []
        # "Repeats: 1" means play the section one extra time.
        plays = int(section.get("Repeats") or 0) + 1
        if endings:
            # N endings means N passes: main + ending[k] on pass k.
            plays = max(plays, len(endings))

        for pass_index in range(plays):
            body, carry = _expand_segment(main, beats_per_bar, carry)
            beats.extend(body)
            if endings:
                ending = endings[min(pass_index, len(endings) - 1)]
                tail, carry = _expand_segment(ending.get("Chords", ""), beats_per_bar, carry)
                beats.extend(tail)

    # Charts sometimes open on an empty slot; back-fill from the first real chord
    # so every beat carries harmony.
    first = next((c for c in beats if c is not None), None)
    if first is None:
        return None
    chords = [c if c is not None else first for c in beats]

    # Trim a ragged final bar rather than let it shift the form length.
    usable = (len(chords) // beats_per_bar) * beats_per_bar
    chords = chords[:usable]
    if len(chords) < beats_per_bar * 4:
        return None

    return Song(
        title=raw.get("Title", "?"),
        composer=raw.get("Composer", ""),
        key=raw.get("Key"),
        rhythm=raw.get("Rhythm", ""),
        beats_per_bar=beats_per_bar,
        chords=chords,
    )


def load_corpus(path: Path | str | None = None) -> list[Song]:
    """Load and expand every chart.  Charts that fail to expand are skipped."""
    path = Path(path) if path else DEFAULT_DATA
    if not path.exists():
        raise FileNotFoundError(
            f"corpus not found at {path}\nRun: python scripts/fetch_data.py"
        )
    with open(path) as handle:
        raw_songs = json.load(handle)

    songs = []
    for raw in raw_songs:
        try:
            song = _expand_song(raw)
        except Exception:
            song = None
        if song is not None:
            songs.append(song)
    return songs

"""Break contrafact ties using the melody of the head.

Harmony gets you to "a blues in F".  It cannot get you to Blue Monk, because
twenty-eight tunes in the corpus are written on those same twelve bars.  Only
the head separates them, and the head is stated at the top and again at the end
of the performance - which is precisely where the form alignment already tells
us to look.

The rerank runs *inside* one family, against a few dozen candidates rather than
1382, so it can afford to be strict where the harmonic search had to be fast.

Everything here is transposition-invariant by construction: melodies are stored
and compared as sequences of intervals between successive notes, never as
absolute pitches.  That also makes the stored form a poor reproduction of the
tune - you cannot sing a melody back from its interval histogram alone - which
matters, because chord progressions are broadly held to be unprotectable while
melodies are squarely copyrighted.  A production index should keep only these
derived sequences, never reconstructable lead sheets.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# Scoring for interval alignment.  A jazz musician ornaments the written line
# constantly, so the costs are deliberately forgiving: near-misses score
# positively, and gaps are cheap enough that inserted approach notes and
# swallowed notes do not destroy an otherwise good match.
MATCH_EXACT = 3.0
MATCH_NEAR = 0.5        # one semitone out: an ornament, a blue note
MISMATCH = -1.5
GAP_OPEN = -2.0
GAP_EXTEND = -0.4


@dataclass
class MelodyEntry:
    """One head, stored as intervals so it carries no absolute key."""

    title: str
    intervals: list[int]
    durations: list[float] = field(default_factory=list)

    def to_json(self) -> dict:
        return {"title": self.title, "intervals": self.intervals}


@dataclass
class MelodyIndex:
    """Interval sequences for tunes whose heads we know."""

    entries: dict[str, MelodyEntry] = field(default_factory=dict)

    def add(self, title: str, pitches: list[int]) -> None:
        self.entries[title] = MelodyEntry(title, to_intervals(pitches))

    def __contains__(self, title: str) -> bool:
        return title in self.entries

    def __len__(self) -> int:
        return len(self.entries)

    @classmethod
    def load(cls, path: Path | str) -> "MelodyIndex":
        with open(path) as handle:
            raw = json.load(handle)
        index = cls()
        for item in raw:
            index.entries[item["title"]] = MelodyEntry(
                item["title"], list(item["intervals"]))
        return index

    def save(self, path: Path | str) -> None:
        with open(path, "w") as handle:
            json.dump([e.to_json() for e in self.entries.values()], handle, indent=1)


def to_intervals(pitches: list[int] | np.ndarray) -> list[int]:
    """Successive semitone differences, which is what makes this key-blind."""
    pitches = [int(p) for p in pitches]
    return [b - a for a, b in zip(pitches[:-1], pitches[1:])]


# --- extraction from audio --------------------------------------------------

def extract_melody(audio: np.ndarray, sample_rate: int, beat_times: np.ndarray,
                   fmin: float = 130.0, fmax: float = 1200.0
                   ) -> tuple[np.ndarray, np.ndarray]:
    """Predominant pitch track, quantised to notes.

    Returns note pitches (MIDI) and their onset times.  pYIN is a classical
    estimator with no training behind it; a learned tracker (CREPE) or a
    separation front end would do better on a full band, and is the obvious
    place to spend an ML budget if this stage proves to be the bottleneck.
    """
    import librosa

    f0, voiced, _ = librosa.pyin(
        audio, fmin=fmin, fmax=fmax, sr=sample_rate,
        frame_length=2048, hop_length=256,
    )
    times = librosa.times_like(f0, sr=sample_rate, hop_length=256)
    midi = np.full_like(f0, np.nan, dtype=np.float64)
    ok = voiced & np.isfinite(f0)
    midi[ok] = librosa.hz_to_midi(f0[ok])
    return _segment_notes(midi, times)


def _segment_notes(midi: np.ndarray, times: np.ndarray,
                   min_frames: int = 3) -> tuple[np.ndarray, np.ndarray]:
    """Collapse a frame-wise pitch track into discrete notes."""
    pitches: list[int] = []
    onsets: list[float] = []
    run_start = None
    run: list[float] = []

    def flush() -> None:
        if run_start is not None and len(run) >= min_frames:
            pitches.append(int(round(float(np.median(run)))))
            onsets.append(times[run_start])

    for i, value in enumerate(midi):
        if not np.isfinite(value):
            flush()
            run_start, run = None, []
            continue
        # A move of more than a semitone starts a new note; smaller wobble is
        # vibrato or a scoop into the note.
        if run and abs(value - np.median(run)) > 1.0:
            flush()
            run_start, run = i, [value]
        else:
            if run_start is None:
                run_start = i
            run.append(value)
    flush()
    return np.asarray(pitches, dtype=int), np.asarray(onsets, dtype=float)


def head_window(beat_times: np.ndarray, period_beats: int,
                rotation: int, chorus: int = 0) -> tuple[float, float]:
    """Time span of one chorus, given the alignment the matcher already found.

    The head is chorus 0 (and usually the last chorus).  Because the harmonic
    match reports where the query sits in the form, we can cut exactly one
    chorus rather than guessing.
    """
    start_beat = (-rotation) % period_beats + chorus * period_beats
    stop_beat = start_beat + period_beats
    n = len(beat_times)
    if start_beat >= n:
        return 0.0, 0.0
    start = float(beat_times[start_beat])
    stop = float(beat_times[min(stop_beat, n - 1)])
    return start, stop


# --- alignment --------------------------------------------------------------

def align_score(query: list[int], reference: list[int]) -> float:
    """Smith-Waterman local alignment of two interval sequences.

    Local rather than global: a listener catches part of the head, the band
    paraphrases the rest, and only a fragment need line up.  The score is
    normalised by the length of the shorter sequence so that long heads do not
    win on length alone.
    """
    if not query or not reference:
        return 0.0

    n, m = len(query), len(reference)
    previous = np.zeros(m + 1, dtype=np.float32)
    prev_gap = np.full(m + 1, -np.inf, dtype=np.float32)
    best = 0.0

    for i in range(1, n + 1):
        current = np.zeros(m + 1, dtype=np.float32)
        cur_gap = np.full(m + 1, -np.inf, dtype=np.float32)
        q = query[i - 1]
        for j in range(1, m + 1):
            delta = abs(q - reference[j - 1])
            if delta == 0:
                sub = MATCH_EXACT
            elif delta == 1:
                sub = MATCH_NEAR
            else:
                sub = MISMATCH
            diagonal = previous[j - 1] + sub
            # Gaps in either sequence: ornaments added, or notes swallowed.
            cur_gap[j] = max(current[j - 1] + GAP_OPEN, cur_gap[j - 1] + GAP_EXTEND)
            up = max(previous[j] + GAP_OPEN, prev_gap[j] + GAP_EXTEND)
            value = max(0.0, diagonal, cur_gap[j], up)
            current[j] = value
            if value > best:
                best = float(value)
        previous, prev_gap = current, cur_gap

    return best / (MATCH_EXACT * min(n, m))


@dataclass
class MelodyMatch:
    title: str
    score: float


def rerank(candidates: list[str], query_pitches: list[int] | np.ndarray,
           index: MelodyIndex) -> list[MelodyMatch]:
    """Score family members against an extracted head, best first.

    Candidates with no entry in the index are returned with score 0 rather than
    dropped: an unknown head is not evidence against a tune.
    """
    query = to_intervals(query_pitches)
    scored = []
    for title in candidates:
        entry = index.entries.get(title)
        scored.append(MelodyMatch(title, align_score(query, entry.intervals)
                                  if entry else 0.0))
    scored.sort(key=lambda m: -m.score)
    return scored

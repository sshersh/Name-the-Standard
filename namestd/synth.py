"""Render a chart to audio, so the matcher can be evaluated on known ground truth.

Recordings of live jazz with reliable labels are scarce; synthesised choruses
are unlimited and their key, tempo and form offset are known exactly.  That
makes this the harness that isolates matcher bugs from front-end bugs.

The synthesis is deliberately *not* the inverse of the front end.  If it
rendered the chroma templates directly, matching would be trivially perfect and
the evaluation would prove nothing.  Instead it plays notes: a walking bass that
spends most of its time on non-root chord tones and chromatic approaches, piano
voicings that are rootless and carry 9ths and 13ths the templates don't
emphasise, plus drums and noise.  The front end has to work to recover harmony
from that, which is the point.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .chords import DIM, DOM, HDIM, MAJ, MIN, SUS, Chord
from .corpus import Song

SAMPLE_RATE = 22050


@dataclass
class PerformanceStyle:
    """Knobs that vary between performances of the same tune."""

    tempo: float = 160.0          # quarter notes per minute
    transpose: int = 0            # semitones
    n_choruses: int = 3
    swing: float = 0.62           # first of a pair of eighths, as a fraction of the beat
    start_offset_bars: int = 0    # begin mid-form, as when you walk into a club
    # The head is stated on the first and last chorus, with solos in between -
    # the shape of nearly every small-group performance.
    head: tuple[np.ndarray, np.ndarray] | None = None
    head_level: float = 1.0
    bass_level: float = 1.0
    comp_level: float = 0.8
    drums_level: float = 0.5
    noise_level: float = 0.01
    seed: int = 0
    rubato: float = 0.0           # random per-beat timing jitter, as a fraction of a beat


# Rootless voicings, as semitone offsets above the chord root.  These are the
# shapes a pianist actually plays: the root is left to the bass, and the colour
# comes from 9ths and 13ths that the matching templates weight only lightly.
_VOICINGS: dict[str, list[int]] = {
    MAJ:  [4, 7, 11, 14],       # 3 5 7 9
    MIN:  [3, 7, 10, 14],       # b3 5 b7 9
    DOM:  [4, 10, 14, 21],      # 3 b7 9 13
    HDIM: [3, 6, 10, 14],       # b3 b5 b7 9
    DIM:  [3, 6, 9, 14],        # b3 b5 bb7 9
    SUS:  [5, 10, 14, 19],      # 4 b7 9 12
}

_CHORD_TONES: dict[str, list[int]] = {
    MAJ: [0, 4, 7, 11], MIN: [0, 3, 7, 10], DOM: [0, 4, 7, 10],
    HDIM: [0, 3, 6, 10], DIM: [0, 3, 6, 9], SUS: [0, 5, 7, 10],
}


def _midi_to_hz(midi: float) -> float:
    return 440.0 * 2.0 ** ((midi - 69) / 12.0)


def _pluck(freq: float, duration: float, sr: int, harmonics: list[float],
           decay: float, rng: np.random.Generator) -> np.ndarray:
    """One note: summed harmonics under an exponential decay."""
    n = max(1, int(duration * sr))
    t = np.arange(n, dtype=np.float32) / sr
    out = np.zeros(n, dtype=np.float32)
    for k, amp in enumerate(harmonics, start=1):
        if amp <= 0:
            continue
        phase = rng.uniform(0, 2 * np.pi)
        out += amp * np.sin(2 * np.pi * freq * k * t + phase).astype(np.float32)
    env = np.exp(-decay * t).astype(np.float32)
    attack = min(n, int(0.005 * sr))
    if attack > 1:
        env[:attack] *= np.linspace(0.0, 1.0, attack, dtype=np.float32)
    return out * env


def _add(buffer: np.ndarray, chunk: np.ndarray, at: int, gain: float) -> None:
    start = max(0, at)
    stop = min(len(buffer), at + len(chunk))
    if stop > start:
        buffer[start:stop] += gain * chunk[start - at: stop - at]


def _walking_line(chords: list[Chord], rng: np.random.Generator) -> list[int]:
    """A MIDI note per beat: roots on chord changes, approaches into the next.

    Real walking spends most beats away from the root, which is exactly what
    makes bass chroma an interesting signal rather than a giveaway.
    """
    line: list[int] = []
    for i, chord in enumerate(chords):
        nxt = chords[(i + 1) % len(chords)]
        changing = nxt != chord
        prev_chord = chords[i - 1] if i else chords[-1]
        on_change = chord != prev_chord

        if on_change:
            pitch = chord.bass_pc
        elif changing:
            # Chromatic or scalar approach into the next root.
            pitch = (nxt.bass_pc + rng.choice([-1, 1, -2, 5])) % 12
        else:
            pitch = (chord.root + int(rng.choice(_CHORD_TONES[chord.quality]))) % 12

        # Keep the line inside a bass register, moving by the smallest step.
        target = line[-1] if line else 40
        octave = round((target - pitch) / 12.0)
        midi = int(pitch + 12 * octave)
        midi = int(np.clip(midi, 33, 55))       # A1 .. G3
        line.append(midi)
    return line


def _voicing(chord: Chord, prev_top: int, rng: np.random.Generator) -> list[int]:
    """Rootless voicing placed near the previous one, for plausible voice leading."""
    offsets = _VOICINGS[chord.quality]
    pitches = [(chord.root + off) % 12 for off in offsets]
    # Place the lowest voice in the octave that lands nearest the previous top.
    base = 55 + int(rng.integers(-2, 3))
    notes = []
    for pc in pitches:
        octave = round((base - pc) / 12.0)
        notes.append(int(pc + 12 * octave))
        base = notes[-1] + 3
    shift = 12 * round((prev_top - max(notes)) / 12.0)
    return [int(np.clip(n + shift, 48, 84)) for n in notes]


def render(song: Song, style: PerformanceStyle | None = None,
           sample_rate: int = SAMPLE_RATE) -> tuple[np.ndarray, dict]:
    """Render ``n_choruses`` of ``song``.

    Returns the audio and a truth dict describing exactly what was played.
    """
    style = style or PerformanceStyle()
    rng = np.random.default_rng(style.seed)

    chorus = [c.transpose(style.transpose) for c in song.chords]
    beats_per_bar = song.beats_per_bar
    offset = (style.start_offset_bars * beats_per_bar) % len(chorus)
    rotated = chorus[offset:] + chorus[:offset]

    beats = rotated * style.n_choruses
    beat_seconds = 60.0 / style.tempo

    # Beat onset times, with optional rubato.
    times = np.arange(len(beats) + 1, dtype=np.float64) * beat_seconds
    if style.rubato > 0:
        jitter = rng.normal(0.0, style.rubato * beat_seconds, size=len(times))
        jitter[0] = 0.0
        times = times + np.cumsum(jitter) * 0.3

    total = int((times[-1] + 2.0) * sample_rate)
    buffer = np.zeros(total, dtype=np.float32)

    # --- bass -------------------------------------------------------------
    line = _walking_line(beats, rng)
    for i, midi in enumerate(line):
        at = int(times[i] * sample_rate)
        note = _pluck(_midi_to_hz(midi), beat_seconds * 0.95, sample_rate,
                      harmonics=[1.0, 0.5, 0.22, 0.1, 0.05], decay=3.2, rng=rng)
        _add(buffer, note, at, style.bass_level * 0.30)

    # --- comping ----------------------------------------------------------
    # Sparse and syncopated, the way a pianist actually comps: not every beat,
    # and often pushed onto the swung upbeat.
    prev_top = 67
    i = 0
    while i < len(beats):
        chord = beats[i]
        notes = _voicing(chord, prev_top, rng)
        prev_top = max(notes)
        push = style.swing if rng.random() < 0.45 else 0.0
        at = int((times[i] + push * beat_seconds) * sample_rate)
        duration = beat_seconds * float(rng.uniform(0.8, 2.0))
        for midi in notes:
            note = _pluck(_midi_to_hz(midi), duration, sample_rate,
                          harmonics=[1.0, 0.45, 0.25, 0.12], decay=2.4, rng=rng)
            _add(buffer, note, at, style.comp_level * 0.075)
        i += int(rng.choice([1, 2, 2, 3]))

    # --- drums ------------------------------------------------------------
    if style.drums_level > 0:
        for i in range(len(beats)):
            at = int(times[i] * sample_rate)
            _add(buffer, _cymbal(sample_rate, beat_seconds * 0.9, rng), at,
                 style.drums_level * 0.10)
            # Swung eighth on beats 2 and 4 - the ride pattern.
            if i % beats_per_bar in (1, 3):
                up = int((times[i] + style.swing * beat_seconds) * sample_rate)
                _add(buffer, _cymbal(sample_rate, beat_seconds * 0.5, rng), up,
                     style.drums_level * 0.07)

    # --- head -------------------------------------------------------------
    # Stated on the first and last chorus only; the choruses between are solos,
    # so a melody matcher pointed anywhere else would find improvisation.
    head_choruses: list[int] = []
    if style.head is not None and style.n_choruses >= 1:
        head_choruses = [0] if style.n_choruses == 1 else [0, style.n_choruses - 1]
        head_positions, head_pitches = style.head
        chorus_beats = len(rotated)
        for chorus in head_choruses:
            for position, midi in zip(head_positions, head_pitches):
                # The head follows the form, so shift it by the same rotation.
                beat = (float(position) - offset) % chorus_beats + chorus * chorus_beats
                index = int(np.floor(beat))
                if index >= len(beats):
                    continue
                frac = beat - index
                at = int((times[index] + frac * beat_seconds) * sample_rate)
                duration = beat_seconds * 0.55
                note = _pluck(_midi_to_hz(float(midi)), duration, sample_rate,
                              harmonics=[1.0, 0.6, 0.35, 0.2, 0.1], decay=3.0, rng=rng)
                _add(buffer, note, at, style.head_level * 0.22)

    if style.noise_level > 0:
        buffer += rng.normal(0, style.noise_level, size=total).astype(np.float32)

    peak = float(np.abs(buffer).max())
    if peak > 0:
        buffer = buffer / peak * 0.85

    truth = {
        "title": song.title,
        "transpose": style.transpose,
        "tempo": style.tempo,
        "n_bars": song.n_bars,
        "beats_per_bar": beats_per_bar,
        "start_offset_bars": style.start_offset_bars,
        "n_choruses": style.n_choruses,
        "beat_times": times[:len(beats)],
        "head_choruses": head_choruses,
        "chords": beats,
        "sample_rate": sample_rate,
    }
    return buffer, truth


def _cymbal(sample_rate: int, duration: float, rng: np.random.Generator) -> np.ndarray:
    """A ride hit: high-passed noise with a fast decay."""
    n = max(1, int(duration * sample_rate))
    noise = rng.normal(0, 1, n).astype(np.float32)
    # Cheap high-pass via first difference, twice.
    noise = np.diff(noise, prepend=0.0).astype(np.float32)
    noise = np.diff(noise, prepend=0.0).astype(np.float32)
    t = np.arange(n, dtype=np.float32) / sample_rate
    return noise * np.exp(-14.0 * t).astype(np.float32)


def write_wav(path: str, audio: np.ndarray, sample_rate: int = SAMPLE_RATE) -> None:
    import wave

    clipped = np.clip(audio, -1.0, 1.0)
    pcm = (clipped * 32767).astype("<i2")
    with wave.open(path, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(pcm.tobytes())


def random_style(rng: np.random.Generator, **overrides) -> PerformanceStyle:
    """A plausible performance: club tempos, any key, starting anywhere in the form."""
    style = PerformanceStyle(
        tempo=float(rng.uniform(90, 260)),
        transpose=int(rng.integers(0, 12)),
        swing=float(rng.uniform(0.55, 0.68)),
        seed=int(rng.integers(0, 2**31)),
        noise_level=float(rng.uniform(0.003, 0.02)),
        drums_level=float(rng.uniform(0.3, 0.8)),
        comp_level=float(rng.uniform(0.6, 1.0)),
    )
    for key, value in overrides.items():
        setattr(style, key, value)
    return style

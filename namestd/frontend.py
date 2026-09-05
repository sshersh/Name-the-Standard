"""Audio -> beat-synchronous harmony.

Two decisions shape this module.

*Index by beats, not seconds.*  Pooling chroma onto the beat grid makes
everything downstream tempo-invariant for free: the same tune at 140 and at 260
produces the same sequence.

*No downbeat tracking.*  The obvious next step after beat tracking is finding
bar lines, and it is the most fragile part of any such pipeline.  We skip it
entirely: the matcher searches every rotation of the form at beat resolution, so
the form's phase - and with it the downbeat - falls out of the alignment.  That
removes a whole class of failure for free.

Two chroma bands are kept separate throughout.  Jazz pianists play rootless
voicings, so the root of the chord is usually only in the bass register, while
thirds and sevenths live in the comping.  Folding them into one vector, as
ordinary chroma does, throws that split away.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

SAMPLE_RATE = 22050
BINS_PER_OCTAVE = 36
N_OCTAVES = 6
FMIN_HZ = 32.703195662574829      # C1

# Band split at C3.  Below it, in a small group, is almost entirely the bass;
# above it is comping and horns.
_BASS_BINS = slice(0, 2 * BINS_PER_OCTAVE)                     # C1 - B2
_TREBLE_BINS = slice(2 * BINS_PER_OCTAVE, 5 * BINS_PER_OCTAVE)  # C3 - B5

# The pulse we want to track is the jazz quarter note.  Beat trackers routinely
# lock onto half or double that; anything outside this range gets folded back.
MIN_TEMPO, MAX_TEMPO = 100.0, 330.0


@dataclass
class Features:
    """Beat-synchronous harmony extracted from one recording."""

    beat_times: np.ndarray     # (n_beats,) seconds
    chroma: np.ndarray         # (n_beats, 24): bass 12 then treble 12, unit-norm each
    tempo: float
    sample_rate: int = SAMPLE_RATE

    @property
    def n_beats(self) -> int:
        return self.chroma.shape[0]

    @property
    def duration(self) -> float:
        return float(self.beat_times[-1] - self.beat_times[0]) if self.n_beats > 1 else 0.0


def load_audio(path: str, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    import librosa

    audio, _ = librosa.load(path, sr=sample_rate, mono=True)
    return audio.astype(np.float32)


def track_beats(audio: np.ndarray, sample_rate: int = SAMPLE_RATE,
                tempo_hint: float | None = None) -> tuple[np.ndarray, float]:
    """Beat times in seconds, plus the tempo they imply.

    The tracker's octave error - reporting half or double the true tempo - is
    corrected by folding the result into the range a jazz quarter note actually
    occupies.  Getting this wrong is not a small error: it halves or doubles the
    number of beats per chorus and the form no longer aligns.
    """
    import librosa

    tempo, beat_frames = librosa.beat.beat_track(
        y=audio, sr=sample_rate, trim=False,
        start_bpm=tempo_hint if tempo_hint else 150.0,
    )
    beats = librosa.frames_to_time(beat_frames, sr=sample_rate)
    tempo = float(np.atleast_1d(tempo)[0])

    if len(beats) < 2:
        return beats, tempo

    # Work from the observed spacing rather than the reported tempo, which can
    # disagree with the beats actually returned.
    observed = 60.0 / float(np.median(np.diff(beats)))
    factor = 1.0
    while observed * factor < MIN_TEMPO:
        factor *= 2.0
    while observed * factor > MAX_TEMPO:
        factor /= 2.0

    if factor > 1.0:
        beats = _subdivide(beats, int(round(factor)))
    elif factor < 1.0:
        beats = beats[::int(round(1.0 / factor))]

    tempo = 60.0 / float(np.median(np.diff(beats))) if len(beats) > 1 else tempo
    return beats, tempo


def _subdivide(beats: np.ndarray, factor: int) -> np.ndarray:
    """Insert ``factor - 1`` evenly spaced beats between each pair."""
    if factor < 2 or len(beats) < 2:
        return beats
    out = []
    for start, stop in zip(beats[:-1], beats[1:]):
        out.extend(start + (stop - start) * np.arange(factor) / factor)
    out.append(beats[-1])
    return np.asarray(out)


def _cqt_bands(audio: np.ndarray, sample_rate: int) -> tuple[np.ndarray, np.ndarray]:
    """Log-compressed CQT magnitude, split into bass and treble registers."""
    import librosa

    cqt = np.abs(librosa.cqt(
        y=audio, sr=sample_rate, fmin=FMIN_HZ,
        n_bins=BINS_PER_OCTAVE * N_OCTAVES,
        bins_per_octave=BINS_PER_OCTAVE,
        hop_length=512,
    ))
    # Log compression tames the dynamics of a room: a loud piano stab should not
    # outweigh eight bars of steady walking.
    cqt = np.log1p(20.0 * cqt).astype(np.float32)
    return cqt[_BASS_BINS], cqt[_TREBLE_BINS]


def _fold_to_chroma(band: np.ndarray, tuning_offset: int = 0) -> np.ndarray:
    """Fold CQT bins onto 12 pitch classes.  Returns (12, n_frames).

    With three bins per semitone we take only the on-pitch bin rather than
    summing all three.  Summing pulls in energy a third of a semitone sharp and
    flat, which smears neighbouring pitch classes together and measurably flattens
    the result; the two off-pitch bins are mostly smearing, not signal.
    """
    step = BINS_PER_OCTAVE // 12
    chroma = np.zeros((12, band.shape[1]), dtype=np.float32)
    for bin_index in range(tuning_offset % step, band.shape[0], step):
        chroma[(bin_index // step) % 12] += band[bin_index]
    return chroma


def smooth_chroma(chroma: np.ndarray, window: int) -> np.ndarray:
    """Moving-average over beats, then renormalise.

    Harmony moves at roughly the half-bar, but a walking bassline moves every
    beat.  Averaging over a few beats lets the chord tones accumulate while the
    passing tones, which keep changing, average away.  The window stays short
    because a ii-V can occupy only two beats each.
    """
    if window <= 1:
        return chroma
    kernel = np.ones(window, dtype=np.float32) / window
    smoothed = np.stack(
        [np.convolve(chroma[:, j], kernel, mode="same") for j in range(chroma.shape[1])],
        axis=1,
    )
    # Renormalise the bass and treble halves separately, so that a loud passage
    # in one register cannot swamp the other.
    out = np.empty_like(smoothed)
    for half in (slice(0, 12), slice(12, 24)):
        block = smoothed[:, half]
        out[:, half] = block / np.maximum(
            np.linalg.norm(block, axis=1, keepdims=True), 1e-8)
    return out.astype(np.float32)


def _beat_sync(chroma: np.ndarray, frame_times: np.ndarray,
               beat_times: np.ndarray) -> np.ndarray:
    """Median-pool frames within each beat.  Returns (n_beats, 12).

    Median rather than mean: a cymbal crash or a passing horn note inside a beat
    should not drag the beat's harmony with it.
    """
    edges = np.searchsorted(frame_times, beat_times)
    out = np.zeros((len(beat_times), 12), dtype=np.float32)
    for i in range(len(beat_times)):
        start = edges[i]
        stop = edges[i + 1] if i + 1 < len(edges) else chroma.shape[1]
        if stop <= start:
            stop = min(start + 1, chroma.shape[1])
        if start >= chroma.shape[1]:
            out[i] = out[i - 1] if i else 0.0
            continue
        out[i] = np.median(chroma[:, start:stop], axis=1)
    return out


def _normalise(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.maximum(norms, 1e-8)


def analyse(audio: np.ndarray, sample_rate: int = SAMPLE_RATE,
            beat_times: np.ndarray | None = None,
            tempo_hint: float | None = None) -> Features:
    """Full front end: audio in, beat-synchronous 24-d chroma out.

    ``beat_times`` may be supplied to bypass beat tracking, which is how the
    evaluation separates matcher errors from beat-tracking errors.
    """
    import librosa

    if beat_times is None:
        beat_times, tempo = track_beats(audio, sample_rate, tempo_hint)
    else:
        beat_times = np.asarray(beat_times, dtype=np.float64)
        tempo = (60.0 / float(np.median(np.diff(beat_times)))
                 if len(beat_times) > 1 else 0.0)

    if len(beat_times) < 2:
        return Features(beat_times=np.asarray(beat_times),
                        chroma=np.zeros((0, 24), dtype=np.float32),
                        tempo=tempo, sample_rate=sample_rate)

    bass_band, treble_band = _cqt_bands(audio, sample_rate)
    frame_times = librosa.frames_to_time(
        np.arange(bass_band.shape[1]), sr=sample_rate, hop_length=512)

    bass = _normalise(_beat_sync(_fold_to_chroma(bass_band), frame_times, beat_times))
    treble = _normalise(_beat_sync(_fold_to_chroma(treble_band), frame_times, beat_times))

    return Features(
        beat_times=beat_times,
        chroma=np.concatenate([bass, treble], axis=1).astype(np.float32),
        tempo=tempo,
        sample_rate=sample_rate,
    )


def analyse_hypotheses(audio: np.ndarray, sample_rate: int = SAMPLE_RATE,
                       factors: tuple[float, ...] = (1.0, 2.0, 0.5)) -> list[Features]:
    """Front-end output for several tempo readings of the same audio.

    Beat trackers make octave errors - reporting 120 for a tune played at 240 -
    and no range check can catch them, because both readings are perfectly
    plausible tempos.  Rather than guess here, we hand the matcher one Features
    per hypothesis and let it decide: it can ask whether a form actually aligns,
    which is far stronger evidence than anything available at this stage.
    """
    base, _ = track_beats(audio, sample_rate)
    if len(base) < 2:
        return [analyse(audio, sample_rate, beat_times=base)]

    out = []
    for factor in factors:
        if factor >= 1.0:
            grid = _subdivide(base, int(round(factor)))
        else:
            grid = base[::int(round(1.0 / factor))]
        if len(grid) >= 8:
            out.append(analyse(audio, sample_rate, beat_times=grid))
    return out


def analyse_file(path: str, sample_rate: int = SAMPLE_RATE) -> Features:
    return analyse(load_audio(path, sample_rate), sample_rate)

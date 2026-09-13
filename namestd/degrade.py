"""Degradations that stand in for the things a real performance does.

Everything measured so far has been synthesised under ideal conditions: metronomic
time, the chart played literally, no room, no audience.  Real recordings differ in
specific, nameable ways, and the design document lists those as *claims* about where
the system will break.

This module makes each one a separate, controllable condition so the claims can be
measured instead of asserted.  They split into two kinds: what the room does to the
sound (reverb, noise, a phone's band-limited capture) and what the band does to the
music (reharmonisation, playing outside, dropping the piano, drifting in tempo).

None of this replaces evaluation on real recordings.  It isolates which single
factor hurts, which a real recording - where every factor varies at once - cannot.
"""

from __future__ import annotations

import numpy as np

from .chords import DOM, MAJ, MIN, Chord


# --- what the room does -----------------------------------------------------

def reverb(audio: np.ndarray, sample_rate: int, seconds: float = 1.1,
           wet: float = 0.35, seed: int = 0) -> np.ndarray:
    """Convolve with an exponentially decaying noise burst - a crude small room.

    Reverb smears harmony across time, which is the part that matters here: a
    chord bleeds into the next bar and blurs the chroma the matcher relies on.
    """
    if seconds <= 0 or wet <= 0:
        return audio
    rng = np.random.default_rng(seed)
    n = int(seconds * sample_rate)
    t = np.arange(n, dtype=np.float32) / sample_rate
    impulse = (rng.normal(0, 1, n).astype(np.float32) * np.exp(-3.6 * t))
    impulse[0] = 1.0
    impulse /= np.linalg.norm(impulse)
    wet_signal = np.convolve(audio, impulse)[:len(audio)].astype(np.float32)
    peak = float(np.abs(wet_signal).max()) or 1.0
    wet_signal *= float(np.abs(audio).max()) / peak
    return ((1.0 - wet) * audio + wet * wet_signal).astype(np.float32)


def band_limit(audio: np.ndarray, sample_rate: int, low_hz: float = 150.0,
               high_hz: float = 5000.0) -> np.ndarray:
    """A phone microphone: little below ~150 Hz, nothing much up top.

    This is the pointed one.  The bass band carries root motion, and the bass
    register is exactly what a phone mic throws away.
    """
    from scipy import signal

    nyquist = sample_rate / 2.0
    out = audio
    if low_hz > 0:
        b, a = signal.butter(4, min(low_hz / nyquist, 0.99), btype="high")
        out = signal.lfilter(b, a, out)
    if high_hz < nyquist:
        b, a = signal.butter(4, min(high_hz / nyquist, 0.99), btype="low")
        out = signal.lfilter(b, a, out)
    return out.astype(np.float32)


def crowd(audio: np.ndarray, sample_rate: int, level: float = 0.05,
          seed: int = 0) -> np.ndarray:
    """Room tone plus the low rumble of a room with people in it."""
    if level <= 0:
        return audio
    rng = np.random.default_rng(seed)
    noise = rng.normal(0, 1, len(audio)).astype(np.float32)
    # Low-passed noise reads as a room far better than white noise does.
    kernel = np.ones(64, dtype=np.float32) / 64
    rumble = np.convolve(noise, kernel, mode="same").astype(np.float32)
    mix = 0.6 * rumble + 0.4 * noise
    mix *= level * float(np.abs(audio).max()) / (float(np.abs(mix).max()) or 1.0)
    return (audio + mix).astype(np.float32)


# --- what the band does -----------------------------------------------------

def reharmonise(chords: list[Chord], amount: float = 0.3,
                seed: int = 0) -> list[Chord]:
    """Apply the substitutions a rhythm section makes without discussion.

    Tritone subs on dominants, a ii stuck in front of a V, a major chord read as
    its relative minor.  These are the cases the design document says a
    theory-aware substitution cost would absorb - which is not yet implemented,
    so this measures what plain correlation does with them.
    """
    rng = np.random.default_rng(seed)
    out = list(chords)
    n = len(out)
    i = 0
    while i < n:
        chord = out[i]
        # Find how long this chord is held.
        span = 1
        while i + span < n and out[i + span] == chord:
            span += 1

        if rng.random() < amount:
            if chord.quality == DOM:
                choice = rng.random()
                if choice < 0.5:
                    # Tritone substitution.
                    sub = Chord((chord.root + 6) % 12, DOM)
                    for k in range(i, i + span):
                        out[k] = sub
                elif span >= 2:
                    # Put the related ii in front of the V.
                    related = Chord((chord.root + 7) % 12, MIN)
                    for k in range(i, i + span // 2):
                        out[k] = related
            elif chord.quality == MAJ and rng.random() < 0.5:
                # Relative minor for the tonic.
                for k in range(i, i + span):
                    out[k] = Chord((chord.root + 9) % 12, MIN)
        i += span
    return out


def play_outside(chords: list[Chord], fraction: float = 0.25,
                 seed: int = 0) -> list[Chord]:
    """Replace whole bars with unrelated harmony, as a soloist going out does."""
    rng = np.random.default_rng(seed)
    out = list(chords)
    bar = 4
    for start in range(0, len(out) - bar + 1, bar):
        if rng.random() < fraction:
            sub = Chord(int(rng.integers(0, 12)),
                        str(rng.choice([MAJ, MIN, DOM])))
            for k in range(start, min(start + bar, len(out))):
                out[k] = sub
    return out

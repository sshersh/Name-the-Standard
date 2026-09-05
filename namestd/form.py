"""Find the chorus period - how many beats before the form comes round again.

This runs before the corpus is touched, and needs no database at all: a jazz
performance repeats its changes every chorus, so the beat-synchronous chroma is
periodic with the form.  Comparing each beat against the beat one candidate
period later and averaging gives a clean peak at the true period.

It earns its place twice over.  It prunes the search to tunes whose form could
actually be what we are hearing, which removes most cross-length false matches;
and a 12-bar query stops being scored against 32-bar tunes on the strength of a
lucky alignment.
"""

from __future__ import annotations

import numpy as np

# Jazz forms are built from 4-bar phrases, so in 4/4 the plausible periods are
# multiples of 16 beats - plus 12-bar blues at 48 and its relatives.
MIN_PERIOD = 24
MAX_PERIOD = 384


def periodicity_curve(chroma: np.ndarray, min_period: int = MIN_PERIOD,
                      max_period: int = MAX_PERIOD) -> tuple[np.ndarray, np.ndarray]:
    """Mean self-similarity at each candidate period.

    Returns the periods considered and, for each, the average cosine similarity
    between beats one period apart.
    """
    n_beats = chroma.shape[0]
    limit = min(max_period, n_beats // 2)
    if limit < min_period:
        return np.zeros(0, dtype=int), np.zeros(0, dtype=np.float32)

    unit = chroma / np.maximum(np.linalg.norm(chroma, axis=1, keepdims=True), 1e-8)
    periods = np.arange(min_period, limit + 1)
    strength = np.empty(len(periods), dtype=np.float32)
    for i, period in enumerate(periods):
        overlap = (unit[:-period] * unit[period:]).sum(axis=1)
        strength[i] = float(overlap.mean())
    return periods, strength


def detect_periods(chroma: np.ndarray, top_k: int = 3,
                   min_period: int = MIN_PERIOD,
                   max_period: int = MAX_PERIOD) -> list[tuple[int, float]]:
    """Rank candidate chorus periods, most likely first.

    Two corrections matter.  Self-similarity drifts upward with period simply
    because fewer pairs are averaged, so we subtract a smooth baseline.  And a
    32-bar AABA is genuinely periodic at its 8-bar A section too, so several
    hypotheses are returned rather than one - the matcher accepts any form length
    that is a whole multiple of a detected period, which covers both readings.
    """
    periods, strength = periodicity_curve(chroma, min_period, max_period)
    if len(periods) == 0:
        return []

    # Subtract a slow-moving baseline so only genuine peaks survive.  The
    # baseline is padded by edge replication rather than with zeros: zero
    # padding drags it down at both ends of the range and manufactures a peak
    # at the shortest and longest period considered, which is exactly where a
    # spurious answer does the most damage.
    window = max(9, (len(strength) // 8) | 1)
    kernel = np.ones(window, dtype=np.float32) / window
    padded = np.pad(strength, window // 2, mode="edge")
    baseline = np.convolve(padded, kernel, mode="valid")[:len(strength)]
    excess = strength - baseline

    spread = float(excess.std()) or 1.0
    ranked = []
    for index in np.argsort(-excess):
        period = int(periods[index])
        # Keep peaks well separated; neighbouring lags are the same peak.
        if any(abs(period - kept) < 6 for kept, _ in ranked):
            continue
        ranked.append((period, float(excess[index] / spread)))
        if len(ranked) >= top_k:
            break
    return ranked


def allowed_lengths(candidate_periods: list[tuple[int, float]],
                    available: list[int], tolerance: int = 2) -> list[int]:
    """Which corpus form lengths are consistent with the detected periods.

    A length qualifies when it is a whole multiple of some detected period, which
    admits both readings of an AABA form: the full 32 bars, and the 8-bar A
    section that also repeats.
    """
    if not candidate_periods:
        return list(available)

    keep = set()
    for period, _ in candidate_periods:
        for length in available:
            if length < period - tolerance:
                continue
            multiple = round(length / period)
            if multiple >= 1 and abs(length - multiple * period) <= tolerance:
                keep.add(length)
    return sorted(keep) if keep else list(available)

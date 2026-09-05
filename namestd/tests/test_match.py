"""Tests for form detection and harmonic matching.

Most of these drive the matcher from *synthesised* audio, so they exercise the
real path - CQT, chroma, templates, correlation - rather than a mock.
"""

from __future__ import annotations

import numpy as np
import pytest

from namestd.chords import template_matrix
from namestd.corpus import load_corpus
from namestd.form import allowed_lengths, detect_periods
from namestd.frontend import Features, analyse, smooth_chroma
from namestd.match import build_matcher
from namestd.synth import PerformanceStyle, render


@pytest.fixture(scope="module")
def songs():
    try:
        return load_corpus()
    except FileNotFoundError:
        pytest.skip("corpus not fetched; run scripts/fetch_data.py")


@pytest.fixture(scope="module")
def matcher(songs):
    return build_matcher(songs)


@pytest.fixture(scope="module")
def by_title(songs):
    return {s.title: s for s in songs}


def _oracle_features(song, transpose=0, rotation=0, choruses=3):
    """A query built from the chart itself: perfect chord knowledge, no audio."""
    chords = [c.transpose(transpose) for c in song.chords]
    chords = chords[rotation:] + chords[:rotation]
    chroma = template_matrix(chords * choruses)
    return Features(beat_times=np.arange(len(chroma)) * 0.4, chroma=chroma, tempo=150.0)


def _audio_features(song, **style):
    audio, truth = render(song, PerformanceStyle(**style))
    # Oracle beat times: these tests are about the matcher, not the beat tracker.
    return analyse(audio, beat_times=truth["beat_times"])


# --- form detection ---------------------------------------------------------

@pytest.mark.parametrize("title", [
    "Blue Monk", "Autumn Leaves", "Oleo", "Alone Together", "Cherokee",
    "All The Things You Are", "Take The A Train",
])
def test_detects_chorus_period_from_audio(by_title, title):
    song = by_title[title]
    features = _audio_features(song, tempo=170.0, transpose=3, n_choruses=4, seed=7)
    periods = detect_periods(smooth_chroma(features.chroma, 3))
    assert periods, title
    assert abs(periods[0][0] - song.n_beats) <= 2, (title, periods, song.n_beats)


def test_period_baseline_does_not_manufacture_edge_peaks():
    # Zero-padding the baseline used to invent peaks at the shortest and longest
    # period considered.  Flat, structureless input must produce no strong peak
    # at either end of the range.
    rng = np.random.default_rng(0)
    chroma = rng.normal(size=(600, 24)).astype(np.float32)
    periods = detect_periods(chroma)
    extremes = {24, 25, 26, 298, 299, 300}
    assert not (periods and periods[0][0] in extremes and periods[0][1] > 4.0)


def test_allowed_lengths_accepts_multiples():
    available = [48, 96, 128, 144, 176]
    # An AABA form is also periodic at its A section, so both readings must pass.
    assert 128 in allowed_lengths([(128, 5.0)], available)
    assert 128 in allowed_lengths([(32, 5.0)], available)
    assert 128 not in allowed_lengths([(48, 5.0)], available)


def test_allowed_lengths_falls_back_when_nothing_fits(by_title):
    assert allowed_lengths([(97, 5.0)], [48, 128]) == [48, 128]


# --- matching ---------------------------------------------------------------

@pytest.mark.parametrize("transpose", [0, 1, 5, 7, 11])
def test_transposition_invariance(matcher, by_title, transpose):
    """The same tune in any key must give the same answer, and report the key."""
    song = by_title["Autumn Leaves"]
    report = matcher.match(_oracle_features(song, transpose=transpose), smooth_window=1)
    assert report.best.title == "Autumn Leaves"
    assert report.best.transposition == transpose


@pytest.mark.parametrize("rotation", [0, 16, 64, 100])
def test_rotation_invariance(matcher, by_title, rotation):
    """Walking in mid-tune must not change the answer - and the offset is recovered."""
    song = by_title["All The Things You Are"]
    report = matcher.match(_oracle_features(song, rotation=rotation), smooth_window=1)
    assert report.best.title == "All The Things You Are"
    assert report.best.rotation == rotation


@pytest.mark.parametrize("title,transpose,offset_bars", [
    ("Autumn Leaves", 5, 0),
    ("Oleo", 3, 8),
    ("Alone Together", 7, 0),
    ("All The Things You Are", 4, 16),
    ("Cherokee", 9, 0),
    ("Take The A Train", 1, 0),
])
def test_identifies_tune_from_audio(matcher, songs, by_title, title, transpose, offset_bars):
    song = by_title[title]
    features = _audio_features(song, tempo=175.0, transpose=transpose,
                               start_offset_bars=offset_bars, n_choruses=3, seed=11)
    report = matcher.match(features)

    # The tune itself, or one written on the same changes.  Oleo and Anthropology
    # are identical bar for bar, so demanding the exact title would be demanding
    # something harmony cannot deliver - that is the melody's job.
    true_index = next(i for i, s in enumerate(songs) if s.title == title)
    assert report.best.family_index == matcher.family_of[true_index], report.best.title
    assert report.best.transposition == transpose
    # Within a beat: smoothing spreads harmony slightly across the beat grid, so
    # the correlation peak can sit one beat either side of the true downbeat.
    expected = offset_bars * song.beats_per_bar
    error = (report.best.rotation - expected) % song.n_beats
    assert min(error, song.n_beats - error) <= 1


def test_contrafacts_are_reported_as_a_family(matcher, by_title):
    """Blue Monk cannot be separated from other blues by harmony, and shouldn't be.

    The honest answer is the family; the tune needs the melody of the head.
    """
    song = by_title["Blue Monk"]
    features = _audio_features(song, tempo=132.0, n_choruses=5, seed=3)
    report = matcher.match(features, top_k=6)
    assert report.families[0][0] == "12-bar blues"
    assert all(m.family_name == "12-bar blues" for m in report.matches[:4])


def test_rhythm_changes_family_wins_for_oleo(matcher, by_title):
    song = by_title["Oleo"]
    features = _audio_features(song, tempo=210.0, transpose=2, n_choruses=3, seed=5)
    report = matcher.match(features, top_k=5)
    assert report.families[0][0] == "rhythm changes"


def test_short_query_is_refused_rather_than_guessed(matcher, by_title):
    song = by_title["Autumn Leaves"]
    features = _oracle_features(song, choruses=1)
    truncated = Features(beat_times=features.beat_times[:8],
                         chroma=features.chroma[:8], tempo=150.0)
    assert matcher.match(truncated).matches == []


def test_score_rises_with_more_listening(matcher, by_title):
    """Confidence should grow with listening time - the opposite of a fingerprinter."""
    song = by_title["Cherokee"]
    features = _audio_features(song, tempo=240.0, transpose=4, n_choruses=6, seed=9)
    scores = []
    for beats in (song.n_beats, song.n_beats * 3, song.n_beats * 6):
        window = Features(beat_times=features.beat_times[:beats],
                          chroma=features.chroma[:beats], tempo=features.tempo)
        report = matcher.match(window)
        scores.append(report.best.score if report.best else 0.0)
    assert scores[-1] > scores[0]

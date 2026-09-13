"""Tests for the melody stage - the tie-break harmony cannot perform."""

from __future__ import annotations

import numpy as np
import pytest

from namestd.corpus import load_corpus
from namestd.families import build_families
from namestd.heads import build_index, generate_head
from namestd.melody import (
    MelodyIndex, align_score, extract_melody, head_window, rerank, to_intervals,
)
from namestd.synth import PerformanceStyle, render


@pytest.fixture(scope="module")
def songs():
    try:
        return load_corpus()
    except FileNotFoundError:
        pytest.skip("corpus not fetched; run scripts/fetch_data.py")


# --- interval representation ------------------------------------------------

def test_intervals_are_transposition_invariant():
    line = [60, 62, 65, 64, 60]
    assert to_intervals(line) == to_intervals([p + 7 for p in line])
    assert to_intervals(line) == [2, 3, -1, -4]


def test_intervals_of_a_single_note_are_empty():
    assert to_intervals([60]) == []
    assert to_intervals([]) == []


# --- alignment --------------------------------------------------------------

def test_identical_lines_score_one():
    line = to_intervals([60, 62, 64, 65, 67, 65, 64])
    assert align_score(line, line) == pytest.approx(1.0, abs=1e-5)


def test_unrelated_lines_score_low():
    a = to_intervals([60, 62, 64, 65, 67, 69, 71, 72])
    b = to_intervals([60, 55, 61, 48, 70, 52, 66, 53])
    assert align_score(a, b) < 0.5


def test_alignment_is_local_so_a_fragment_still_matches():
    """Catching half the head, with the rest paraphrased, must still count."""
    reference = to_intervals([60, 62, 64, 65, 67, 69, 67, 65, 64, 62, 60, 59])
    fragment = to_intervals([67, 69, 67, 65, 64])
    assert align_score(fragment, reference) > 0.8


def test_alignment_tolerates_an_inserted_ornament():
    plain = [60, 64, 67, 72, 71, 67]
    ornamented = [60, 61, 64, 67, 72, 71, 67]      # an approach note added
    assert align_score(to_intervals(ornamented), to_intervals(plain)) > 0.55


def test_empty_query_scores_zero():
    assert align_score([], [1, 2, 3]) == 0.0


# --- head generation --------------------------------------------------------

def test_generated_heads_are_deterministic_and_distinct(songs):
    by = {s.title: s for s in songs}
    a1 = generate_head(by["Blue Monk"])[1]
    a2 = generate_head(by["Blue Monk"])[1]
    b = generate_head(by["Bags' Groove"])[1]
    assert np.array_equal(a1, a2)
    # Two tunes on identical changes must get different heads - that is the
    # whole case this stage exists to resolve.
    assert not np.array_equal(a1[:20], b[:20])


def test_generated_head_stays_in_range(songs):
    by = {s.title: s for s in songs}
    _, pitches = generate_head(by["Autumn Leaves"])
    assert pitches.min() >= 60 and pitches.max() <= 84


# --- rerank -----------------------------------------------------------------

def test_rerank_picks_the_right_tune_from_its_own_head(songs):
    """Noise-free: the index and the query come from the same generator."""
    families, _ = build_families(songs)
    blues = next(f for f in families if f.name == "12-bar blues")
    titles = [songs[i].title for i in blues.members]
    index = build_index(songs, titles)

    for index_of_song in blues.members[:8]:
        song = songs[index_of_song]
        _, pitches = generate_head(song)
        scored = rerank(titles, list(pitches), index)
        assert scored[0].title == song.title


def test_rerank_survives_pitch_extraction_from_audio(songs):
    """The real path: render the head, hear it back, and still pick the tune."""
    families, _ = build_families(songs)
    blues = next(f for f in families if f.name == "12-bar blues")
    titles = [songs[i].title for i in blues.members]
    index = build_index(songs, titles)

    hits = 0
    tried = blues.members[:5]
    for i in tried:
        song = songs[i]
        positions, pitches = generate_head(song)
        audio, truth = render(song, PerformanceStyle(
            tempo=140.0, n_choruses=2, seed=4, head=(positions, pitches)))
        beats = truth["beat_times"]
        stop = float(beats[min(song.n_beats, len(beats) - 1)])
        segment = audio[: int(stop * 22050)]
        heard, _ = extract_melody(segment, 22050, beats)
        scored = rerank(titles, heard, index)
        hits += scored[0].title == song.title
    # Extraction is imperfect on a full band; most should still land.
    assert hits >= len(tried) - 1


def test_unknown_tunes_score_zero_rather_than_being_dropped():
    index = MelodyIndex()
    index.add("Known", [60, 62, 64, 65])
    scored = rerank(["Known", "Unindexed"], [60, 62, 64, 65], index)
    assert {m.title for m in scored} == {"Known", "Unindexed"}
    assert dict((m.title, m.score) for m in scored)["Unindexed"] == 0.0


def test_index_round_trips_through_json(tmp_path):
    index = MelodyIndex()
    index.add("Tune", [60, 64, 67, 72])
    path = tmp_path / "melodies.json"
    index.save(path)
    reloaded = MelodyIndex.load(path)
    assert reloaded.entries["Tune"].intervals == [4, 3, 5]


def test_head_window_uses_the_alignment_the_matcher_found():
    beats = np.arange(200, dtype=float) * 0.5
    # Query started 8 beats into a 48-beat form, so the next form top is beat 40.
    start, stop = head_window(beats, period_beats=48, rotation=8, chorus=0)
    assert start == pytest.approx(beats[40])
    assert stop == pytest.approx(beats[88])

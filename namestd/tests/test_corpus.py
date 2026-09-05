"""Tests for chord parsing, form expansion, and family clustering."""

from __future__ import annotations

import pytest

from namestd.chords import (
    DOM, HDIM, MAJ, MIN, SUS, DIM,
    Chord, note_to_pc, parse_chord, treble_template, bass_template,
)
from namestd.corpus import _expand_song, load_corpus
from namestd.families import build_families


# --- chord symbols ----------------------------------------------------------

@pytest.mark.parametrize("name,pc", [("C", 0), ("Bb", 10), ("F#", 6), ("Cb", 11), ("B#", 0)])
def test_note_to_pc(name, pc):
    assert note_to_pc(name) == pc


@pytest.mark.parametrize("symbol,root,quality", [
    ("Bbmaj7", 10, MAJ), ("C6", 0, MAJ), ("Ab69", 8, MAJ),
    ("Dm7", 2, MIN), ("Gm6", 7, MIN), ("Fm11", 5, MIN),
    ("G7", 7, DOM), ("A7b9", 9, DOM), ("Eb13#11", 3, DOM), ("C7alt", 0, DOM),
    ("Em7b5", 4, HDIM),
    ("B07", 11, DIM), ("E0", 4, DIM),
    ("G7sus", 7, SUS), ("C9sus", 0, SUS),
])
def test_parse_quality(symbol, root, quality):
    chord = parse_chord(symbol)
    assert chord == Chord(root, quality)


def test_parse_slash_chord():
    assert parse_chord("Bb7/D") == Chord(10, DOM, bass=2)
    # A slash bass equal to the root is redundant and gets dropped.
    assert parse_chord("Bb7/Bb") == Chord(10, DOM, bass=None)


def test_parse_drops_parenthesised_alternates():
    # iReal writes one player's substitution in parens; it is not the form.
    assert parse_chord("Dmaj7(Em7b5)") == Chord(2, MAJ)
    assert parse_chord("(A7b9)") is None


def test_parse_empty():
    assert parse_chord("") is None
    assert parse_chord("   ") is None


def test_transpose_wraps():
    assert Chord(10, DOM).transpose(3) == Chord(1, DOM)
    assert Chord(2, MIN, bass=9).transpose(5) == Chord(7, MIN, bass=2)


def test_templates_are_unit_norm():
    for chord in (Chord(0, MAJ), Chord(7, DOM), Chord(4, HDIM), Chord(11, DIM)):
        assert treble_template(chord).sum() > 0
        assert abs(float((treble_template(chord) ** 2).sum()) - 1.0) < 1e-5
        assert abs(float((bass_template(chord) ** 2).sum()) - 1.0) < 1e-5


def test_bass_template_peaks_on_the_bass_note():
    # A slash chord's bass note, not its root, is where the bassist sits.
    vec = bass_template(Chord(10, DOM, bass=2))
    assert int(vec.argmax()) == 2


# --- form expansion ---------------------------------------------------------

def test_two_chords_per_bar_split_evenly():
    song = _expand_song({
        "Title": "t", "Composer": "c", "TimeSignature": "4/4",
        "Sections": [{"MainSegment": {"Chords": "Dm7,G7|Cmaj7|Dm7,G7|Cmaj7"}}],
    })
    assert song.n_bars == 4
    assert song.chords[0:2] == [Chord(2, MIN)] * 2
    assert song.chords[2:4] == [Chord(7, DOM)] * 2


def test_three_chords_in_a_bar_give_the_first_the_extra_beat():
    song = _expand_song({
        "Title": "t", "Composer": "c", "TimeSignature": "4/4",
        "Sections": [{"MainSegment": {"Chords": "C,D7,G7|C|C|C"}}],
    })
    assert song.chords[:4] == [Chord(0, MAJ), Chord(0, MAJ), Chord(2, DOM), Chord(7, DOM)]


def test_empty_slot_holds_the_previous_chord():
    # The corpus writes trailing commas to mean "hold", not "rest".
    song = _expand_song({
        "Title": "t", "Composer": "c", "TimeSignature": "4/4",
        "Sections": [{"MainSegment": {"Chords": "G7,Ab7,G7,|C|C|C"}}],
    })
    assert song.chords[:4] == [Chord(7, DOM), Chord(8, DOM), Chord(7, DOM), Chord(7, DOM)]


def test_endings_expand_to_one_pass_each():
    # Two endings means the section is played twice: main+1st, then main+2nd.
    song = _expand_song({
        "Title": "t", "Composer": "c", "TimeSignature": "4/4",
        "Sections": [{
            "MainSegment": {"Chords": "C|C|C|C|C|C"},
            "Endings": [{"Chords": "D7|D7"}, {"Chords": "G7|G7"}],
        }],
    })
    assert song.n_bars == 16
    assert song.chords[6 * 4] == Chord(2, DOM)     # bar 7 = first ending
    assert song.chords[14 * 4] == Chord(7, DOM)    # bar 15 = second ending


def test_repeats_field_plays_the_section_twice():
    song = _expand_song({
        "Title": "t", "Composer": "c", "TimeSignature": "4/4",
        "Sections": [{"MainSegment": {"Chords": "C|F|G|C"}, "Repeats": 1}],
    })
    assert song.n_bars == 8


def test_three_four_time():
    song = _expand_song({
        "Title": "t", "Composer": "c", "TimeSignature": "3/4",
        "Sections": [{"MainSegment": {"Chords": "C|F|G|C"}}],
    })
    assert song.beats_per_bar == 3
    assert song.n_beats == 12


# --- the real corpus --------------------------------------------------------

@pytest.fixture(scope="module")
def corpus():
    try:
        return load_corpus()
    except FileNotFoundError:
        pytest.skip("corpus not fetched; run scripts/fetch_data.py")


def test_corpus_loads(corpus):
    assert len(corpus) > 1000


def test_standard_forms_have_expected_lengths(corpus):
    by = {s.title: s for s in corpus}
    expected = {
        "Blue Monk": 12, "Straight No Chaser": 12,   # blues
        "Oleo": 32, "Anthropology": 32,              # rhythm changes, AABA
        "52nd Street Theme": 32,                     # AABA via the Repeats field
        "Autumn Leaves": 32,
        "So What": 32,
        "Cherokee": 48,
    }
    for title, n_bars in expected.items():
        assert by[title].n_bars == n_bars, title


def test_every_beat_carries_a_chord(corpus):
    for song in corpus:
        assert song.n_beats == song.n_bars * song.beats_per_bar
        assert all(c is not None for c in song.chords)


def test_families_recover_known_contrafacts(corpus):
    families, family_of = build_families(corpus)
    index = {s.title: i for i, s in enumerate(corpus)}

    def same_family(a, b):
        return family_of[index[a]] == family_of[index[b]]

    # Tunes written on borrowed changes must land together...
    assert same_family("Donna Lee", "Indiana (Back Home Again In)")
    assert same_family("So What", "Impressions")
    assert same_family("Cherokee", "Ko Ko")
    assert same_family("Half Nelson", "Lady Bird")
    assert same_family("Hot House", "What Is This Thing Called Love")
    assert same_family("Blue Monk", "Straight No Chaser")
    assert same_family("Oleo", "Anthropology")

    # ...and tunes that merely share a key and era must not.
    assert not same_family("Autumn Leaves", "Blue Monk")
    assert not same_family("Oleo", "So What")


def test_blues_and_rhythm_families_are_the_big_ones(corpus):
    families, _ = build_families(corpus)
    biggest = families[0]
    assert len(biggest) > 20
    titles = {corpus[i].title for i in biggest.members}
    assert {"Blue Monk", "Billie's Bounce", "Bags' Groove"} <= titles

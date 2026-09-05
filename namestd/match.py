"""Match beat-synchronous chroma against the corpus of changes.

The problem is a search over three unknowns at once: which tune, in which key,
and starting where in the form.  Written out, matching a query X against a
reference chord sequence T of length L beats is

    score(transposition, rotation) = mean over query beats i of
        < X[i] , template(T[(rotation + i) mod L] transposed) >

Three things make that cheap.

*Rotation is a circular cross-correlation*, so all L rotations come from one
FFT rather than L separate alignments.

*Transposition rotates the chroma vector*, so instead of building twelve copies
of the corpus we rotate the (small) query twelve times and reuse one precomputed
corpus FFT across every query.

*The query folds modulo L.*  Because ``(rotation + i) mod L`` only depends on
``i mod L``, summing the query's beats into L bins loses nothing - and that fold
is exactly the accumulation of evidence across choruses.  A six-minute
performance of a 32-bar tune contributes all fifteen of its choruses to the same
32 bars of evidence, which is why confidence should climb the longer you listen
rather than being fixed by a short window the way a fingerprinter's is.

The score reported is the peak's *prominence*: how far the best rotation stands
above that song's own average across rotations.  Raw similarity is not
comparable between tunes, because a harmonically bland tune scores moderately
well everywhere.  What identifies a tune is a sharp peak at one alignment.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np

from .chords import template_matrix
from .corpus import Song
from .families import Family
from .form import allowed_lengths, detect_periods
from .frontend import Features, smooth_chroma

DEFAULT_SMOOTH = 3


@dataclass
class Match:
    """One candidate answer."""

    song_index: int
    title: str
    composer: str
    score: float            # peak prominence, the ranking statistic
    similarity: float       # raw mean cosine at the winning alignment
    transposition: int      # semitones the performance is above the chart
    rotation: int           # query beat 0 lands here in the chart
    n_bars: int
    family_index: int = -1
    family_name: str = ""
    family_size: int = 1

    def __repr__(self) -> str:
        return (f"<Match {self.title!r} score={self.score:.3f} "
                f"+{self.transposition} semitones, from beat {self.rotation}>")


@dataclass
class MatchReport:
    """Ranked matches plus the family-level answer."""

    matches: list[Match]
    families: list[tuple[str, float, list[str]]]   # (name, score, member titles)
    n_beats: int
    tempo: float
    periods: list[tuple[int, float]] = field(default_factory=list)

    @property
    def best(self) -> Match | None:
        return self.matches[0] if self.matches else None

    @property
    def margin(self) -> float:
        """Gap between the top two *families* - the real confidence signal.

        The gap between the top two tunes is often meaningless, because the top
        two are frequently contrafacts sitting on identical changes.
        """
        if len(self.families) < 2:
            return float("inf")
        return self.families[0][1] - self.families[1][1]


class Matcher:
    """Precomputes the corpus side of the correlation once, then answers queries."""

    def __init__(self, songs: list[Song], families: list[Family] | None = None,
                 family_of: list[int] | None = None):
        self.songs = songs
        self.families = families
        self.family_of = family_of

        # Bucket by chorus length: correlation is only defined between sequences
        # of the same length, and most tunes share a handful of lengths.
        self._buckets: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        by_length: dict[int, list[int]] = defaultdict(list)
        for i, song in enumerate(songs):
            by_length[song.n_beats].append(i)

        for length, indices in by_length.items():
            templates = np.stack([template_matrix(songs[i].chords) for i in indices])
            # (n_songs, L, 24) -> (n_songs, 24, L) so the FFT runs over time.
            templates = np.ascontiguousarray(templates.transpose(0, 2, 1))
            spectra = np.fft.rfft(templates, axis=2)
            self._buckets[length] = (np.asarray(indices, dtype=np.int32), spectra)

    # -- query -------------------------------------------------------------

    def match(self, features: Features, top_k: int = 10,
              smooth_window: int = DEFAULT_SMOOTH,
              min_beats: int = 16, use_form: bool = True) -> MatchReport:
        """Rank the corpus against one Features."""
        chroma = smooth_chroma(features.chroma, smooth_window)
        n_beats = chroma.shape[0]
        if n_beats < min_beats:
            return MatchReport([], [], n_beats, features.tempo)

        # Detecting the chorus period needs no database, and restricting the
        # search to forms of a compatible length removes most of the accidental
        # cross-length matches - a 12-bar blues has no business being scored
        # against a 32-bar tune at some lucky rotation.
        lengths = list(self._buckets)
        periods: list[tuple[int, float]] = []
        if use_form:
            periods = detect_periods(chroma)
            lengths = allowed_lengths(periods, lengths)

        scores = np.full(len(self.songs), -np.inf, dtype=np.float32)
        similarity = np.zeros(len(self.songs), dtype=np.float32)
        best_t = np.zeros(len(self.songs), dtype=np.int16)
        best_r = np.zeros(len(self.songs), dtype=np.int32)

        for length in lengths:
            indices, spectra = self._buckets[length]
            folded = _fold(chroma, length)                    # (12, L, 24)
            # (12, 24, L) -> spectra, then correlate against every song at once.
            query_spectra = np.fft.rfft(
                np.ascontiguousarray(folded.transpose(0, 2, 1)), axis=2)

            # (transposition, song, dim, lag) summed over dim -> (t, song, lag)
            product = np.conj(query_spectra)[:, None, :, :] * spectra[None, :, :, :]
            corr = np.fft.irfft(product.sum(axis=2), n=length, axis=2) / n_beats

            # Prominence: how far the best rotation stands above this song's own
            # average across rotations.  Subtracting the mean cancels the "matches
            # everything moderately" advantage of harmonically bland tunes.
            #
            # Dividing by the *spread* across rotations, which is the tempting
            # next step, turned out to be badly wrong: an AABA form is genuinely
            # self-similar, so its A section repeating inflates the spread and the
            # most common form in the repertoire gets penalised for having a form.
            # Measured over the corpus, that statistic ranks the true tune first
            # 71% of the time against 98% for this one.
            #
            # The divisor instead corrects for having taken a maximum over
            # 12 x L trials.  The expected maximum of N draws grows like
            # sqrt(2 ln N), so without it long forms outscore short ones purely by
            # having had more chances - a 12-bar blues cannot compete with a
            # 32-bar tune on a fair peak.
            prominence = (corr - corr.mean(axis=2, keepdims=True)) / _max_correction(length)

            flat = prominence.reshape(prominence.shape[0], prominence.shape[1], -1)
            per_t_best = flat.argmax(axis=2)                              # (t, song)
            per_t_value = np.take_along_axis(flat, per_t_best[..., None], 2)[..., 0]

            t_star = per_t_value.argmax(axis=0)                           # (song,)
            song_slot = np.arange(len(indices))
            value = per_t_value[t_star, song_slot]
            rotation = per_t_best[t_star, song_slot]

            scores[indices] = value
            best_t[indices] = t_star
            best_r[indices] = rotation
            similarity[indices] = corr[t_star, song_slot, rotation]

        report = self._report(scores, similarity, best_t, best_r, top_k,
                              n_beats, features.tempo)
        report.periods = periods
        return report

    def match_best_of(self, hypotheses: list[Features], **kwargs) -> MatchReport:
        """Run several tempo hypotheses and keep the most confident.

        This is where the beat tracker's octave error gets resolved: a grid at
        the wrong tempo cannot make any form line up, so its best score stays
        low.  The matcher has evidence the beat tracker never had.
        """
        best: MatchReport | None = None
        for features in hypotheses:
            report = self.match(features, **kwargs)
            if report.best is None:
                continue
            if best is None or report.best.score > best.best.score:
                best = report
        return best or MatchReport([], [], 0, 0.0)

    # -- reporting ---------------------------------------------------------

    def _report(self, scores, similarity, best_t, best_r, top_k, n_beats, tempo
                ) -> MatchReport:
        order = np.argsort(-scores)[:max(top_k, 40)]
        matches = []
        for i in order:
            song = self.songs[i]
            match = Match(
                song_index=int(i), title=song.title, composer=song.composer,
                score=float(scores[i]), similarity=float(similarity[i]),
                # A chart in Bb heard in C was transposed up; the correlation
                # finds the shift applied to the query, so invert it.
                transposition=int(-best_t[i] % 12),
                rotation=int(best_r[i]), n_bars=song.n_bars,
            )
            if self.families is not None and self.family_of is not None:
                fam_index = self.family_of[i]
                family = self.families[fam_index]
                match.family_index = fam_index
                match.family_name = family.name
                match.family_size = len(family)
            matches.append(match)

        # Roll tunes up to families, scoring each family by its best member.
        family_scores: dict[int, tuple[float, str, list[str]]] = {}
        for match in matches:
            key = match.family_index if match.family_index >= 0 else -match.song_index - 1
            name = match.family_name or match.title
            if key not in family_scores:
                family_scores[key] = (match.score, name, [])
            if len(family_scores[key][2]) < 8:
                family_scores[key][2].append(match.title)
        families = sorted(
            ((name, score, titles) for score, name, titles in family_scores.values()),
            key=lambda item: -item[1],
        )

        return MatchReport(matches[:top_k], families, n_beats, tempo)


def _max_correction(length: int) -> float:
    """Gumbel scale for a maximum taken over ``12 * length`` alignments."""
    return float(np.sqrt(2.0 * np.log(max(12 * length, 2))))


def _fold(chroma: np.ndarray, length: int) -> np.ndarray:
    """Fold a query into ``length`` bins, once per transposition.

    Returns (12, length, 24).  Transposition ``t`` rotates both the bass and the
    treble half of each chroma vector by ``t`` semitones; rotating the query is
    equivalent to transposing the whole corpus, and vastly cheaper.
    """
    n_beats = chroma.shape[0]
    bins = np.zeros((length, 24), dtype=np.float32)
    # np.add.at handles the repeated indices that folding necessarily produces.
    np.add.at(bins, np.arange(n_beats) % length, chroma)

    out = np.empty((12, length, 24), dtype=np.float32)
    for t in range(12):
        out[t, :, 0:12] = np.roll(bins[:, 0:12], t, axis=1)
        out[t, :, 12:24] = np.roll(bins[:, 12:24], t, axis=1)
    return out


def build_matcher(songs: list[Song]) -> Matcher:
    from .families import build_families

    families, family_of = build_families(songs)
    return Matcher(songs, families, family_of)

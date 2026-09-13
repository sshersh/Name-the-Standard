"""Measure identification accuracy against listening time.

The headline number is not top-1 accuracy at some fixed window - that is a
fingerprinter's metric.  It is the *curve*: how confidence grows the longer the
band plays, since this system accumulates evidence chorus after chorus.

Two accuracies are reported and they mean different things.  Tune accuracy is
bounded above by the contrafacts: nothing in the harmony separates Oleo from
Anthropology, so a system that got 100% here would be reading tea leaves.
Family accuracy is what harmony can honestly deliver, and it is the number to
judge this stage on.

Run:  python -m namestd.eval synthetic --songs 100
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass, field

import numpy as np

from dataclasses import replace as _replace

from .corpus import Song, load_corpus
from .degrade import band_limit, crowd, play_outside, reharmonise, reverb
from .frontend import Features, analyse, analyse_hypotheses
from .match import Matcher, build_matcher
from .synth import PerformanceStyle, random_style, render

DEFAULT_BEAT_BUDGETS = (32, 64, 128, 256, 512)


@dataclass
class Outcome:
    """What happened for one simulated performance at one listening length."""

    beats: int
    tune_rank: int | None
    family_rank: int | None
    transposition_ok: bool
    score: float
    margin: float


@dataclass
class Results:
    budgets: tuple[int, ...]
    rows: list[list[Outcome]] = field(default_factory=list)

    def table(self) -> str:
        lines = [
            f"{len(self.rows)} simulated performances, random key / tempo / entry point",
            "",
            f"{'beats':>7} {'~bars':>6} | {'family top1':>12} {'family top5':>12}"
            f" | {'tune top1':>10} {'tune top5':>10} | {'key ok':>7} {'margin':>7}",
            "-" * 88,
        ]
        for i, beats in enumerate(self.budgets):
            column = [row[i] for row in self.rows if row[i] is not None]
            if not column:
                continue
            fam = [o.family_rank for o in column if o.family_rank is not None]
            tune = [o.tune_rank for o in column if o.tune_rank is not None]
            n = len(column)
            f1 = 100.0 * sum(1 for r in fam if r == 0) / n
            f5 = 100.0 * sum(1 for r in fam if r < 5) / n
            t1 = 100.0 * sum(1 for r in tune if r == 0) / n
            t5 = 100.0 * sum(1 for r in tune if r < 5) / n
            key = 100.0 * sum(1 for o in column if o.transposition_ok) / n
            margin = float(np.median([o.margin for o in column]))
            lines.append(
                f"{beats:>7} {beats // 4:>6} | {f1:>11.1f}% {f5:>11.1f}%"
                f" | {t1:>9.1f}% {t5:>9.1f}% | {key:>6.1f}% {margin:>7.3f}"
            )
        return "\n".join(lines)


def _rank_of(report, predicate) -> int | None:
    for i, match in enumerate(report.matches):
        if predicate(match):
            return i
    return None


def evaluate_synthetic(matcher: Matcher, songs, n_songs: int = 100,
                       budgets: tuple[int, ...] = DEFAULT_BEAT_BUDGETS,
                       seed: int = 0, oracle_beats: bool = False,
                       verbose: bool = True) -> Results:
    """Render random performances and score the matcher on each."""
    rng = np.random.default_rng(seed)
    # Only tunes long enough to fill the largest budget with a few choruses.
    eligible = [i for i, s in enumerate(songs) if 32 <= s.n_beats <= 256]
    chosen = rng.choice(eligible, size=min(n_songs, len(eligible)), replace=False)

    results = Results(budgets=budgets)
    started = time.time()
    for count, index in enumerate(chosen, start=1):
        song = songs[index]
        needed = max(budgets)
        style = random_style(rng, n_choruses=int(np.ceil(needed / song.n_beats)) + 1,
                             start_offset_bars=int(rng.integers(0, song.n_bars)))
        audio, truth = render(song, style)

        if oracle_beats:
            hypotheses = [analyse(audio, beat_times=truth["beat_times"])]
        else:
            hypotheses = analyse_hypotheses(audio)

        row: list[Outcome] = []
        for beats in budgets:
            windows = [
                Features(beat_times=h.beat_times[:beats],
                         chroma=h.chroma[:beats], tempo=h.tempo)
                for h in hypotheses if h.chroma.shape[0] >= min(beats, 24)
            ]
            if not windows:
                row.append(Outcome(beats, None, None, False, 0.0, 0.0))
                continue
            report = matcher.match_best_of(windows, top_k=40)
            if not report.matches:
                row.append(Outcome(beats, None, None, False, 0.0, 0.0))
                continue

            true_family = matcher.family_of[index]
            row.append(Outcome(
                beats=beats,
                tune_rank=_rank_of(report, lambda m: m.song_index == index),
                family_rank=_rank_of(report, lambda m: m.family_index == true_family),
                transposition_ok=report.best.transposition == style.transpose % 12,
                score=report.best.score,
                margin=min(report.margin, 9.99),
            ))
        results.rows.append(row)

        if verbose and count % 10 == 0:
            rate = (time.time() - started) / count
            print(f"  {count}/{len(chosen)} ({rate:.1f}s each)", flush=True)
    return results


# Each condition isolates one way a real performance differs from a clean
# synthetic one, so a drop can be attributed instead of guessed at.
CONDITIONS: dict[str, str] = {
    "clean":     "ideal synthetic baseline",
    "reverb":    "small room, harmony smeared across bar lines",
    "phone":     "phone mic: nothing below 150 Hz, so the bass band is gutted",
    "crowd":     "room tone and audience rumble",
    "rubato":    "loose time, the beat grid drifting against the form",
    "reharm":    "tritone subs, ii inserted before V, relative minor for tonic",
    "outside":   "a quarter of the bars played on unrelated harmony",
    "no_piano":  "piano-less trio: harmony implied by the bass alone",
    "club":      "everything at once",
}


def _apply_condition(song: Song, condition: str, style: PerformanceStyle,
                     seed: int) -> tuple[Song, PerformanceStyle]:
    """Symbol-level damage: what the band plays, before any audio exists."""
    chords = song.chords
    if condition in ("reharm", "club"):
        chords = reharmonise(chords, 0.35, seed)
    if condition in ("outside", "club"):
        chords = play_outside(chords, 0.25, seed + 1)
    if condition in ("rubato", "club"):
        style.rubato = 0.055
    if condition in ("no_piano", "club"):
        style.comp_level = 0.0
    return (_replace(song, chords=chords) if chords is not song.chords else song), style


def _degrade_audio(audio: np.ndarray, condition: str, seed: int) -> np.ndarray:
    """Room-level damage: what happens between the band and the microphone."""
    sr = 22050
    if condition in ("reverb", "club"):
        audio = reverb(audio, sr, seconds=1.1, wet=0.35, seed=seed)
    if condition in ("phone", "club"):
        audio = band_limit(audio, sr, low_hz=150.0, high_hz=5000.0)
    if condition in ("crowd", "club"):
        audio = crowd(audio, sr, level=0.06, seed=seed)
    return audio


def evaluate_stress(matcher: Matcher, songs, n_songs: int = 20,
                    beats: int = 512, seed: int = 0,
                    conditions: list[str] | None = None) -> str:
    """Family top-1 under each condition, everything else held equal."""
    names = conditions or list(CONDITIONS)
    eligible = [i for i, s in enumerate(songs) if 32 <= s.n_beats <= 256]
    picked = np.random.default_rng(seed).choice(
        eligible, size=min(n_songs, len(eligible)), replace=False)

    lines = [f"{len(picked)} tunes per condition, {beats} beats heard "
             f"(~{beats // 4} bars), full pipeline including beat tracking",
             "",
             f"{'condition':<10} {'family top1':>12} {'tune top1':>10} {'key ok':>8}   what it simulates",
             "-" * 96]

    for condition in names:
        hits = tune_hits = key_hits = total = 0
        for index in picked:
            song = songs[int(index)]
            rng = np.random.default_rng(int(index) * 7919 + seed)
            style = random_style(
                rng,
                n_choruses=int(np.ceil(beats / song.n_beats)) + 1,
                start_offset_bars=int(rng.integers(0, song.n_bars)),
            )
            played, style = _apply_condition(song, condition, style, int(index))
            audio, _ = render(played, style)
            audio = _degrade_audio(audio, condition, int(index))

            windows = []
            for hypothesis in analyse_hypotheses(audio):
                if hypothesis.chroma.shape[0] >= 24:
                    windows.append(Features(
                        beat_times=hypothesis.beat_times[:beats],
                        chroma=hypothesis.chroma[:beats], tempo=hypothesis.tempo))
            total += 1
            if not windows:
                continue
            report = matcher.match_best_of(windows, top_k=5)
            if not report.matches:
                continue
            best = report.best
            hits += best.family_index == matcher.family_of[int(index)]
            tune_hits += best.song_index == int(index)
            key_hits += best.transposition == style.transpose % 12

        lines.append(
            f"{condition:<10} {100.0 * hits / total:>11.0f}% {100.0 * tune_hits / total:>9.0f}%"
            f" {100.0 * key_hits / total:>7.0f}%   {CONDITIONS.get(condition, '')}"
        )
        print("  done:", condition, flush=True)
    return "\n".join(lines)


def evaluate_ablation(matcher: Matcher, songs, n_songs: int = 20,
                      beats: int = 512, seed: int = 3) -> str:
    """How much does each chroma band actually contribute?

    The bass/treble split is the front end's main design claim, so it should be
    checked rather than assumed: zero one half of every chroma vector at match
    time and see what the loss costs.
    """
    eligible = [i for i, s_ in enumerate(songs) if 32 <= s_.n_beats <= 256]
    picked = np.random.default_rng(seed).choice(
        eligible, size=min(n_songs, len(eligible)), replace=False)

    modes = {"both bands": None, "bass only": "treble", "treble only": "bass"}
    hits = {k: 0 for k in modes}
    for index in picked:
        song = songs[int(index)]
        rng = np.random.default_rng(int(index) * 7919 + seed)
        style = random_style(rng, n_choruses=int(np.ceil(beats / song.n_beats)) + 1,
                             start_offset_bars=int(rng.integers(0, song.n_bars)))
        audio, _ = render(song, style)
        hypotheses = analyse_hypotheses(audio)

        for name, drop in modes.items():
            windows = []
            for h in hypotheses:
                if h.chroma.shape[0] < 24:
                    continue
                chroma = h.chroma[:beats].copy()
                if drop == "bass":
                    chroma[:, 0:12] = 0.0
                elif drop == "treble":
                    chroma[:, 12:24] = 0.0
                windows.append(Features(h.beat_times[:beats], chroma, h.tempo))
            if not windows:
                continue
            report = matcher.match_best_of(windows, top_k=5)
            if report.matches and report.best.family_index == matcher.family_of[int(index)]:
                hits[name] += 1

    lines = [f"{len(picked)} tunes, {beats} beats, one chroma band zeroed at match time", ""]
    for name, count in hits.items():
        lines.append(f"  {name:<12} family top1 {100.0 * count / len(picked):>3.0f}%  "
                     f"({count}/{len(picked)})")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["synthetic", "stress", "ablate"],
                        nargs="?", default="synthetic")
    parser.add_argument("--songs", type=int, default=60)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--oracle-beats", action="store_true",
                        help="use the true beat grid, isolating matcher from beat tracking")
    args = parser.parse_args(argv)

    songs = load_corpus()
    print(f"corpus: {len(songs)} tunes")
    matcher = build_matcher(songs)
    print(f"families: {len(matcher.families)}")
    print()

    if args.mode == "ablate":
        print(evaluate_ablation(matcher, songs, n_songs=args.songs, seed=args.seed))
        return 0

    if args.mode == "stress":
        print(evaluate_stress(matcher, songs, n_songs=args.songs, seed=args.seed))
        return 0

    results = evaluate_synthetic(matcher, songs, n_songs=args.songs,
                                 seed=args.seed, oracle_beats=args.oracle_beats)
    print()
    print(results.table())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

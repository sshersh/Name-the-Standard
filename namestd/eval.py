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

from .corpus import load_corpus
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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["synthetic"], nargs="?", default="synthetic")
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

    results = evaluate_synthetic(matcher, songs, n_songs=args.songs,
                                 seed=args.seed, oracle_beats=args.oracle_beats)
    print()
    print(results.table())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Command line entry point.

    namestd identify recording.wav
    namestd identify recording.wav --progress    # show confidence growing
    namestd demo "Autumn Leaves" --key Eb        # synthesise, then identify
"""

from __future__ import annotations

import argparse
import sys

import numpy as np

from .chords import PITCH_CLASSES
from .corpus import load_corpus
from .frontend import Features, analyse_hypotheses, load_audio
from .match import MatchReport, Matcher, build_matcher


def _format_report(report: MatchReport, show_tunes: int = 6) -> str:
    if not report.matches:
        return "No match: too little audio, or no steady pulse to lock onto."

    lines = []
    period = report.periods[0][0] if report.periods else None
    form = f"{period // 4}-bar form" if period else "form unclear"
    lines.append(
        f"heard {report.n_beats} beats at {report.tempo:.0f} bpm  |  {form}"
    )
    lines.append("")

    best = report.best
    key = PITCH_CLASSES[best.transposition]
    lines.append(f"  Best guess: {best.title}")
    if best.family_size > 1:
        lines.append(f"  These changes are shared by {best.family_size} tunes "
                     f"({best.family_name}); harmony alone cannot separate them.")
    lines.append(f"  Played {best.transposition} semitones above the chart "
                 f"(chart in {best.n_bars} bars; heard root {key}).")
    lines.append("")

    lines.append("  Ranked by changes:")
    for i, match in enumerate(report.matches[:show_tunes], start=1):
        lines.append(f"   {i:>2}. {match.title:<34} {match.score:6.3f}"
                     f"   {match.n_bars:>3} bars  +{match.transposition:<2}"
                     f"  {match.family_name if match.family_size > 1 else ''}")

    if len(report.families) > 1:
        lines.append("")
        lines.append(f"  Confidence margin over the next family: {report.margin:.3f}")
    return "\n".join(lines)


def _identify(matcher: Matcher, path: str, progress: bool) -> int:
    print(f"reading {path} ...", flush=True)
    audio = load_audio(path)
    print(f"  {len(audio) / 22050:.1f}s of audio", flush=True)

    hypotheses = analyse_hypotheses(audio)
    if not hypotheses or hypotheses[0].n_beats < 16:
        print("Could not find a steady pulse - is there music in this file?")
        return 1

    if progress:
        # Show the answer firming up as more of the performance is heard, which
        # is the behaviour that distinguishes this from a fingerprinter.
        print()
        print(f"  {'beats':>6} {'best guess':<32} {'score':>7} {'margin':>7}")
        for beats in (32, 64, 128, 256, 512, 1024):
            windows = [Features(h.beat_times[:beats], h.chroma[:beats], h.tempo)
                       for h in hypotheses if h.n_beats >= min(beats, 24)]
            if not windows:
                break
            report = matcher.match_best_of(windows, top_k=10)
            if report.best:
                print(f"  {min(beats, windows[0].n_beats):>6} "
                      f"{report.best.title[:32]:<32} {report.best.score:>7.3f}"
                      f" {report.margin:>7.3f}")
            if windows[0].n_beats < beats:
                break

    print()
    print(_format_report(matcher.match_best_of(hypotheses, top_k=40)))
    return 0


def _demo(matcher: Matcher, songs, title: str, transpose: int, tempo: float,
          choruses: int, offset: int) -> int:
    from .synth import PerformanceStyle, render

    matches = [s for s in songs if s.title.lower() == title.lower()]
    if not matches:
        close = [s.title for s in songs if title.lower() in s.title.lower()][:8]
        print(f"No tune called {title!r}.", file=sys.stderr)
        if close:
            print("Did you mean: " + ", ".join(close), file=sys.stderr)
        return 1
    song = matches[0]

    audio, _ = render(song, PerformanceStyle(
        tempo=tempo, transpose=transpose, n_choruses=choruses,
        start_offset_bars=offset, seed=7))
    print(f"synthesised {song.title}: {tempo:.0f} bpm, "
          f"+{transpose} semitones, entering at bar {offset}")
    print()
    print(_format_report(matcher.match_best_of(analyse_hypotheses(audio), top_k=40)))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="namestd", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    identify = sub.add_parser("identify", help="name the tune in an audio file")
    identify.add_argument("audio")
    identify.add_argument("--progress", action="store_true",
                          help="show confidence growing with listening time")

    demo = sub.add_parser("demo", help="synthesise a tune, then try to identify it")
    demo.add_argument("title")
    demo.add_argument("--key", type=int, default=0, help="semitones to transpose")
    demo.add_argument("--tempo", type=float, default=170.0)
    demo.add_argument("--choruses", type=int, default=3)
    demo.add_argument("--offset", type=int, default=0, help="enter at this bar")

    args = parser.parse_args(argv)

    songs = load_corpus()
    matcher = build_matcher(songs)

    if args.command == "identify":
        return _identify(matcher, args.audio, args.progress)
    return _demo(matcher, songs, args.title, args.key, args.tempo,
                 args.choruses, args.offset)


if __name__ == "__main__":
    raise SystemExit(main())

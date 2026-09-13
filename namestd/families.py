"""Group tunes that share a chord progression into contrafact families.

Harmony alone cannot tell Oleo from Anthropology - they are written on the same
changes, note for note.  Rather than pretend otherwise, we cluster the corpus
into families up front so the matcher can return "rhythm changes in Bb" as an
honest intermediate answer and hand the tie-break to melody.

Two tunes join a family when their beat-level chords agree above a threshold
under the *best* of the twelve transpositions - a contrafact is often played in
a different key from the tune it borrows from.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import numpy as np

from .corpus import Song

DEFAULT_THRESHOLD = 0.85


@dataclass
class Family:
    """A set of tunes on (nearly) the same changes."""

    members: list[int]          # indices into the song list
    name: str                   # display name, from the best-known member
    n_bars: int

    def __len__(self) -> int:
        return len(self.members)


class _UnionFind:
    def __init__(self, n: int):
        self.parent = list(range(n))

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[max(ra, rb)] = min(ra, rb)


def _agreement_matrix(codes: np.ndarray, chunk: int = 64) -> np.ndarray:
    """Pairwise agreement, maximised over the 12 transpositions.

    ``codes`` is (n_songs, n_beats) of ``root * 6 + quality``.
    """
    roots = (codes // 6).astype(np.int8)
    quals = (codes % 6).astype(np.int8)
    n, length = codes.shape
    best = np.zeros((n, n), dtype=np.float32)

    for start in range(0, n, chunk):
        stop = min(start + chunk, n)
        r_chunk = roots[start:stop, None, :]
        q_match = quals[start:stop, None, :] == quals[None, :, :]
        acc = np.zeros((stop - start, n), dtype=np.float32)
        for t in range(12):
            match = ((r_chunk + t) % 12 == roots[None, :, :]) & q_match
            np.maximum(acc, match.mean(axis=2, dtype=np.float32), out=acc)
        best[start:stop] = acc
    return best


def build_families(songs: list[Song], threshold: float = DEFAULT_THRESHOLD
                   ) -> tuple[list[Family], list[int]]:
    """Cluster ``songs`` into families.

    Returns the families and, for each song, the index of its family.

    Only tunes of identical length are compared.  That is a deliberate limit: a
    36-bar I Got Rhythm (with its tag) and a 32-bar Oleo really are different
    forms to play over, even though a musician would call them the same changes.
    """
    uf = _UnionFind(len(songs))

    buckets: dict[tuple[int, int], list[int]] = defaultdict(list)
    for i, song in enumerate(songs):
        buckets[(song.n_beats, song.beats_per_bar)].append(i)

    for indices in buckets.values():
        if len(indices) < 2:
            continue
        codes = np.stack([songs[i].codes() for i in indices])
        agree = _agreement_matrix(codes)
        rows, cols = np.nonzero(np.triu(agree >= threshold, k=1))
        for r, c in zip(rows, cols):
            uf.union(indices[r], indices[c])

    groups: dict[int, list[int]] = defaultdict(list)
    for i in range(len(songs)):
        groups[uf.find(i)].append(i)

    families: list[Family] = []
    family_of = [0] * len(songs)
    for members in groups.values():
        members.sort()
        families.append(
            Family(members=members, name=_family_name(songs, members),
                   n_bars=songs[members[0]].n_bars)
        )
    families.sort(key=lambda f: (-len(f), f.name))
    for idx, family in enumerate(families):
        for member in family.members:
            family_of[member] = idx
    return families, family_of


# Well-known families get their musician's name rather than an arbitrary member.
# Matching is on the family's *shape*, so these labels are only cosmetic.
_KNOWN = [
    ("Blue Monk", "12-bar blues"),
    ("Now's The Time", "12-bar blues"),
    ("Oleo", "rhythm changes"),
    ("Anthropology", "rhythm changes"),
]


def _family_name(songs: list[Song], members: list[int]) -> str:
    titles = {songs[i].title for i in members}
    for title, label in _KNOWN:
        if title in titles:
            return label
    return songs[members[0]].title


def describe(families: list[Family], songs: list[Song], limit: int = 10) -> str:
    """Human-readable summary of the largest families."""
    lines = []
    multi = [f for f in families if len(f) > 1]
    lines.append(
        f"{len(families)} families over {len(songs)} tunes "
        f"({len(multi)} with more than one member)"
    )
    for family in families[:limit]:
        if len(family) < 2:
            break
        titles = ", ".join(songs[i].title for i in family.members[:8])
        more = "" if len(family) <= 8 else f", +{len(family) - 8} more"
        lines.append(f"  [{len(family):3d}] {family.name} ({family.n_bars} bars): {titles}{more}")
    return "\n".join(lines)

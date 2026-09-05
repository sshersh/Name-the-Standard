# Naming the standard: recognition algorithm

How to identify the jazz standard being played in a live performance, and where
machine learning is and is not required.

---

## 1. Why Shazam's approach cannot work

Shazam fingerprints the constellation of spectral peaks in a *specific master
recording* and looks that pattern up in a hash table. It is superb at what it
does and completely inapplicable here.

A quartet in a club playing "Autumn Leaves" shares no spectral peaks with the
Cannonball Adderley recording. Different players, different instruments,
different key, different tempo, different arrangement, and — after the first
chorus — different notes entirely, because everyone is improvising. Audio
fingerprinting has essentially zero recall on live performance. Not poor recall:
zero. There is no tuning that rescues it, because the thing it matches on is not
present.

The nearest well-studied analogue is **cover-song identification**, which does
handle key and tempo changes. But cover-song systems are trained on studio
recordings of pop covers and generally compare whole tracks, and — as section 4
argues — the learned versions need training data that does not exist for this
problem.

## 2. What is actually invariant

What the musicians on the bandstand have agreed on is the **form and the chord
changes**. Two facts about that make this problem tractable, and they are the
foundation of everything below.

**The changes repeat every chorus, for the whole performance.** A six-minute
tune at 200bpm is around fifteen choruses of the same 32-bar harmonic cycle. The
solos are *not* noise to be discarded — they are played over the same changes,
so they are fifteen more pieces of evidence for the same answer. This is the
central asymmetry with Shazam: a fingerprinter gets a fixed 10-second window,
while this system's confidence should climb the longer it listens. It does; see
section 5.

**The head is stated at the top and again at the end.** That is where the melody
lives, and melody is what separates tunes built on the same changes.

Everything else varies and must be searched over or normalised away: the key
(singers move tunes freely), the tempo, and where in the form the listener
happened to start.

## 3. The pipeline

    audio
      -> beat tracking, CQT, bass+treble chroma        frontend.py
      -> chorus period from self-similarity            form.py
      -> tune x key x position, by cross-correlation   match.py
      -> contrafact family                             families.py
      -> melody of the head breaks the tie             melody.py

### 3.1 Front end — index by beats, not seconds

Mono at 22.05 kHz, constant-Q transform, folded to twelve pitch classes, then
**pooled onto the beat grid**. Pooling by beat is what makes the whole system
tempo-invariant for free: the same tune at 140 and at 260 produces the same
sequence of vectors.

Two chroma bands are kept **separate**, not summed into one 12-vector:

- **bass** (C1–B2) — a walking bassline states the root explicitly, so this band
  carries root motion;
- **treble** (C3–B5) — comping and horns, carrying thirds, sevenths and tensions.

The split matters because jazz pianists play *rootless* voicings. The root is
usually absent from the comping and present only in the bass. Ordinary chroma
folds those together and discards the distinction.

**No downbeat tracking.** The obvious next step after beat tracking is finding
bar lines, and it is the most fragile stage in a typical pipeline. It is skipped
entirely: the matcher searches every rotation at *beat* resolution, so the form's
phase — and with it the downbeat — falls out of the alignment. This removes a
whole class of failure at the cost of searching 4x more rotations, which is
free given the FFT formulation below.

### 3.2 Form detection — before touching the database

The chorus period is recovered by comparing each beat's chroma against the beat
one candidate period later and averaging. The true period shows a clean peak.
This needs no database at all, and it prunes the search to tunes whose form
could actually be playing: a 12-bar blues query stops being scored against
32-bar tunes on the strength of a lucky alignment.

Measured on rendered audio, this identifies the exact form length as its first
hypothesis for 10 of 10 test tunes across 12-, 32-, 36-, 44- and 48-bar forms.

One implementation detail turned out to matter enormously. The baseline that
gets subtracted before peak-picking must be padded by **edge replication**.
Zero-padding drags the baseline down at both ends of the period range and
manufactures spurious peaks at the shortest and longest period considered —
which excluded every true form length and broke the whole system until it was
found.

### 3.3 Harmonic matching — one correlation over three unknowns

Matching a query `X` against a reference chord sequence `T` of length `L` beats:

    score(transposition, rotation)
        = mean over query beats i of  < X[i], template(T[(rotation + i) mod L]) >

Three properties make searching all of tune x key x position cheap:

- **Rotation is a circular cross-correlation**, so all `L` rotations come from
  one FFT rather than `L` separate alignments.
- **Transposition rotates the chroma vector**, so instead of storing twelve
  copies of the corpus we rotate the (small) query twelve times and reuse one
  precomputed corpus FFT across every query.
- **The query folds modulo `L`.** Since `(rotation + i) mod L` depends only on
  `i mod L`, summing the query's beats into `L` bins loses nothing — and *that
  fold is exactly the accumulation of evidence across choruses.* Every chorus of
  a six-minute performance contributes to the same 32 bars of evidence.

Chords are reduced to a root plus one of six quality classes (major, minor,
dominant, half-diminished, diminished, suspended). Six is deliberate: it is the
coarsest split that keeps apart chords a listener would never confuse, and finer
distinctions — `7b9` against `7#11` — are exactly the alterations improvisers
change at will, so encoding them would model the chart rather than the
performance.

**The ranking statistic was the single largest source of error, and the obvious
choice was wrong.** Ranking by how far the best rotation stands above that song's
own *spread* across rotations — a z-score, the natural first instinct — ranks the
true tune first only **71%** of the time. The reason is that an AABA form is
genuinely self-similar: its A section repeats three times, which inflates the
spread, so the most common form in the entire repertoire gets penalised for
having a form.

Subtracting the mean and dividing instead by the Gumbel scale `sqrt(2 ln 12L)`,
which corrects for having taken a maximum over that many alignments, gives
**98.3% top-1 and 100% top-5** on the same test. That divisor also makes forms of
different lengths comparable — without it a 12-bar blues cannot compete with a
32-bar tune, purely because the longer form had more chances to score well.

### 3.4 Contrafact families — the honest limit of harmony

A large part of the repertoire is written on borrowed changes, and no amount of
harmonic analysis can separate those tunes. Clustering the corpus by harmonic
agreement (best over twelve transpositions) recovers the canonical cases with no
hand-coded list:

| family | members |
|---|---|
| 12-bar blues | 28 tunes — Blue Monk, Billie's Bounce, Bags' Groove, Straight No Chaser, … |
| rhythm changes | 10 tunes — Oleo, Anthropology, Moose The Mooche, Salt Peanuts, … |
| minor blues | Mr. P.C., Equinox, Interplay, Bags and Trane |
| Hot House | What Is This Thing Called Love, Subconscious Lee |
| Donna Lee | Indiana (Back Home Again In) |
| Cherokee | Ko Ko |
| Half Nelson | Lady Bird |
| Dig | Sweet Georgia Brown |
| Impressions | So What |

Oleo and Anthropology are **byte-identical** bar for bar. A system that claimed
to tell them apart from the changes would be reading tea leaves. So the harmonic
stage returns the *family* — "rhythm changes in Bb" — which is both honest and
genuinely useful, and hands the tie-break to melody.

### 3.5 Melody rerank — what the head is for

Because the harmonic match reports *where the query sits in the form*, the head
can be cut exactly rather than guessed at: it is chorus 0, and again the final
chorus.

Predominant pitch is extracted (pYIN), quantised to notes, and converted to a
sequence of **intervals between successive notes** — which makes the comparison
key-blind by construction. Candidates are scored by Smith-Waterman local
alignment, with forgiving costs: a semitone miss still scores positively, and
gaps are cheap, because a jazz musician ornaments the written line constantly.

The structural win is that this runs against **a few dozen candidates inside one
family, not 1382**, so it can afford to be strict where the harmonic search had
to be fast. It mirrors what a musician actually does: *"that's a blues in F…
oh, it's Blue Monk."*

Storing only interval sequences also matters legally: you cannot sing a tune back
from its interval sequence, which makes the index a poor reproduction of
copyrighted material. See section 6.

---

## 4. Does this need machine learning?

**No ML in the recognition core. ML only as off-the-shelf pretrained components
in the front end. Nothing needs to be trained to build v1.**

| Stage | ML? | Notes |
|---|---|---|
| Beat tracking | **Pretrained, effectively required** | Classical onset+autocorrelation struggles badly on swing — ride patterns, walking bass, ballads. `madmom`'s DBN or Beat This! are off-the-shelf; nothing to train. |
| Downbeat tracking | **Not needed at all** | Eliminated by design — the rotation search recovers the phase (§3.1). |
| Chroma | No | CQT and folding. Pure DSP. |
| Chord estimation | Optional, pretrained | Templates suffice; a pretrained model (Chordino, BTC, CREMA) would raise per-beat accuracy. |
| Form detection | No | Self-similarity and autocorrelation. |
| **Harmonic retrieval** | **No** | FFT cross-correlation. Deterministic and interpretable. **This is the core.** |
| Melody extraction | Pretrained | pYIN is pure DSP; CREPE or a Demucs front end would do better on a full band. |
| Melody rerank | No | Smith-Waterman on intervals. |

**Why not end-to-end?** A cover-song embedding (ByteCover, CoverHunter) is the
ML-shaped version of this problem and would work in principle — but it needs
thousands of *labeled live jazz recordings per tune*. That data does not exist
and cannot be bought. The symbolic route needs only chord charts, which do exist,
for 1382 tunes.

The right sequencing is therefore: build the symbolic system, use it to
auto-label a corpus of live recordings, and *then* optionally train an embedding
on that corpus. **ML is the phase-2 accelerator, not the phase-1 foundation.**

One measurement sharpens this. Per-beat chord classification in this system is
**poor** — around 17% top-1 over 72 classes (chance is 1.4%). A system that
committed to per-beat chord labels would be hopeless. But identification over 512
beats is 98% accurate, because the matcher never discretises: it correlates soft
chroma against template *sequences* and accumulates. **Improving per-beat chord
accuracy is therefore the wrong thing to optimise**, and that is precisely where
an ML budget would instinctively be spent.

---

## 5. Measured results

All figures from synthesised performances with random key, tempo (90–260bpm) and
entry point, matched against the full 1382-tune corpus. Reproduce with
`python -m namestd.eval synthetic --songs 60`.

### Accuracy against listening time

The **full pipeline**, with real beat tracking — the honest number:

| beats heard | ~bars | family top-1 | family top-5 | tune top-1 | key correct |
|---|---|---|---|---|---|
| 32 | 8 | 3.3% | 3.3% | 3.3% | 8.3% |
| 64 | 16 | 6.7% | 8.3% | 6.7% | 6.7% |
| 128 | 32 | 53.3% | 56.7% | 53.3% | 58.3% |
| 256 | 64 | 80.0% | 80.0% | 78.3% | 83.3% |
| 512 | 128 | **95.0%** | 95.0% | 93.3% | 96.7% |

Repeated with the *true* beat grid substituted in, which isolates the matcher
from beat-tracking error:

| beats heard | ~bars | family top-1 | family top-5 | tune top-1 | key correct |
|---|---|---|---|---|---|
| 32 | 8 | 6.7% | 18.3% | 5.0% | 11.7% |
| 128 | 32 | 55.0% | 66.7% | 51.7% | 61.7% |
| 256 | 64 | 76.7% | 80.0% | 75.0% | 81.7% |
| 512 | 128 | **98.3%** | 98.3% | 96.7% | 98.3% |

**One chorus is not enough; four choruses is.** The monotone climb is the design
working as intended — it is what the modulo-fold buys.

The gap between the two tables is the entire cost of beat tracking: **about three
points** at full listening length. That is worth stating plainly, because beat
tracking is the one stage where this design leans on a pretrained model, and it
is not currently the bottleneck. Feeding several tempo hypotheses to the matcher
and letting *it* choose (§7) is what keeps that cost small — the matcher can ask
whether a form actually aligns, which is evidence the beat tracker never had.

This also sets the UX: unlike Shazam's few seconds, this needs 30–120 seconds,
and should present a live-updating ranked list rather than a single answer.

### Melody rerank

Within the 28-tune blues family — all on identical changes, where the harmonic
stage is at chance by construction — extracting the head from rendered audio with
pYIN and reranking: **10/12 top-1, 11/12 top-3**.

### Cost

Index build 2.2s for 1382 tunes; a query is 0.02–0.35s. The beat-synchronous
features are ~100 bytes per bar, small enough to stream to a server continuously
— which also means the audio itself never has to leave the device.

---

## 6. Data

**Chord corpus — solved.** [mikeoliphant/JazzStandards](https://github.com/mikeoliphant/JazzStandards),
1382 tunes as JSON, derived from iReal Pro. Fetched by `scripts/fetch_data.py`
rather than vendored, since it carries no license of its own. Also relevant:
`pyRealParser`, `ireal-musicxml`, the Weimar Jazz Database, and
[JAAH](https://mtg.github.io/JAAH/) (113 audio-aligned jazz tracks — the only
real labeled-audio evaluation set available).

**Melody corpus — the genuine gap.** iReal-derived sources contain *no melodies*.
This repository therefore generates synthetic heads (`heads.py`) to measure the
rerank machinery; they are explicitly not transcriptions. A production system
needs either licensed lead sheets or melodies transcribed from canonical
recordings.

**Legal position, stated once.** Chord progressions are broadly held
unprotectable; melodies are squarely copyrighted, and the iReal community
playlists are compilations. Fine for a prototype. Before shipping: get counsel,
and store melodies only as derived interval sequences, never as reconstructable
lead sheets.

---

## 7. Known failure modes

| case | why it breaks | mitigation |
|---|---|---|
| Solo piano, rubato ballads | No steady pulse, so the beat grid collapses and everything downstream with it | Fall back to non-beat-synchronous chroma DTW |
| Beat-tracker octave error | 240bpm read as 120 halves the beats per chorus and nothing aligns | Already handled: several tempo hypotheses are scored and the matcher picks, since only it can check whether a form actually aligns |
| Piano-less trio | No comping; harmony implied by bass alone | The separate bass chroma band |
| Free or outside solos | Harmony departs from the chart | Per-chorus confidence weighting |
| Heavy reharmonisation (Coltrane changes) | Chart no longer describes what is played | Substitution costs that treat a tritone sub, or ii-V for V, as cheap |
| Contrafacts | Harmony genuinely cannot separate them | Return the family; rerank on melody |
| Modal tunes | Few chords over long spans | Distinctive to retrieve; So What vs Impressions needs melody |
| Tempo drift over long takes | The modulo fold assumes a stable period | Window the query into a few choruses and accumulate scores per window |

---

## 8. What would come next

1. **Real audio evaluation.** Everything above is synthetic. Expect a sharp drop
   on club recordings; that is where the front end gets tuned, and where JAAH
   plus hand-labeled live sets earn their keep.
2. **A pretrained chord model** in place of templates, if measurement shows the
   front end rather than the matcher is the bottleneck.
3. **Streaming.** The offline matcher is already incremental in structure; the
   work is causal beat tracking and windowed accumulation.
4. **Substitution-aware alignment.** The theory-aware DP described in §3.3 is
   specified but not yet implemented — the current matcher is pure correlation.
5. **A melody index worth having**, which is a licensing question before it is
   an engineering one.

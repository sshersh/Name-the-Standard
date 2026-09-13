# Name-the-Standard

Name the jazz standard being played in a live performance.

Not Shazam. Shazam fingerprints a *specific recording*, so it has essentially
zero recall on a band playing the same tune in a different key, at a different
tempo, mostly improvised. What a live performance does give you is the thing the
musicians agreed on — **the form and the changes, repeating every chorus for the
whole tune** — and the head, stated at the top and the end.

So this identifies tunes by their harmony, and separates tunes that share
harmony by their melody.

    audio -> beat-synchronous chroma -> chorus period -> tune x key x position
          -> contrafact family -> melody of the head breaks the tie

**[docs/ALGORITHM.md](docs/ALGORITHM.md) is the design**, including where machine
learning is and is not required. Short version: no ML in the recognition core,
only off-the-shelf pretrained components in the front end, and nothing to train
for v1.

## Results

Synthesised performances, random key / tempo / entry point, against all 1382
tunes, full pipeline including beat tracking. Confidence climbs the longer it
listens, which is the whole point:

| beats heard | ~bars | family top-1 | tune top-1 | key correct |
|---|---|---|---|---|
| 32 | 8 | 3.3% | 3.3% | 8.3% |
| 128 | 32 | 53.3% | 53.3% | 58.3% |
| 256 | 64 | 80.0% | 78.3% | 83.3% |
| 512 | 128 | **95.0%** | 93.3% | 96.7% |

One chorus is not enough; four choruses is. Substituting the true beat grid
raises the last row to 98.3%, so beat tracking — the one stage leaning on a
pretrained model — costs about three points and is not the bottleneck.

Twenty-eight tunes in the corpus are written on the same twelve bars, so
harmony's honest answer there is *"a blues in F"*. Reranking those on the melody
of the head: 10/12 top-1.

## Quick start

    pip install -e ".[audio,dev]"
    python scripts/fetch_data.py          # chord corpus, not vendored (see below)

    namestd demo "Autumn Leaves" --key 3  # synthesise a performance, identify it
    namestd identify recording.wav --progress
    python -m namestd.eval synthetic --songs 60
    pytest

## Trying it on a real performance

Everything measured below is synthetic. To point it at an actual recording you
have locally:

    namestd identify take.wav --progress

`--progress` prints the answer firming up as more of the take is heard, which is
the behaviour worth watching. Any format librosa can open works; `ffmpeg` covers
the rest.

Two things decide whether it has a chance. Give it **four choruses** - one is
near-useless by design, since evidence accumulates across choruses. And the tune
has to be **in the corpus**: 1382 standards, so a modal original or a Coltrane
line written over reharmonised changes will not be there to find.

## Browser demo

A single self-contained page that runs the harmonic matcher client-side over all
1382 tunes: play into the microphone, or let it synthesise a test tune and
identify that. It opens showing a real match so you can see what it does before
granting mic access.

    python scripts/build_web_app.py       # -> web/name-the-standard.html

`web/app.html` is the source; the built page and its inlined `corpus.js` are
generated and not committed, since they embed the fetched corpus.

The browser port covers stages 1-3 (chroma, form, harmonic match). It asks for
the tempo rather than tracking beats - musicians count tunes off anyway - and the
melody rerank stays in the Python prototype.

## Layout

| | |
|---|---|
| `namestd/chords.py` | chord symbols, quality classes, chroma templates |
| `namestd/corpus.py` | charts (sections, repeats, endings) -> chord per beat |
| `namestd/families.py` | contrafact clustering |
| `namestd/form.py` | chorus period from self-similarity |
| `namestd/frontend.py` | beat tracking, CQT, bass+treble chroma |
| `namestd/match.py` | tune x key x position as one circular cross-correlation |
| `namestd/melody.py` | head extraction and Smith-Waterman rerank |
| `namestd/synth.py` | render a chart to audio, for ground truth |
| `namestd/eval.py` | accuracy against listening time |

## Data and licensing

The chord corpus ([mikeoliphant/JazzStandards](https://github.com/mikeoliphant/JazzStandards),
derived from iReal Pro) is **fetched at setup, not vendored**, because it carries
no license of its own.

There is no melody corpus — iReal-derived sources have none, and jazz heads are
copyrighted. `heads.py` generates synthetic heads to measure the rerank
machinery; they are not transcriptions. A production system needs licensed lead
sheets or melodies transcribed from canonical recordings, and should store only
derived interval sequences, which cannot be sung back.

Chord progressions are broadly held unprotectable; melodies are not. Fine for a
prototype — get counsel before shipping.

# Real One

**On the hook, the world freezes and only you keep moving.**

Film your song on your phone, once. Real One turns the take into a music video: you stay real, pixel for pixel, and
only the world around you is made, one world for each section of the song, every cut on a downbeat.

<p>
<img src="docs/media/hook_freeze.gif" width="270" align="top" alt="Hook Freeze: rain falls, then hangs mid-air while the camera keeps moving and the figure keeps dancing">
<img src="docs/media/receipt.jpg" width="405" align="top" alt="Real Receipt: the same frame finished, and with every generated pixel grey">
</p>

*Test render: a drawn figure on code-drawn worlds, with no Meta model in the loop yet. Both images are regenerated from
the first keyed render of a real take (`scripts/readme_media.py`).*

[How the director works](docs/director.md) · [Implementation notes](docs/implementation-notes.md) ·
[Watch page](web/) · [Build status](docs/status.md) · [Changelog](CHANGELOG.md)

Built for the Meta Global AI Developer Hackathon (Cerebral Valley), Oct 27 to Nov 4, 2026. Everything in this repo
was written during the event window.

## Why

Independent artists rarely have a video budget. AI video tools cut the cost, but they tend to generate the artist's
face or illustrate the lyrics word for word. Real One never generates the artist. Every pixel of you comes from your own
take, and the Real Receipt proves it frame by frame: switch it on and everything the models made turns grey while
you stay in colour.

## How it uses Meta's models

| Model | Its job in Real One |
|---|---|
| **SAM 3.1** | Cuts the artist out of every frame of the take (masks streamed over the Responses API). The same masks are the ground truth for the Real Receipt counter. |
| **Muse Wonder** | Hears the song as audio, reads the lyrics and the beat grid, and writes a treatment per section from what the song means, not what the words say. Then watches its own render as video and gives notes for one revision pass. |
| **Muse Image** | Builds one vertical world per section from the director's world prompt. |
| **Muse Voice Transcribe** | Lyrics from the master, placed by section; the artist's own lyric sheet outranks it. |
| **Muse Spark Open** (Baseten) | The open-weight brain: writes the same song's treatment through the same guard, so the two directors can be compared side by side (`plan --brain spark-open`). |

Every path is wired and tested offline against the documented API shapes. None has run with a key yet; the first
keyed run is `scripts/first_keyed_run.sh probe`.

## What it does

- **Beat grid and sections** from the master; the hook is the loudest repeated section.
- **Sync** of the phone's own mic to the master, refusing a weak or ambiguous match instead of guessing.
- **Cuts on downbeats** with wide, medium and close reframes; close-ups come from the phone's native 4K frame; iPhone
  HDR clips are tone-mapped and turned upright.
- **Hook Freeze**: on the hook's downbeat, rain at three depths hangs mid-air, the camera keeps sliding through it, the
  world drains cool and the artist keeps performing in full colour. The same world falls, hangs and resumes.
- **Real Receipt**: the filmed share of every frame, measured from the masks; `--receipt-reveal` flips a section to
  the receipt look and `--receipt-counter` puts the number on screen.
- **Subtext Cut**: the director's rule, enforced in code by a literal-lyrics guard that rejects any world that just
  draws its own line.

## Engineering

- 216 tests. Each guard was also broken on purpose to prove its test goes red, then restored byte-identical; the
  log is in `docs/implementation-notes.md` and the CHANGELOG.
- Every render reads its own output back: cuts within 1 frame of the plan, audio within 1 frame of the master, and the
  freeze proven by hashing the world state. A failed check exits non-zero.
- Every paid call is quoted in US$ and CA$ first and capped (`--max-usd`). The take is uploaded once per run and
  expires on Meta's side after a day.

## Run it

```
.venv/bin/pip install -r requirements.txt
export MODEL_API_KEY=...   # lives outside the repo; without it only the TEST_LOCAL path runs
.venv/bin/python -m realone render --clip footage/take1.mov --song footage/master.wav --section hook \
    --look collage --world "rain-soaked Toronto alley at night" --world "rooftop at dusk" --world "empty stage"

# the whole song, a world list per section, Hook Freeze on every hook, the last hook revealed as a receipt
.venv/bin/python -m realone render --clip footage/take1.mov --song footage/master.wav --section all \
    --look collage,receipt --worlds-json worlds.json --freeze hook --receipt-reveal hook:last --receipt-counter
```

Flags: `--section hook|verse|hook:2|bars:A-B|#i|all`, `--look clean|collage|receipt|all` or a comma list,
`--cut-bars 1|2|4|8`, `--mask sam|local`, `--freeze [SECTION]`, `--freeze-bars N`, `--freeze-style bullet|still`,
`--worlds-json FILE`, `--section-world KEY=PROMPT`, `--receipt-reveal SECTION`, `--receipt-counter`,
`--sam-prompt PHRASE` (repeat for a union), `--workers N`, `--final`, `--sync-near <s>`, `--max-usd`, `--max-minutes`.

- **First keyed runs:** `scripts/first_keyed_run.sh probe` runs SAM 3.1 and Muse Image on the synthetic take, capped
  at US$0.50. `scripts/first_keyed_run.sh real CLIP SONG` renders one hook, capped at US$1. The script loads the key
  from a local key file outside the repo and never prints it.
- **Watch page:** `scripts/build_watch_page.py --tag TAG --section SECTION` builds `web/dist/` from a render made with
  `--look collage,receipt`; `scripts/serve_watch_page.py` serves it locally. Test renders carry a TEST banner.
- **README media:** `scripts/readme_media.py --tag TAG --section SECTION` regenerates the two images above. Test
  renders are stamped TEST RENDER.
- **Director:** `python -m realone.director plan|direct|review`; `plan --brain spark-open` writes the treatment on
  Muse Spark Open (Baseten) instead of Muse Wonder; see `docs/director.md`.

Every render writes `out/<tag>_<section>_<look>.mp4`, its contact sheet, `_shots.json`, `_run.json` and
`_receipt.json`. Keyless test runs and the full pipeline: `docs/implementation-notes.md`.

## Rules this repo follows

- Footage and music are the artist's own.
- No tools built before the event are part of this project.
- API keys live outside the repo (`MODEL_API_KEY` in the environment).

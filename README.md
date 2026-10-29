# Real One

Film your song on your phone. It comes back as a music video where you are real and the world is made for the song. On the hook, the world freezes and only you keep moving.

Built for the Meta Global AI Developer Hackathon (Cerebral Valley), Oct 27 to Nov 4, 2026. Everything in this repo was written during the event window.

## How it works

1. **SAM 3.1** cuts the artist out of every frame of the phone video, real pixels, no generated face.
2. **Muse Wonder** reads the lyrics, the audio and the footage and writes a treatment from what the song means, not what the words say.
3. **Muse Image** builds a world for each section of the song, with depth layers.
4. The cut lands on the beat. On the hook, the world stops and the artist keeps moving (Hook Freeze).
5. **Real Receipt**: one tap greys out every generated pixel and keeps the filmed artist in colour.

## Status

- [~] Freeze spike: script written and plumbing-tested locally (0.0.2); waiting on the Meta API key for the real edge test
- [~] Floor (0.1.0): one song section end to end, three looks (clean, collage, receipt), cuts on downbeats, master
  audio in sync. Proven keyless on a synthetic song and a drawn test figure; the SAM 3.1 and Muse Image path is
  wired but unproven until the key lands
- [ ] Subtext Cut
- [ ] Full song + Real Receipt
- [ ] Hook Freeze on the hook
- [ ] 3-minute demo video

## Run it

```
.venv/bin/pip install -r requirements.txt
export MODEL_API_KEY=...   # lives outside the repo; without it only the TEST_LOCAL path runs
.venv/bin/python -m realone render --clip footage/take1.mov --song footage/master.wav --section hook \
    --look collage --world "rain-soaked Toronto alley at night" --world "rooftop at dusk" --world "empty stage"
```

Flags: `--section hook|verse|hook:2|bars:A-B`, `--look clean|collage|receipt|all`, `--cut-bars 1|2|4|8`,
`--mask sam|local`, `--final` (x264 slow, crf 17), `--sync-near <s>`, `--max-usd`, `--max-minutes`.
Keyless test run, tests and the full pipeline: `docs/implementation-notes.md`.

## Rules this repo follows

- Footage and music are the artist's own.
- No tools built before the event are part of this project.
- API keys live outside the repo (`MODEL_API_KEY` in the environment).

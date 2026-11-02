# Real One: build status

Moved out of the README on 2026-10-29 so the README reads as the product. Updated by hand; the CHANGELOG is the
record of what shipped.

- [~] Freeze spike: script written and plumbing-tested locally (0.0.2); waiting on the Meta API key for the real edge test
- [~] Floor (0.1.0): one song section end to end, three looks (clean, collage, receipt), cuts on downbeats, master
  audio in sync. Proven keyless on a synthetic song and a drawn test figure; the SAM 3.1 and Muse Image path is
  wired but unproven until the key lands
- [~] Hook Freeze in the renderer (0.2.0): `--freeze`; the world stops on the hook's first downbeat while the artist
  keeps moving, proven on every render by hashing the world layer (keyless)
- [~] Full song + Real Receipt (0.2.0): `--section all` with a world list per section, parallel render units joined
  frame-exactly with the master; every render writes the filmed vs generated share per frame; `--receipt-reveal`
  and `--receipt-counter`. The synthetic full song (88 s) renders in about 6 to 7 minutes on a loaded Mac
- [~] Phone footage (0.2.0): native-resolution punch-ins from a 4K take, HDR iPhone clips (HLG/PQ) tone-mapped and
  turned upright, collage border that turns dark on a light world
- [~] Director (0.2.0, `python -m realone.director plan|direct|review`): lyrics, a Muse Wonder treatment per
  section and a tool loop that renders and watches the result; dry run proven on fixtures, see `docs/director.md`
- [~] Subtext Cut: the director's rule (direct from what the song means, not what it says) and the literal-lyrics
  guard are built and tested on fixtures; the first keyed treatment on a real song is next
- [x] Tracked stage (Nov 1, `--stage tracked`, docs/stage.md): floor track, camera solve (tripod takes too), worlds in
  depth on Core ML, filmed through the take's camera; blind critic 2/10 -> 7/10 against real footage (lab plate)
- [x] Camera moves on the beat (Nov 1, `--move`, `--moves-from`, `--from-treatment`): dolly zoom, crash zoom, world
  swap; the director places them per section
- [x] First keyed Muse Wonder treatment (Nov 1, test song): US$0.15, audio in, moves placed
- [x] The director's loop keyed end to end (Nov 1, stand-in clip): generate, render through the stage with SAM 3.1,
  watch, revise, re-render, watch again (run 2: US$0.28, 19 calls; it judged the dolly "reads clearly" and turned
  the dust overlay off itself). Its first run exposed what the watch was not told (the move, the overlay): fixed
- [x] Live watch page (Nov 1, real-one-watch.vercel.app): the stand-in render, the Real Receipt, Wonder's verbatim
  review of its own cut, credits counted from evidence, stand-ins named on the banner and the footer
- [~] Muse Glimmer 30B treatment (open weights, self-hosted on a rented GPU): wired and tested offline; not run (the
  RunPod balance is under US$1)
- [x] SAM 3.1 decode ~100x faster (Nov 1, compiled codec, byte-identical)
- [ ] The artist's real take
- [ ] 3-minute demo video (the edit kit is in the application folder)

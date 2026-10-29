"""python -m realone {render, analyze, sync} ..."""
import argparse
import sys

from realone import __version__
from realone.common import LOCAL_PREFIX, OUT, Refuse, write_json


def build_parser():
    p = argparse.ArgumentParser(prog="realone", description="Real One: a phone performance becomes a music video.")
    p.add_argument("--version", action="version", version=__version__)
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("render", help="render one section of the song as a finished 9:16 clip")
    r.add_argument("--clip", required=True, help="the artist's own phone take (its mic audio is used only to sync)")
    r.add_argument("--song", required=True, help="the master track (this is the audio in the output)")
    r.add_argument("--section", default="hook",
                   help="hook | verse | intro | outro | bridge, ':n' for the n-th (hook:2), bars:A-B, or #index")
    r.add_argument("--look", default="collage", help="clean | collage | receipt | all | a comma list")
    r.add_argument("--mask", choices=["sam", "local"], default="sam",
                   help="sam = Meta SAM 3.1 (default, needs MODEL_API_KEY); local = Apple Vision, TEST ONLY")
    r.add_argument("--local-plate", action="store_true", help="three procedural test plates (TEST ONLY, needs --mask local)")
    r.add_argument("--world", action="append", help="a Muse Image world prompt; repeat for one world per shot")
    r.add_argument("--worlds-file", help="one Muse Image prompt per line")
    r.add_argument("--world-image", action="append", help="use your own image as a world (repeatable)")
    r.add_argument("--no-world-wrap", action="store_true", help="send --world prompts without the plate instructions")
    r.add_argument("--cut-bars", type=int, default=2, help="cut every N bars (1, 2, 4 or 8)")
    r.add_argument("--beats-per-bar", type=int, help="override the detected meter")
    r.add_argument("--sync-near", type=float, help="song time (s) the take starts near, when sync is ambiguous")
    r.add_argument("--tag", default="floor", help="output name stem (out/<tag>_<section>_<look>.mp4)")
    r.add_argument("--final", action="store_true", help="x264 slow / crf 17 (default veryfast / crf 20)")
    r.add_argument("--particles", choices=["dust", "rain", "embers", "none"], default="dust",
                   help="particle layer for --world / --world-image worlds")
    r.add_argument("--push", type=float, default=0.04, help="world push-in over each shot")
    r.add_argument("--erode", type=int, default=2, help="mask erosion px")
    r.add_argument("--feather", type=float, default=1.4, help="mask feather sigma px")
    r.add_argument("--wrap", type=float, default=0.6, help="light wrap strength (clean)")
    r.add_argument("--match", type=float, default=0.35, help="LAB colour match strength")
    r.add_argument("--grain", type=float, default=5.0, help="film grain strength")
    r.add_argument("--no-despill", dest="despill", action="store_false", help="turn the edge despill off (A/B only)")
    r.add_argument("--max-usd", type=float, default=1.00, help="refuse if the API spend quote is over this")
    r.add_argument("--max-minutes", type=float, default=20.0, help="stop a render that runs longer than this")
    r.add_argument("--max-seconds", type=float, default=40.0, help="longest section one run will render")
    r.add_argument("--freeze", nargs="?", action="append", const="section", metavar="SECTION",
                   help="Hook Freeze: on the section's first downbeat the world (plate, particles, camera) stops for "
                        "--freeze-bars while the artist keeps moving. Bare --freeze = the rendered section; or name "
                        "one: hook (every hook in range), hook:2, #3, bars:A-B. Repeatable.")
    r.add_argument("--freeze-bars", type=int, default=2, metavar="N", help="length of each freeze in bars (default 2)")
    r.add_argument("--no-native", action="store_true",
                   help="A/B only: punch-ins upscale the 1080 proxy (the 0.1.0 path) even when the take is bigger")

    an = sub.add_parser("analyze", help="beat grid, downbeats and sections of the master -> timeline json")
    an.add_argument("--song", required=True)
    an.add_argument("--beats-per-bar", type=int)
    an.add_argument("--tag", default="floor")
    an.add_argument("--local", action="store_true", help="prefix outputs TEST_LOCAL_")

    sy = sub.add_parser("sync", help="offset of a phone take against the master (refuses when unsure)")
    sy.add_argument("--clip", required=True)
    sy.add_argument("--song", required=True)
    sy.add_argument("--sync-near", type=float)
    return p


def main(argv=None):
    a = build_parser().parse_args(argv)
    try:
        if a.cmd == "render":
            from realone.render import render
            render(a)
        elif a.cmd == "analyze":
            from realone.audio import analyze
            tl = analyze(a.song, a.beats_per_bar)
            OUT.mkdir(exist_ok=True)
            path = write_json(OUT / f"{LOCAL_PREFIX if a.local else ''}{a.tag}_timeline.json", tl)
            print(f"{tl['bpm']:.2f} BPM ({tl['grid']}), {tl['beats_per_bar']} beats/bar, first downbeat "
                  f"{tl['first_downbeat_s']:.3f}s -> {path}")
            for s in tl["sections"]:
                print(f"  #{s['index']} {s['role']:6s} {s['label']} bars {s['start_bar']:>3}-{s['end_bar']:<3} "
                      f"{s['start_s']:7.3f}-{s['end_s']:7.3f}s  {s['energy_db']:6.1f} dB{'  HOOK' if s['is_hook'] else ''}")
        elif a.cmd == "sync":
            from realone.sync import sync_clip
            r = sync_clip(a.song, a.clip, a.sync_near)
            print(f"offset {r['offset_ms']:+.1f} ms (song_time = clip_time + offset)  r={r['r_peak']:.3f}  "
                  f"margin={r['margin']:.3f}  resolved by {r['resolved_by']}")
    except Refuse as e:
        print(f"realone: REFUSED: {e}", file=sys.stderr)
        sys.exit(e.exit_code)


if __name__ == "__main__":
    main()

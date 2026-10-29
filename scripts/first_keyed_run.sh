#!/usr/bin/env bash
# The first runs with a Meta Model API key, in two stages. Every render quotes its spend in US$ and CA$ before any
# call and refuses over --max-usd, so neither stage can spend more than its cap.
#
#   scripts/first_keyed_run.sh probe                  stage 1: key only. SAM 3.1 + Muse Image on the SYNTHETIC
#                                                     test take (a drawn figure, no real person), cap US$0.50.
#                                                     Settles the API shapes docs/implementation-notes.md marks
#                                                     UNCONFIRMED: streamed masks, newlines, size="1080x1920".
#   scripts/first_keyed_run.sh real CLIP SONG         stage 2: the artist's own take + a song he owns outright.
#                                                     One hook, collage, Hook Freeze, receipt counter, cap US$1.
#
# The key is read from ~/.secrets/meta-model-api (chmod 600) and never printed. If the file is missing and a
# terminal is attached, the key is asked for without echo and saved there.
set -euo pipefail
# resolve the caller's paths (symlinks too) before moving to the repo root
ARGS=()
for a in "$@"; do
  if [[ -e "$a" ]]; then ARGS+=("$(realpath "$a")"); else ARGS+=("$a"); fi
done
set -- "${ARGS[@]+"${ARGS[@]}"}"
cd "$(dirname "$(realpath "$0")")/.."

KEYFILE="$HOME/.secrets/meta-model-api"
PY=.venv/bin/python

die() { echo "refused: $*" >&2; exit 2; }

load_key() {
  if [[ ! -s "$KEYFILE" ]]; then
    [[ -t 0 ]] || die "no key at $KEYFILE and no terminal to ask on; paste the key into the Claude chat instead"
    printf 'Meta Model API key (input hidden): ' >&2
    IFS= read -rs key; echo >&2
    [[ -n "$key" ]] || die "empty key"
    (umask 077; mkdir -p "$(dirname "$KEYFILE")"; printf '%s' "$key" > "$KEYFILE")
    chmod 600 "$KEYFILE"
    unset key
    echo "key saved to $KEYFILE (600)" >&2
  fi
  [[ "$(stat -f '%Lp' "$KEYFILE")" == "600" ]] || die "$KEYFILE must be chmod 600"
  MODEL_API_KEY="$(cat "$KEYFILE")"
  export MODEL_API_KEY
}

# The gym clip has bystanders in the mirror: it never leaves this Mac, keyed or not.
guard_clip() {
  [[ -f "$1" ]] || die "clip not found: $1"
  case "$(realpath "$1")" in
    */footage/own/*|*IMG_0849*) die "$1 is the gym clip (bystanders in the mirror); it is never sent to any API" ;;
  esac
}

case "${1:-}" in
  probe)
    load_key
    "$PY" -m realone render --clip footage/perf_test.mp4 --song footage/test_song.wav --section hook \
      --mask sam --look collage --tag KEYED_PROBE \
      --world "a neon alley at night, wet asphalt, magenta and cyan tubes" \
      --freeze --max-usd 0.50
    echo "stage 1 done: out/KEYED_PROBE_hook_collage.mp4 + contact sheet; masks and world cached under out/"
    ;;
  real)
    [[ $# -eq 3 ]] || die "usage: $0 real CLIP SONG"
    guard_clip "$2"
    [[ -f "$3" ]] || die "song not found: $3"
    load_key
    "$PY" -m realone render --clip "$2" --song "$3" --section hook \
      --mask sam --look collage --tag REAL_FIRST \
      --world "a rooftop in Toronto at dusk, the CN Tower far off, wind" \
      --world "an empty subway platform at 2am, fluorescent light, tiles" \
      --world "a neon alley at night, wet asphalt, magenta and cyan tubes" \
      --freeze hook --receipt-counter --max-usd 1.00
    echo "stage 2 done: out/REAL_FIRST_hook_collage.mp4 + contact sheet + receipt json"
    ;;
  *)
    sed -n 2,14p "$0"; exit 64 ;;
esac

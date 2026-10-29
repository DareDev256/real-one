"""Meta Model API: client, model ids, prices. Read from dev.meta.ai/docs on 2026-10-28 (see
docs/implementation-notes.md for what is CONFIRMED and what is not). No key in this repo, ever: MODEL_API_KEY
comes from the environment and is never printed."""
import os

from realone.common import Refuse

BASE_URL = "https://api.meta.ai/v1"
SAM_MODEL = "sam-3.1"
IMAGE_MODEL = "muse-image-1.0"
SAM_USD_PER_1K_FRAMES = 0.20
SAM_MIN_FRAMES = 50
IMAGE_USD = 0.01
USD_TO_CAD = 1.40  # rough, for the pre-spend quote only


def api_key():
    return os.environ.get("MODEL_API_KEY") or None


def require_key(why):
    key = api_key()
    if not key:
        raise Refuse(f"MODEL_API_KEY is not set, and {why} needs the Meta Model API. Nothing was called. "
                     "Keyless test run: --mask local --local-plate (outputs are prefixed TEST_LOCAL_).")
    return key


def make_client(key):
    from openai import OpenAI
    return OpenAI(base_url=BASE_URL, api_key=key, timeout=1800.0)

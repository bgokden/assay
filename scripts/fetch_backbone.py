"""Download a backbone, retrying: a stalled CDN connection should not waste a GPU slot.

    uv run python scripts/fetch_backbone.py Qwen/Qwen3-8B-Base

huggingface_hub resumes partial files, so a retry continues rather than restarting. The part
that hangs is the connection itself, so each attempt gets its own timeout.
"""

import os
import sys
import time

os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "30")

import httpx
from huggingface_hub import snapshot_download
from huggingface_hub.errors import HfHubHTTPError

ATTEMPTS = 20
PATTERNS = ["*.safetensors", "*.json", "*.txt", "*.model"]
RETRIABLE = (HfHubHTTPError, httpx.HTTPError, OSError, TimeoutError)


def main() -> int:
    repo = sys.argv[1]
    for attempt in range(1, ATTEMPTS + 1):
        try:
            path = snapshot_download(repo, allow_patterns=PATTERNS, max_workers=4)
        except RETRIABLE as error:  # a stall, a reset, a 5xx: all worth retrying
            print(f"attempt {attempt} failed: {type(error).__name__}: {error}", flush=True)
            time.sleep(min(60, 5 * attempt))
            continue
        print(f"downloaded to {path}", flush=True)
        return 0
    print(f"giving up on {repo} after {ATTEMPTS} attempts", flush=True)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

"""Record the /chat page as a video, by driving the live page and capturing frames.

    uv run python -m assay.server --model Berk/assay-4b --agents examples/agents --port 8080
    python scripts/record_chat.py --out tmp/post

Needs playwright and ffmpeg, neither of which this project depends on, so run it with a
python that has playwright installed rather than the project environment.

Three things it does that matter for the numbers being honest. It warms the model first, so
the first turn is a steady-state latency and not a cold start. It stops capturing frames
while a request is in flight, because screenshotting against the page inflates the latency
the page measures -- around 80 ms against a true 35. And it writes the latencies it observed
to latencies.txt, so whatever quotes the video can quote the take that was actually made.
"""

import argparse
import json
import math
import pathlib
import shutil
import subprocess
import time

from playwright.sync_api import sync_playwright

DEFAULT_URL = "http://127.0.0.1:8080/chat"
DEFAULT_OUT = "tmp/post"
WIDTH, HEIGHT = 1080, 1350
FPS = 10
HOLD_LAST = 2.4  # on the final decision, before moving
HOLD_SCROLL = 1.2  # easing back to the top
HOLD_END = 3.6  # the read-it hold, and where the poster frame comes from
TYPED_FRAMES = 20  # frames spent typing each line, so it is visible at any playback rate
SCRIPT = [
    "I was charged twice for order A-104, that is 240 euros.",
    "Actually it is the whole annual contract, 2,400 euros.",
    "Can you confirm someone will look at it today?",
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default=DEFAULT_URL, help="where the /chat page is served")
    parser.add_argument("--out", default=DEFAULT_OUT, help="directory for the video and stills")
    args = parser.parse_args()

    global URL, OUT, FRAMES
    URL = args.url
    OUT = pathlib.Path(args.out)
    OUT.mkdir(parents=True, exist_ok=True)
    FRAMES = OUT / "frames"

    if FRAMES.exists():
        shutil.rmtree(FRAMES)
    FRAMES.mkdir(parents=True)
    frame = 0

    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": WIDTH, "height": HEIGHT})
        page.goto(URL)
        page.wait_for_function("document.getElementById('conditions').children.length > 0")

        # a cold first request pays for the CUDA graphs; that is not the number to show
        page.evaluate("(async () => { await turn('warming up'); reset(); })()")
        page.wait_for_function("document.querySelectorAll('.turn').length === 0", timeout=120000)
        time.sleep(0.5)

        def capture(seconds: float) -> None:
            nonlocal frame
            deadline = time.time() + seconds
            while time.time() < deadline:
                page.screenshot(path=str(FRAMES / f"frame_{frame:04d}.png"))
                frame += 1
                time.sleep(max(0.0, 1.0 / FPS - 0.06))

        def type_line(line: str) -> None:
            """Advance the composer one frame at a time.

            The page can type itself, but a browser being screenshotted does not honour small
            setTimeout delays -- 40 characters meant to take 1.4s arrived in 0.4s, so the
            typing crossed two or three frames and read as the text simply appearing. Driving
            it from here makes the pace exact: TYPED_FRAMES frames, whatever the line length.
            """
            nonlocal frame
            step = math.ceil(len(line) / TYPED_FRAMES)
            for end in range(step, len(line) + step, step):
                page.evaluate(f"setComposer({json.dumps(line[:end])})")
                page.screenshot(path=str(FRAMES / f"frame_{frame:04d}.png"))
                frame += 1

        capture(1.2)  # the conditions, before anything is said
        for line in SCRIPT:
            type_line(line)
            capture(0.6)  # a beat with the line sitting there, before it is sent
            page.evaluate(f"commit({json.dumps(line)})")
            capture(0.4)
            # the decision itself is faster than one frame, and screenshotting while the
            # request is in flight inflates the latency the page reports -- so do not, and
            # let the compositor go idle first or the fetch resolves behind a screenshot
            time.sleep(0.4)
            page.evaluate(f"decideTurn({json.dumps(line)})")
            capture(1.7)
        # Let it sit on the last decision, then ease back to the top and hold there. Autoplay
        # has no scrubber, so the hold is the only chance anyone gets to read the numbers.
        page.evaluate("document.getElementById('message').blur()")
        capture(HOLD_LAST)
        page.evaluate("window.scrollTo({top: 0, behavior: 'smooth'})")
        capture(HOLD_SCROLL)
        page.evaluate("window.scrollTo(0, 0)")
        capture(HOLD_END)
        page.screenshot(path=str(OUT / "07-chat.png"), full_page=True)
        # the post quotes these, and they change with every take, so report them
        latencies = page.eval_on_selector_all(
            ".turn.agent .meta span.ms", "els => els.map(e => e.textContent)"
        )
        outcomes = page.eval_on_selector_all(
            ".turn.agent .outcome", "els => els.map(e => e.textContent.trim().split('\\n')[0])"
        )
        (OUT / "latencies.txt").write_text(
            "\n".join(f"{o} -- {ms}" for o, ms in zip(outcomes, latencies)) + "\n"
        )

        # A second, shorter pass for the feed image. Three turns no longer fit 1080x1350 now
        # that each one carries a name and an avatar, and a turn clipped halfway reads as a
        # mistake -- so the still stops at the escalation, which is the beat worth showing.
        page.goto(URL)
        page.wait_for_function("document.getElementById('conditions').children.length > 0")
        for line in SCRIPT[:2]:
            page.evaluate(f"setComposer({json.dumps(line)})")
            page.evaluate(f"commit({json.dumps(line)})")
            time.sleep(0.3)
            page.evaluate(f"decideTurn({json.dumps(line)})")
        page.evaluate("document.getElementById('message').blur()")
        page.evaluate("window.scrollTo(0, 0)")
        time.sleep(0.6)
        page.screenshot(path=str(OUT / "08-chat-linkedin.png"))
        browser.close()

    video = OUT / "chat-demo.mp4"
    bare = OUT / "chat-demo-bare.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-framerate",
            str(FPS),
            "-i",
            str(FRAMES / "frame_%04d.png"),
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(bare),
        ],
        check=True,
        capture_output=True,
    )
    # The cover is the frame a platform shows before anyone presses play. LinkedIn lets you
    # upload one; embedding it as attached_pic covers the players that read it from the file.
    cover = OUT / "chat-cover.png"
    shutil.copy(OUT / "08-chat-linkedin.png", cover)
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(bare),
            "-i",
            str(cover),
            "-map",
            "0",
            "-map",
            "1",
            "-c",
            "copy",
            "-c:v:1",
            "png",
            "-disposition:v:1",
            "attached_pic",
            "-movflags",
            "+faststart",
            str(video),
        ],
        check=True,
        capture_output=True,
    )
    bare.unlink()
    gif = OUT / "chat-demo.gif"
    palette = FRAMES / "palette.png"
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(video),
            "-map",
            "0:v:0",
            "-vf",
            "fps=8,scale=720:-1:flags=lanczos,palettegen",
            str(palette),
        ],
        check=True,
        capture_output=True,
    )
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(video),
            "-i",
            str(palette),
            "-filter_complex",
            "[0:v:0]fps=8,scale=720:-1:flags=lanczos[x];[x][1:v]paletteuse",
            str(gif),
        ],
        check=True,
        capture_output=True,
    )
    print(
        f"{frame} frames -> {video} ({video.stat().st_size // 1024} KB), "
        f"{gif} ({gif.stat().st_size // 1024} KB), cover {cover.name}"
    )
    shutil.rmtree(FRAMES)


if __name__ == "__main__":
    main()

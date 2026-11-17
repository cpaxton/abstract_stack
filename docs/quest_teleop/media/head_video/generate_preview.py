"""Render a four-second synthetic QTP1 round-trip, not a headset recording."""

from pathlib import Path

import numpy as np
from PIL import Image

from abstract_stack.teleop.video import encode_pair, synthetic_pair, unpack_pair


def main():
    destination = Path(__file__).parent
    frames = []
    for pair_id in range(40):
        pair = synthetic_pair(pair_id, width=320, height=240, captured=100.0 + pair_id / 10)
        packet = encode_pair(pair, quality=75).packet("documentation-synthetic", 0)
        metadata, left, right = unpack_pair(packet, decode=True)
        assert metadata["pair_id"] == pair_id
        assert metadata["eye_order"] == ["left", "right"]
        frames.append(Image.fromarray(np.concatenate((left, right), axis=1)))
    frames[0].save(
        destination / "paired-jpeg.gif",
        save_all=True,
        append_images=frames[1:],
        duration=100,
        loop=0,
        optimize=True,
    )
    with Image.open(destination / "paired-jpeg.gif") as animation:
        assert animation.n_frames == 40
        assert animation.size == (640, 240)
    print("40 verified whole-pair frames; four seconds at10 fps playback")


if __name__ == "__main__":
    main()

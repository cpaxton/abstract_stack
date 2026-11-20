"""Delay-based adaptation of the teleop streams to the operator's link.

Socket buffers on the way (the host's send buffer, routers, the browser's network
stack) can hold seconds of data before anything blocks, so a host cannot see
congestion by watching its own sends. The browser can: it measures each pose
frame's one-way delay on a synced clock, and the excess over its recent minimum is
queueing delay anywhere along the path. It reports that a few times a second
(``link_feedback``) and this controller trades stream rate for latency:

* queueing above ``high_ms``: back off, in this order: the H.264 bitrate
  (multiplicatively, down to its floor), the pose and status rate (every 2nd, then
  3rd, tick), then the video frame rate (every 2nd, then 3rd, pair);
* queueing below ``low_ms`` for a while: recover in the reverse order;
* no feedback for a while (older client, JPEG-only): hold the current levels.
"""

from __future__ import annotations

import math
import time
from collections import deque
from dataclasses import dataclass


@dataclass
class LinkConfig:
    high_ms: float = 80.0
    low_ms: float = 25.0
    decrease_every_s: float = 0.5  # one step per this, while congested
    hold_after_decrease_s: float = 2.0
    increase_every_s: float = 1.0
    min_kbps: int = 250
    max_pose_divider: int = 3  # 50 Hz → 25 Hz → ~17 Hz
    max_video_divider: int = 3  # 20 fps → 10 fps → ~7 fps


class LinkController:
    def __init__(self, max_kbps: int, config: LinkConfig | None = None):
        self.config = config or LinkConfig()
        self.max_kbps = max_kbps
        self.kbps = max_kbps
        self.pose_divider = 1
        self.video_divider = 1
        self.queue_ms = 0.0
        self.decreases = 0
        self._last_decrease = self._last_increase = float("-inf")

    def feedback(self, queue_ms: float, now: float | None = None) -> None:
        now = time.monotonic() if now is None else now
        cfg = self.config
        self.queue_ms = float(queue_ms)
        if queue_ms > cfg.high_ms:
            if now - self._last_decrease >= cfg.decrease_every_s:
                self._decrease(now)
        elif queue_ms < cfg.low_ms:
            if (
                now - self._last_decrease >= cfg.hold_after_decrease_s
                and now - self._last_increase >= cfg.increase_every_s
            ):
                self._increase(now)

    def video_congestion(self, now: float | None = None) -> None:
        """An H.264 viewer had to resynchronise (a keyframe): lower only the bitrate.

        Host-side signals cannot tell a slow link from a slow page (late acks), so
        they never touch the pose or frame rates; the browser's stall-filtered delay
        feedback makes those decisions.
        """
        now = time.monotonic() if now is None else now
        if now - self._last_decrease >= self.config.decrease_every_s:
            self._last_decrease = now
            self.decreases += 1
            self.kbps = max(self.config.min_kbps, round(self.kbps * 0.7))

    def _decrease(self, now: float) -> None:
        self._last_decrease = now
        self.decreases += 1
        if self.kbps > self.config.min_kbps:
            self.kbps = max(self.config.min_kbps, round(self.kbps * 0.7))
        elif self.pose_divider < self.config.max_pose_divider:
            self.pose_divider += 1
        elif self.video_divider < self.config.max_video_divider:
            self.video_divider += 1

    def _increase(self, now: float) -> None:
        self._last_increase = now
        if self.video_divider > 1:
            self.video_divider -= 1
        elif self.pose_divider > 1:
            self.pose_divider -= 1
        elif self.kbps < self.max_kbps:
            self.kbps = min(self.max_kbps, round(self.kbps * 1.15) + 1)

    def report(self) -> dict:
        return {
            "queue_ms": round(self.queue_ms, 1),
            "video_kbps": self.kbps,
            "pose_hz_divider": self.pose_divider,
            "video_fps_divider": self.video_divider,
            "decreases": self.decreases,
        }


class InFlightWindow:
    """Bound a stream's unacknowledged frames to about one round trip plus slack.

    The browser acks the highest frame id it has received (cumulative). Frames sent
    but not yet acked are in flight. The window allows ``(rtt + slack) /
    frame_interval`` of them (at least ``min_frames``, at most ``max_window_s``
    worth), where ``rtt`` is the 90th-percentile send→ack time of the last few
    seconds. A percentile, not the minimum, because acks are sent by page script:
    on a slow device they arrive late in bursts, and a minimum-based window would
    starve it. Anything beyond the window would only queue in buffers, so the sender
    skips instead. Rate decisions belong to the browser's delay feedback
    (``LinkController``); the window is the backstop against runaway buffering.
    """

    def __init__(
        self,
        slack_s: float = 0.06,
        min_frames: int = 2,
        max_window_s: float = 1.0,
        rtt_memory_s: float = 5.0,
        lost_after_s: float = 1.5,
    ):
        self.slack_s, self.min_frames, self.rtt_memory_s = slack_s, min_frames, rtt_memory_s
        self.max_window_s = max_window_s
        # A frame unacked this long counts as lost (the client dropped it, or acks
        # were lost with a stalled link): it must not hold the window shut forever.
        self.lost_after_s = lost_after_s
        self._sent: dict[int, float] = {}  # frame id -> send time (insertion-ordered)
        self._rtts: deque[tuple[float, float]] = deque(maxlen=1024)  # (time, rtt)
        self.skipped = 0
        self._acked = False

    @property
    def in_flight(self) -> int:
        return len(self._sent)

    def rtt(self, now: float) -> float | None:
        """90th-percentile round trip over the last ``rtt_memory_s``."""
        while self._rtts and now - self._rtts[0][0] > self.rtt_memory_s:
            self._rtts.popleft()
        if not self._rtts:
            return None
        values = sorted(r for _, r in self._rtts)
        return values[min(len(values) - 1, int(0.9 * len(values)))]

    def allowed(self, now: float, frame_interval_s: float) -> int:
        if not self._acked:
            return 1 << 30  # this client does not ack (older client): no window
        rtt = self.rtt(now)
        if rtt is None:  # acks seen, but none recent enough to estimate from
            return max(self.min_frames, math.ceil(self.max_window_s / frame_interval_s))
        seconds = min(rtt + self.slack_s, self.max_window_s)
        return max(self.min_frames, math.ceil(seconds / frame_interval_s))

    def can_send(self, now: float, frame_interval_s: float) -> bool:
        for key in [k for k, t in self._sent.items() if now - t > self.lost_after_s]:
            del self._sent[key]
        if len(self._sent) < self.allowed(now, frame_interval_s):
            return True
        self.skipped += 1
        return False

    def sent(self, frame_id: int, now: float) -> None:
        self._sent[frame_id] = now

    def ack(self, frame_id: int, now: float) -> None:
        self._acked = True
        sent_at = self._sent.get(frame_id)
        if sent_at is not None:
            self._rtts.append((now, now - sent_at))
        for key in [k for k in self._sent if k <= frame_id]:
            del self._sent[key]

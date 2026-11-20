import type { ClockSync } from "./clock";
import { H264StereoDecoder, isH264Message, parseH264, type DecodedPair } from "./h264";

// Stereo head-camera pairs in the Quest v1 "QTP1" framing:
// ASCII "QTP1", uint32 BE lengths (metadata, left, right), JSON, JPEG, JPEG.

export interface PairMetadata {
  pair_id: number;
  calibration_generation: number;
  capture_host_monotonic_s: number;
  sim_time_s: number;
  codec: "jpeg" | "h264";
  width: number;
  height: number;
  eye_order: ["left", "right"];
  cameras: [string, string];
  vertical_fov_deg: [number, number];
}

export interface StereoPair {
  meta: PairMetadata;
  left: ImageBitmap;
  right: ImageBitmap;
  /** Local time the pair's message arrived, and when both eyes finished decoding. */
  arrivedAt: number;
  receivedAt: number;
}

/** Metadata of a QTP1 (JPEG) or QTH1 (H.264) message, for acks; null if malformed. */
export function pairMetaOf(buffer: ArrayBuffer): PairMetadata | null {
  try {
    return isH264Message(buffer) ? parseH264(buffer).meta : parsePair(buffer).meta;
  } catch {
    return null;
  }
}

/** Capture age (ms) of `pair` on the host clock when synced, else time since it arrived. */
export function videoAgeMs(clock: ClockSync, pair: StereoPair, now: number): number {
  return clock.synced
    ? clock.ageMs(pair.meta.capture_host_monotonic_s, now)
    : now - pair.receivedAt;
}

export interface EncodedPair {
  meta: PairMetadata;
  left: Uint8Array;
  right: Uint8Array;
}

export function parsePair(buffer: ArrayBuffer): EncodedPair {
  const view = new DataView(buffer);
  if (buffer.byteLength < 16) throw new Error("stereo pair too short");
  const magic = String.fromCharCode(...new Uint8Array(buffer, 0, 4));
  if (magic !== "QTP1") throw new Error(`bad stereo pair magic ${magic}`);
  const metaBytes = view.getUint32(4, false);
  const leftBytes = view.getUint32(8, false);
  const rightBytes = view.getUint32(12, false);
  if (16 + metaBytes + leftBytes + rightBytes !== buffer.byteLength) {
    throw new Error("stereo pair length mismatch");
  }
  const meta = JSON.parse(
    new TextDecoder().decode(new Uint8Array(buffer, 16, metaBytes)),
  ) as PairMetadata;
  return {
    meta,
    left: new Uint8Array(buffer, 16 + metaBytes, leftBytes),
    right: new Uint8Array(buffer, 16 + metaBytes + leftBytes, rightBytes),
  };
}

/**
 * Stereo pairs from either codec, by message magic. JPEG ("QTP1") decodes
 * latest-wins: at most one pair at a time and only the newest waiting pair is
 * kept, so a slow device skips frames instead of falling behind. H.264 ("QTH1")
 * goes to WebCodecs (see h264.ts). Either way `latest` holds both eyes of one pair.
 */
export class StereoDecoder {
  latest: StereoPair | null = null;
  decodedPairs = 0;
  /** Codec of the newest message: "jpeg", "h264", or null before any. */
  codec: "jpeg" | "h264" | null = null;
  private waiting: ArrayBuffer | null = null;
  private waitingSince = 0;
  private busy = false;
  private onPair: () => void;
  private h264: H264StereoDecoder;

  constructor(onPair: () => void, requestKeyframe: () => void = () => {}) {
    this.onPair = onPair;
    this.h264 = new H264StereoDecoder((pair) => this.accept(pair), requestKeyframe);
  }

  push(buffer: ArrayBuffer, arrivedAt: number = performance.now()): void {
    if (isH264Message(buffer)) {
      this.codec = "h264";
      try {
        this.h264.push(buffer, arrivedAt);
      } catch (error) {
        console.warn("H.264 message dropped:", error);
      }
      return;
    }
    this.codec = "jpeg";
    this.waiting = buffer;
    this.waitingSince = arrivedAt;
    if (!this.busy) void this.drain();
  }

  clear(): void {
    this.h264.clear();
    this.codec = null;
    this.waiting = null;
    this.latest?.left.close();
    this.latest?.right.close();
    this.latest = null;
  }

  /** Swap in a decoded pair: both eyes together, never a new left with an old right. */
  private accept(pair: DecodedPair): void {
    const previous = this.latest;
    this.latest = { ...pair, receivedAt: performance.now() };
    previous?.left.close();
    previous?.right.close();
    this.decodedPairs += 1;
    this.onPair();
  }

  private async drain(): Promise<void> {
    this.busy = true;
    try {
      while (this.waiting) {
        const buffer = this.waiting;
        const arrivedAt = this.waitingSince;
        this.waiting = null;
        const pair = parsePair(buffer);
        const blob = (bytes: Uint8Array) => new Blob([bytes as BlobPart], { type: "image/jpeg" });
        const [left, right] = await Promise.all([
          createImageBitmap(blob(pair.left)),
          createImageBitmap(blob(pair.right)),
        ]);
        // Swap both eyes together: never show a new left with an old right.
        const previous = this.latest;
        this.latest = {
          meta: pair.meta,
          left,
          right,
          arrivedAt,
          receivedAt: performance.now(),
        };
        previous?.left.close();
        previous?.right.close();
        this.decodedPairs += 1;
        this.onPair();
      }
    } catch (error) {
      console.warn("stereo pair dropped:", error);
    } finally {
      this.busy = false;
    }
  }
}

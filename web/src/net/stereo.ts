// Stereo head-camera pairs in the Quest v1 "QTP1" framing:
// ASCII "QTP1", uint32 BE lengths (metadata, left, right), JSON, JPEG, JPEG.

export interface PairMetadata {
  pair_id: number;
  calibration_generation: number;
  capture_host_monotonic_s: number;
  sim_time_s: number;
  codec: "jpeg";
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
  receivedAt: number;
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
 * Latest-wins JPEG decoding: at most one pair decodes at a time and only the newest
 * waiting pair is kept, so a slow device skips frames instead of falling behind.
 */
export class StereoDecoder {
  latest: StereoPair | null = null;
  decodedPairs = 0;
  private waiting: ArrayBuffer | null = null;
  private busy = false;
  private onPair: () => void;

  constructor(onPair: () => void) {
    this.onPair = onPair;
  }

  push(buffer: ArrayBuffer): void {
    this.waiting = buffer;
    if (!this.busy) void this.drain();
  }

  clear(): void {
    this.waiting = null;
    this.latest?.left.close();
    this.latest?.right.close();
    this.latest = null;
  }

  private async drain(): Promise<void> {
    this.busy = true;
    try {
      while (this.waiting) {
        const buffer = this.waiting;
        this.waiting = null;
        const pair = parsePair(buffer);
        const blob = (bytes: Uint8Array) => new Blob([bytes as BlobPart], { type: "image/jpeg" });
        const [left, right] = await Promise.all([
          createImageBitmap(blob(pair.left)),
          createImageBitmap(blob(pair.right)),
        ]);
        // Swap both eyes together: never show a new left with an old right.
        const previous = this.latest;
        this.latest = { meta: pair.meta, left, right, receivedAt: performance.now() };
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

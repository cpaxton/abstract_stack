// H.264 stereo video in "QTH1" framing: ASCII "QTH1", uint32 BE lengths
// (metadata, access unit), JSON metadata, Annex-B access unit. Both eyes are
// side by side in one frame, decoded with WebCodecs.
import type { PairMetadata } from "./stereo";

/** Constrained Baseline, level 3.1, as the host encodes it. */
export const H264_CODEC = "avc1.42E01F";

export interface H264Metadata extends PairMetadata {
  layout: "side_by_side";
  keyframe: boolean;
  bitrate_kbps: number;
}

export function isH264Message(buffer: ArrayBuffer): boolean {
  if (buffer.byteLength < 4) return false;
  const b = new Uint8Array(buffer, 0, 4);
  return b[0] === 0x51 && b[1] === 0x54 && b[2] === 0x48 && b[3] === 0x31; // "QTH1"
}

export function parseH264(buffer: ArrayBuffer): { meta: H264Metadata; data: Uint8Array } {
  if (buffer.byteLength < 12 || !isH264Message(buffer)) throw new Error("bad H.264 message");
  const view = new DataView(buffer);
  const metaBytes = view.getUint32(4, false);
  const dataBytes = view.getUint32(8, false);
  if (12 + metaBytes + dataBytes !== buffer.byteLength) {
    throw new Error("H.264 message length mismatch");
  }
  const meta = JSON.parse(
    new TextDecoder().decode(new Uint8Array(buffer, 12, metaBytes)),
  ) as H264Metadata;
  return { meta, data: new Uint8Array(buffer, 12 + metaBytes, dataBytes) };
}

let support: Promise<boolean> | null = null;

/** Whether this browser can decode the host's H.264 (WebCodecs needs a secure context). */
export function canDecodeH264(): Promise<boolean> {
  support ??= (async () => {
    if (typeof window === "undefined" || !window.isSecureContext) return false;
    if (typeof VideoDecoder === "undefined") return false;
    try {
      const result = await VideoDecoder.isConfigSupported({
        codec: H264_CODEC,
        optimizeForLatency: true,
      });
      return result.supported === true;
    } catch {
      return false;
    }
  })();
  return support;
}

export interface DecodedPair {
  meta: H264Metadata;
  left: ImageBitmap;
  right: ImageBitmap;
  arrivedAt: number;
}

/**
 * WebCodecs decoder for side-by-side stereo H.264. Delta frames cannot be skipped,
 * so a decoder that errors or falls behind resets and asks the host for a
 * keyframe (`requestKeyframe`); until one arrives, delta frames are dropped.
 */
export class H264StereoDecoder {
  private decoder: VideoDecoder | null = null;
  private needKeyframe = true;
  private lastRequest = 0;
  private pending = new Map<number, { meta: H264Metadata; arrivedAt: number }>();
  private newest = 0;
  private deliver: (pair: DecodedPair) => void;
  private requestKeyframe: () => void;

  constructor(deliver: (pair: DecodedPair) => void, requestKeyframe: () => void) {
    this.deliver = deliver;
    this.requestKeyframe = requestKeyframe;
  }

  push(buffer: ArrayBuffer, arrivedAt: number): void {
    const { meta, data } = parseH264(buffer);
    if (this.needKeyframe && !meta.keyframe) {
      this.askForKeyframe();
      return;
    }
    const decoder = this.ensureDecoder();
    // More than a few frames queued means this device cannot keep up: start over
    // from a keyframe rather than fall further behind.
    if (decoder.decodeQueueSize > 3) {
      this.reset();
      return;
    }
    this.needKeyframe = false;
    this.pending.set(meta.pair_id, { meta, arrivedAt });
    decoder.decode(
      new EncodedVideoChunk({
        type: meta.keyframe ? "key" : "delta",
        timestamp: meta.pair_id,
        data,
      }),
    );
  }

  clear(): void {
    this.decoder?.close();
    this.decoder = null;
    this.pending.clear();
    this.needKeyframe = true;
  }

  private ensureDecoder(): VideoDecoder {
    if (this.decoder && this.decoder.state !== "closed") return this.decoder;
    const decoder = new VideoDecoder({
      output: (frame) => this.onFrame(frame),
      error: (error) => {
        console.warn("H.264 decoder error, waiting for a keyframe:", error);
        this.reset();
      },
    });
    decoder.configure({ codec: H264_CODEC, optimizeForLatency: true });
    this.decoder = decoder;
    return decoder;
  }

  private onFrame(frame: VideoFrame): void {
    const id = frame.timestamp;
    const info = this.pending.get(id);
    for (const key of this.pending.keys()) if (key <= id) this.pending.delete(key);
    if (!info || id <= this.newest) {
      frame.close();
      return;
    }
    this.newest = id;
    const width = frame.codedWidth / 2;
    const height = info.meta.height;
    Promise.all([
      createImageBitmap(frame, 0, 0, width, height),
      createImageBitmap(frame, width, 0, width, height),
    ])
      .then(([left, right]) => this.deliver({ ...info, left, right }))
      .catch((error) => console.warn("H.264 frame dropped:", error))
      .finally(() => frame.close());
  }

  private reset(): void {
    this.clear();
    this.askForKeyframe();
  }

  private askForKeyframe(): void {
    const now = performance.now();
    if (now - this.lastRequest < 500) return;
    this.lastRequest = now;
    this.requestKeyframe();
  }
}

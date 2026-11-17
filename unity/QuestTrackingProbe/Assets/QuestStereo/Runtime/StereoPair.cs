using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using Newtonsoft.Json;
using Newtonsoft.Json.Linq;

namespace AbstractStack.QuestStereo
{
    // No Unity API: parsing and latest-pair exchange may run off the render thread.
    public sealed class StereoPair
    {
        public const int MaxMetadata = 16384, MaxEye = 2097152;
        public const int MaxMessage = 16 + MaxMetadata + 2 * MaxEye;
        public const long MaxInteger = 9007199254740991L;
        public readonly JObject Metadata;
        public readonly byte[] Left, Right;
        public string SessionId => (string)Metadata["session_id"];
        public long Generation => (long)Metadata["calibration_generation"];
        public long PairId => (long)Metadata["pair_id"];
        public int Width => (int)Metadata["width"];
        public int Height => (int)Metadata["height"];
        public double CaptureHostTime => (double)Metadata["capture_host_monotonic_s"];

        StereoPair(JObject metadata, byte[] left, byte[] right)
        { Metadata = metadata; Left = left; Right = right; }

        public static StereoPair Parse(byte[] packet, int count)
        {
            Require(packet != null && count >= 16 && count <= packet.Length && count <= MaxMessage, "packet size");
            Require(packet[0] == 'Q' && packet[1] == 'T' && packet[2] == 'P' && packet[3] == '1', "magic");
            long metadataSize = UInt(packet, 4), leftSize = UInt(packet, 8), rightSize = UInt(packet, 12);
            Require(metadataSize > 0 && metadataSize <= MaxMetadata && leftSize > 0 && leftSize <= MaxEye && rightSize > 0 && rightSize <= MaxEye, "region size");
            Require(16L + metadataSize + leftSize + rightSize == count, "truncated/trailing pair");
            JObject header;
            using (var input = new StringReader(new UTF8Encoding(false, true).GetString(packet, 16, (int)metadataSize)))
            using (var reader = new JsonTextReader(input) { DateParseHandling = DateParseHandling.None, MaxDepth = 12 })
            {
                header = JObject.Load(reader, new JsonLoadSettings { DuplicatePropertyNameHandling = DuplicatePropertyNameHandling.Error });
                Require(!reader.Read(), "trailing JSON");
            }
            ValidateHeader(header);
            int leftStart = 16 + (int)metadataSize, rightStart = leftStart + (int)leftSize;
            // Check SOF dimensions before decoder allocation, not only after LoadImage.
            ValidateJpeg(packet, leftStart, (int)leftSize, (int)header["width"], (int)header["height"]);
            ValidateJpeg(packet, rightStart, (int)rightSize, (int)header["width"], (int)header["height"]);
            byte[] left = new byte[leftSize], right = new byte[rightSize];
            Buffer.BlockCopy(packet, leftStart, left, 0, left.Length);
            Buffer.BlockCopy(packet, rightStart, right, 0, right.Length);
            return new StereoPair(header, left, right);
        }

        public static void ValidateHeader(JObject h)
        {
            Keys(h, "schema_version,type,session_id,calibration_generation,pair_id,sim_time_s,capture_host_monotonic_s,codec,pixel_format,frame,width,height,eye_order,cameras");
            Integer(h["schema_version"]); Require((long)h["schema_version"] == 1, "schema");
            Exact(h["type"], "stereo_pair"); Exact(h["codec"], "jpeg"); Exact(h["pixel_format"], "rgb8"); Exact(h["frame"], "mabel_chassis_native");
            Require(h["session_id"].Type == JTokenType.String && ((string)h["session_id"]).Length > 0 && ((string)h["session_id"]).Length <= 128, "session");
            Integer(h["pair_id"]); Integer(h["calibration_generation"]);
            Number(h["sim_time_s"], true); Number(h["capture_host_monotonic_s"], true);
            Integer(h["width"]); Integer(h["height"]);
            int width = (int)h["width"], height = (int)h["height"];
            Require(width >= 4 && height >= 3 && width <= 1600 && height <= 1200 && width * 3 == height * 4, "dimensions/aspect");
            var order = h["eye_order"] as JArray;
            Require(order != null && order.Count == 2, "eye order"); Exact(order[0], "left"); Exact(order[1], "right");
            var cameras = h["cameras"] as JObject; Keys(cameras, "left,right");
            foreach (string eye in new[] { "left", "right" })
            {
                var camera = cameras[eye] as JObject; Keys(camera, "name,position_m,quaternion_wxyz,vertical_fov_deg"); Exact(camera["name"], "zed_" + eye);
                Array(camera["position_m"], 3); Array(camera["quaternion_wxyz"], 4);
                double norm = ((JArray)camera["quaternion_wxyz"]).Sum(v => (double)v * (double)v);
                Require(Math.Abs(Math.Sqrt(norm) - 1) <= .001, "quaternion norm");
                double fov = Number(camera["vertical_fov_deg"]); Require(fov > 0 && fov < 180, "FOV");
            }
        }

        static long UInt(byte[] p, int i) => ((long)p[i] << 24) | ((long)p[i + 1] << 16) | ((long)p[i + 2] << 8) | p[i + 3];
        public static void Require(bool condition, string message) { if (!condition) throw new FormatException(message); }
        public static void Keys(JObject obj, string names)
        {
            Require(obj != null, "object"); var expected = new HashSet<string>(names.Split(','));
            Require(obj.Count == expected.Count && obj.Properties().All(p => expected.Contains(p.Name)), "object fields");
        }
        public static long Integer(JToken token)
        {
            Require(token != null && token.Type == JTokenType.Integer, "integer type");
            long value = (long)token; Require(value >= 0 && value <= MaxInteger, "integer range"); return value;
        }
        public static double Number(JToken token, bool nonnegative = false)
        {
            Require(token != null && (token.Type == JTokenType.Float || token.Type == JTokenType.Integer), "number type");
            double value = (double)token; Require(!double.IsNaN(value) && !double.IsInfinity(value) && (!nonnegative || value >= 0), "finite number"); return value;
        }
        static void Exact(JToken token, string value) => Require(token != null && token.Type == JTokenType.String && (string)token == value, "expected " + value);
        static void Array(JToken token, int count)
        { var array = token as JArray; Require(array != null && array.Count == count, "array size"); foreach (var value in array) Number(value); }

        static void ValidateJpeg(byte[] p, int start, int length, int width, int height)
        {
            int end = start + length;
            Require(length >= 4 && p[start] == 255 && p[start + 1] == 216 && p[end - 2] == 255 && p[end - 1] == 217, "JPEG markers");
            int i = start + 2;
            while (i + 3 < end)
            {
                Require(p[i++] == 255, "JPEG segment"); while (i < end && p[i] == 255) i++;
                Require(i < end, "JPEG marker"); int marker = p[i++];
                Require(marker != 218 && marker != 217, "missing JPEG SOF");
                Require(i + 2 <= end, "JPEG length"); int size = (p[i] << 8) | p[i + 1];
                Require(size >= 2 && i + size <= end, "JPEG region");
                bool sof = marker >= 192 && marker <= 207 && marker != 196 && marker != 200 && marker != 204;
                if (sof)
                {
                    Require((marker == 192 || marker == 194) && size >= 11 && p[i + 2] == 8 && p[i + 7] == 3, "8-bit RGB JPEG required");
                    Require(((p[i + 3] << 8) | p[i + 4]) == height && ((p[i + 5] << 8) | p[i + 6]) == width, "JPEG dimensions"); return;
                }
                i += size;
            }
            throw new FormatException("missing JPEG dimensions");
        }
    }

    public sealed class LatestStereoPair
    {
        readonly object gate = new object();
        StereoPair pending; string session; long generation, highWater = -1;
        public long Replaced { get; private set; }
        public long Rejected { get; private set; }
        public void Reset(string id, long gen) { lock (gate) { session = id; generation = gen; pending = null; highWater = -1; } }
        public bool Offer(StereoPair pair)
        {
            lock (gate)
            {
                if (pair.SessionId != session || pair.Generation != generation || pair.PairId <= highWater) { Rejected++; return false; }
                highWater = pair.PairId; if (pending != null) Replaced++; pending = pair; return true;
            }
        }
        public StereoPair Take() { lock (gate) { var result = pending; pending = null; return result; } }
    }
}

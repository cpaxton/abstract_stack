using System;
using System.Text;
using AbstractStack.QuestStereo;
using Newtonsoft.Json.Linq;
using NUnit.Framework;
using UnityEditor;
using UnityEngine;

public sealed class StereoPairTests
{
    static JObject Header(long id = 1, string session = "test", long generation = 0)
    {
        JObject Camera(string eye) => new JObject { ["name"] = "zed_" + eye, ["position_m"] = new JArray(0, eye == "left" ? -.0315 : .0315, 1), ["quaternion_wxyz"] = new JArray(1, 0, 0, 0), ["vertical_fov_deg"] = 83.3 };
        return new JObject { ["schema_version"] = 1, ["type"] = "stereo_pair", ["session_id"] = session, ["calibration_generation"] = generation, ["pair_id"] = id,
            ["sim_time_s"] = 0.0, ["capture_host_monotonic_s"] = 1.0, ["codec"] = "jpeg", ["pixel_format"] = "rgb8", ["frame"] = "mabel_chassis_native",
            ["width"] = 64, ["height"] = 48, ["eye_order"] = new JArray("left", "right"), ["cameras"] = new JObject { ["left"] = Camera("left"), ["right"] = Camera("right") } };
    }
    static byte[] Jpeg(Color color)
    {
        var texture = new Texture2D(64, 48, TextureFormat.RGB24, false);
        var pixels = new Color[64 * 48]; for (int i = 0; i < pixels.Length; i++) pixels[i] = color;
        texture.SetPixels(pixels); texture.Apply(); var result = texture.EncodeToJPG(95); UnityEngine.Object.DestroyImmediate(texture); return result;
    }
    static byte[] Packet(JObject header = null, byte[] left = null, byte[] right = null)
    {
        byte[] metadata = Encoding.UTF8.GetBytes((header ?? Header()).ToString(Newtonsoft.Json.Formatting.None));
        left = left ?? Jpeg(Color.red); right = right ?? Jpeg(Color.blue);
        byte[] packet = new byte[16 + metadata.Length + left.Length + right.Length];
        Encoding.ASCII.GetBytes("QTP1").CopyTo(packet, 0);
        Write(packet, 4, metadata.Length); Write(packet, 8, left.Length); Write(packet, 12, right.Length);
        metadata.CopyTo(packet, 16); left.CopyTo(packet, 16 + metadata.Length); right.CopyTo(packet, 16 + metadata.Length + left.Length); return packet;
    }
    static void Write(byte[] bytes, int at, int value)
    { bytes[at] = (byte)(value >> 24); bytes[at + 1] = (byte)(value >> 16); bytes[at + 2] = (byte)(value >> 8); bytes[at + 3] = (byte)value; }
    static StereoPair Pair(long id = 1, string session = "test", long generation = 0)
    { var p = Packet(Header(id, session, generation)); return StereoPair.Parse(p, p.Length); }

    [Test] public void ValidPairPreservesIdentityAndEyeRegions()
    {
        var pair = Pair(123); Assert.That(pair.PairId, Is.EqualTo(123)); Assert.That(pair.Width, Is.EqualTo(64));
        var left = new Texture2D(2, 2); var right = new Texture2D(2, 2);
        Assert.That(left.LoadImage(pair.Left), Is.True); Assert.That(right.LoadImage(pair.Right), Is.True);
        Assert.That(left.GetPixel(20, 20).r, Is.GreaterThan(.9)); Assert.That(left.GetPixel(20, 20).b, Is.LessThan(.1));
        Assert.That(right.GetPixel(20, 20).b, Is.GreaterThan(.9)); Assert.That(right.GetPixel(20, 20).r, Is.LessThan(.1));
        UnityEngine.Object.DestroyImmediate(left); UnityEngine.Object.DestroyImmediate(right);
    }
    [Test] public void LatestQueueReplacesWholePairWithoutHistory()
    {
        var queue = new LatestStereoPair(); queue.Reset("test", 0);
        for (int i = 0; i < 20; i++) Assert.That(queue.Offer(Pair(i)), Is.True);
        Assert.That(queue.Replaced, Is.EqualTo(19)); Assert.That(queue.Take().PairId, Is.EqualTo(19)); Assert.That(queue.Take(), Is.Null);
    }
    [Test] public void RejectsReorderedDuplicateOldSessionAndOldGeneration()
    {
        var queue = new LatestStereoPair(); queue.Reset("test", 1); Assert.That(queue.Offer(Pair(5, "test", 1)), Is.True);
        Assert.That(queue.Offer(Pair(5, "test", 1)), Is.False); Assert.That(queue.Offer(Pair(4, "test", 1)), Is.False);
        Assert.That(queue.Offer(Pair(6, "old", 1)), Is.False); Assert.That(queue.Offer(Pair(6, "test", 0)), Is.False);
        Assert.That(queue.Take().PairId, Is.EqualTo(5)); Assert.That(queue.Rejected, Is.EqualTo(4));
        queue.Reset("new", 0); Assert.That(queue.Take(), Is.Null); Assert.That(queue.Offer(Pair(0, "new", 0)), Is.True);
    }
    [TestCase("magic")][TestCase("length")][TestCase("trailing")][TestCase("dimensions")][TestCase("aspect")][TestCase("eye_order")][TestCase("unknown")][TestCase("fraction_id")][TestCase("nan")][TestCase("quaternion")][TestCase("jpeg_bomb")]
    public void RejectsMalformedPacketsBeforeDecode(string fault)
    {
        var h = Header(); byte[] packet;
        switch (fault)
        {
            case "dimensions": h["width"] = 128; h["height"] = 96; break;
            case "aspect": h["width"] = 65; break;
            case "eye_order": h["eye_order"] = new JArray("right", "left"); break;
            case "unknown": h["quality"] = 80; break;
            case "fraction_id": h["pair_id"] = 1.5; break;
            case "nan": h["capture_host_monotonic_s"] = double.NaN; break;
            case "quaternion": h["cameras"]["left"]["quaternion_wxyz"] = new JArray(0, 0, 0, 0); break;
        }
        packet = Packet(h);
        if (fault == "magic") packet[0] = 0;
        if (fault == "length") Write(packet, 8, StereoPair.MaxEye + 1);
        if (fault == "trailing") System.Array.Resize(ref packet, packet.Length + 1);
        if (fault == "jpeg_bomb")
        {
            int metadata = (packet[4] << 24) | (packet[5] << 16) | (packet[6] << 8) | packet[7];
            for (int i = 16 + metadata; i < packet.Length - 8; i++) if (packet[i] == 255 && packet[i + 1] == 192) { packet[i + 5] = 127; break; }
        }
        Assert.Throws<FormatException>(() => StereoPair.Parse(packet, packet.Length));
    }
    [Test] public void StereoShaderCompilesAndDefaultsToRealXrEyeIndex()
    {
        var shader = Shader.Find("QuestTeleop/RoomStereo"); Assert.That(shader, Is.Not.Null); Assert.That(ShaderUtil.ShaderHasError(shader), Is.False);
        var material = new Material(shader); Assert.That(material.GetFloat("_DebugEye"), Is.EqualTo(-1)); UnityEngine.Object.DestroyImmediate(material);
    }
}

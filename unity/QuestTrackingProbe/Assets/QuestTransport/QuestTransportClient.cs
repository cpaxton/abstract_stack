using System;
using System.Collections.Concurrent;
using System.Diagnostics;
using System.IO;
using System.Net.WebSockets;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using Newtonsoft.Json;
using Newtonsoft.Json.Linq;
using UnityEngine;

namespace QuestTeleop.Transport
{
    // Add to a GameObject; canonical acquisition and video UI use the public injection API.
    public sealed class QuestTransportClient : MonoBehaviour
    {
        public string ControlUri = "ws://127.0.0.1:8765/quest/control";
        public event Action<JObject> MessageReceived;
        public string SessionId { get; private set; }
        public long CalibrationGeneration { get; private set; }
        public bool Connected { get; private set; }
        public static double MonotonicSeconds => (double)Stopwatch.GetTimestamp() / Stopwatch.Frequency;
        readonly ConcurrentQueue<JObject> reliable = new ConcurrentQueue<JObject>();
        readonly ConcurrentQueue<JObject> received = new ConcurrentQueue<JObject>();
        readonly object poseLock = new object();
        JObject latest;
        CancellationTokenSource lifetime;
        long sequence, eventId, exchangeId;
        bool referenceChangePending;
        long referenceEventId;
        public string LastError { get; private set; }

        void OnEnable() { lifetime = new CancellationTokenSource(); _ = Run(lifetime.Token); }
        void OnDisable() { lifetime?.Cancel(); Connected = false; }
        void Update()
        {
            while (received.TryDequeue(out var message))
            {
                if ((string)message["session_id"] != SessionId) continue;
                // An exact retry can replay an ACK from an earlier generation.
                // Deliver the ACK unchanged without moving the current session backwards.
                if ((long?)message["calibration_generation"] is long generation &&
                    generation > CalibrationGeneration)
                    CalibrationGeneration = generation;
                if ((string)message["type"] == "control_ack" && referenceChangePending && (long?)message["event_id"] == referenceEventId) referenceChangePending = false;
                MessageReceived?.Invoke(message);
            }
        }
        public void SubmitTracking(JObject canonical)
        {
            // DeepCopy prevents acquisition mutating an observation while serialization runs.
            // Each source's observed_monotonic_s and source timestamp are preserved verbatim.
            if (!Connected || referenceChangePending) return;
            lock (poseLock) latest = (JObject)canonical.DeepClone();
        }
        public void SendReliable(JObject message)
        {
            if (!Connected) return;
            if (reliable.Count >= 64) { lifetime.Cancel(); return; } // fail closed on stalled link
            reliable.Enqueue((JObject)message.DeepClone());
        }
        public void SendControl(string action)
        {
            lock (poseLock) latest = null;
            SendReliable(new JObject { ["type"] = "control", ["event_id"] = eventId++,
                ["calibration_generation"] = CalibrationGeneration, ["action"] = action });
        }
        public void ReferenceSpaceChanged() { referenceChangePending = true; referenceEventId = eventId; SendControl("reference_space_changed"); }
        void OnApplicationPause(bool paused)
        {
            if (paused) { Connected = false; lifetime?.Cancel(); }
            else if (isActiveAndEnabled && lifetime != null && lifetime.IsCancellationRequested)
            { lifetime.Dispose(); lifetime = new CancellationTokenSource(); _ = Run(lifetime.Token); }
        }
        JObject Envelope(JObject message)
        {
            message["schema_version"] = 1; message["session_id"] = SessionId; return message;
        }
        static async Task Send(ClientWebSocket socket, JObject message, CancellationToken token)
        {
            var bytes = Encoding.UTF8.GetBytes(message.ToString(Formatting.None));
            if (bytes.Length > 262144) throw new InvalidDataException("oversized control JSON");
            using (var timeout = CancellationTokenSource.CreateLinkedTokenSource(token))
            {
                timeout.CancelAfter(2000);
                await socket.SendAsync(new ArraySegment<byte>(bytes), WebSocketMessageType.Text, true, timeout.Token);
            }
        }
        async Task Receive(ClientWebSocket socket, CancellationToken token)
        {
            var buffer = new byte[8192];
            while (!token.IsCancellationRequested)
            {
                using (var stream = new MemoryStream())
                {
                    WebSocketReceiveResult result;
                    do {
                        result = await socket.ReceiveAsync(new ArraySegment<byte>(buffer), token);
                        if (result.MessageType != WebSocketMessageType.Text) throw new IOException("non-text control");
                        if (stream.Length + result.Count > 262144) throw new IOException("oversized JSON");
                        stream.Write(buffer, 0, result.Count);
                    } while (!result.EndOfMessage);
                    var message = JObject.Parse(new UTF8Encoding(false, true).GetString(stream.ToArray()));
                    if ((int?)message["schema_version"] != 1 || (string)message["session_id"] != SessionId)
                        throw new IOException("session/schema mismatch");
                    if ((string)message["type"] == "hello_ack") Connected = true;
                    if ((string)message["type"] == "clock_pong")
                    {
                        message["type"] = "clock_result"; message["t3"] = MonotonicSeconds;
                        SendReliable(message);
                    }
                    else
                    {
                        if (received.Count >= 64) throw new IOException("status consumer stalled");
                        received.Enqueue(message);
                    }
                }
            }
        }
        async Task Run(CancellationToken token)
        {
            var retryMs = 250;
            while (!token.IsCancellationRequested)
            {
                using (var socket = new ClientWebSocket())
                using (var connectionLife = CancellationTokenSource.CreateLinkedTokenSource(token))
                {
                    SessionId = Guid.NewGuid().ToString(); CalibrationGeneration = 0;
                    sequence = eventId = exchangeId = 0; Connected = false; referenceChangePending = false;
                    lock (poseLock) latest = null;
                    while (reliable.TryDequeue(out _)) { }
                    while (received.TryDequeue(out _)) { }
                    try
                    {
                        connectionLife.CancelAfter(5000);
                        await socket.ConnectAsync(new Uri(ControlUri), connectionLife.Token);
                        connectionLife.CancelAfter(Timeout.Infinite);
                        await Send(socket, Envelope(new JObject { ["type"] = "hello", ["profile"] = "quest_teleop_v1" }), connectionLife.Token);
                        var receive = Receive(socket, connectionLife.Token);
                        double lastClock = double.NegativeInfinity, lastPose = 0, connectedAt = MonotonicSeconds;
                        while (!receive.IsCompleted && !token.IsCancellationRequested)
                        {
                            while (reliable.TryDequeue(out var message)) await Send(socket, Envelope(message), connectionLife.Token);
                            var now = MonotonicSeconds;
                            if (!Connected && now - connectedAt > 5) throw new IOException("hello ack timeout");
                            if (Connected && now - lastClock >= 2)
                            {
                                await Send(socket, Envelope(new JObject { ["type"] = "clock_ping", ["exchange_id"] = exchangeId++, ["t0"] = now }), connectionLife.Token);
                                lastClock = now;
                            }
                            JObject pose = null;
                            if (Connected && now - lastPose >= 1.0 / 72)
                            {
                                lock (poseLock) { pose = latest; latest = null; }
                                if (pose != null)
                                {
                                    pose["type"] = "tracking"; pose["sequence"] = sequence++;
                                    pose["calibration_generation"] = CalibrationGeneration;
                                    pose["sent_monotonic_s"] = now;
                                    await Send(socket, Envelope(pose), connectionLife.Token); lastPose = now;
                                }
                            }
                            await Task.Delay(2, connectionLife.Token);
                        }
                        await receive;
                    }
                    catch (Exception ex) when (ex is WebSocketException || ex is IOException || ex is JsonException || ex is OperationCanceledException)
                    { LastError = ex.GetType().Name + ": " + ex.Message; }
                    finally { Connected = false; connectionLife.Cancel(); socket.Abort(); lock (poseLock) latest = null; }
                }
                try { await Task.Delay(retryMs, token); } catch (OperationCanceledException) { break; }
                retryMs = Math.Min(retryMs * 2, 4000);
            }
        }
    }
}

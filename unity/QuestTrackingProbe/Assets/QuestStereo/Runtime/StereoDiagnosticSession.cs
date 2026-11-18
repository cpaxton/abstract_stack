using System;
using System.Collections.Concurrent;
using System.Net.WebSockets;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using Newtonsoft.Json.Linq;

namespace AbstractStack.QuestStereo
{
    // A receive-only demonstration session, not the teleoperation capture client.
    // Integrated apps supply their existing #15 session to StereoViewer instead.
    public sealed class StereoDiagnosticSession : IDisposable
    {
        readonly StereoViewer viewer;
        readonly Uri host;
        readonly CancellationTokenSource cancel = new CancellationTokenSource();
        readonly ConcurrentQueue<Action> main = new ConcurrentQueue<Action>();
        readonly object pendingGate = new object();
        JObject feedback; bool pause;
        ClientWebSocket socket;
        public StereoDiagnosticSession(StereoViewer target, Uri hostUri)
        {
            viewer = target; host = hostUri;
            if (host.Scheme != "ws" && host.Scheme != "wss") throw new ArgumentException("WebSocket host required");
        }
        public void Start() { _ = Task.Run(Run); }
        public void Pump() { while (main.TryDequeue(out var action)) if (!cancel.IsCancellationRequested) action(); }
        void Dispatch(Action action) { StereoPair.Require(main.Count < 32, "main-thread control queue overflow"); main.Enqueue(action); }
        void Feedback(JObject message) { lock (pendingGate) feedback = message; }
        void Pause() { lock (pendingGate) pause = true; }
        async Task Run()
        {
            while (!cancel.IsCancellationRequested)
            {
                var ws = new ClientWebSocket(); socket = ws;
                var writer = new SemaphoreSlim(1, 1);
                using (var connectionCancel = CancellationTokenSource.CreateLinkedTokenSource(cancel.Token))
                {
                    var token = connectionCancel.Token; string id = Guid.NewGuid().ToString(); long generation = 0, exchange = 0, eventId = 0;
                    double pingT0 = -1; long pingId = 0;
                    async Task Send(JObject message)
                    {
                        await writer.WaitAsync(token).ConfigureAwait(false);
                        try { await StereoVideoReceiver.SendJson(ws, message, token).ConfigureAwait(false); }
                        finally { writer.Release(); }
                    }
                    JObject Envelope(string kind) => new JObject { ["schema_version"] = 1, ["type"] = kind, ["session_id"] = id };
                    try
                    {
                        var controlUri = new UriBuilder(host) { Path = "/quest/control" }.Uri;
                        await ws.ConnectAsync(controlUri, token).ConfigureAwait(false);
                        var hello = Envelope("hello"); hello["profile"] = "quest_teleop_v1"; await Send(hello).ConfigureAwait(false);
                        byte[] buffer = new byte[262144]; bool ready = false;
                        Task sender = null;
                        while (!token.IsCancellationRequested)
                        {
                            var received = await StereoVideoReceiver.Receive(ws, buffer, buffer.Length, token).ConfigureAwait(false);
                            if (received.Item1 == WebSocketMessageType.Close) break;
                            StereoPair.Require(received.Item1 == WebSocketMessageType.Text, "control text required");
                            var message = JObject.Parse(new UTF8Encoding(false, true).GetString(buffer, 0, received.Item2));
                            StereoPair.Require((int?)message["schema_version"] == 1 && (string)message["session_id"] == id, "control session");
                            switch ((string)message["type"])
                            {
                                case "hello_ack":
                                    StereoPair.Require(!ready && (string)message["video_codec"] == "jpeg" && (string)message["video_profile"] == "QTP1", "unsupported hello");
                                    generation = StereoPair.Integer(message["calibration_generation"]); ready = true;
                                    Dispatch(() => viewer.ConfigureSession(new UriBuilder(host) { Path = "/quest/video" }.Uri, id, generation, Feedback, Pause));
                                    sender = Task.Run(async () => {
                                        double nextPing = 0;
                                        while (!token.IsCancellationRequested)
                                        {
                                            double now = StereoViewer.MonotonicSeconds;
                                            if (now >= nextPing)
                                            {
                                                nextPing = now + 2; pingId = ++exchange; pingT0 = now;
                                                var ping = Envelope("clock_ping"); ping["exchange_id"] = pingId; ping["t0"] = now; await Send(ping).ConfigureAwait(false);
                                            }
                                            JObject status; bool request;
                                            lock (pendingGate) { status = feedback; feedback = null; request = pause; pause = false; }
                                            if (status != null) await Send(status).ConfigureAwait(false);
                                            if (request)
                                            {
                                                var command = Envelope("control"); command["event_id"] = ++eventId; command["calibration_generation"] = generation; command["action"] = "pause";
                                                await Send(command).ConfigureAwait(false);
                                            }
                                            await Task.Delay(20, token).ConfigureAwait(false);
                                        }
                                    }, token);
                                    break;
                                case "clock_pong":
                                    double t3 = StereoViewer.MonotonicSeconds;
                                    StereoPair.Require(StereoPair.Integer(message["exchange_id"]) == pingId && StereoPair.Number(message["t0"], true) == pingT0, "clock echo");
                                    double t1 = StereoPair.Number(message["t1"], true), t2 = StereoPair.Number(message["t2"], true);
                                    StereoPair.Require(t2 >= t1 && t3 >= pingT0 && (t3 - pingT0) >= (t2 - t1), "clock ordering");
                                    var result = Envelope("clock_result"); result["exchange_id"] = pingId; result["t0"] = pingT0; result["t1"] = t1; result["t2"] = t2; result["t3"] = t3;
                                    await Send(result).ConfigureAwait(false); break;
                                case "clock_ack":
                                    StereoPair.Require(StereoPair.Integer(message["exchange_id"]) == pingId, "clock acknowledgement");
                                    double offset = StereoPair.Number(message["offset_s"]), uncertainty = StereoPair.Number(message["uncertainty_s"], true);
                                    Dispatch(() => viewer.UpdateClock(offset, uncertainty)); break;
                                case "control_ack":
                                case "status":
                                    if (message["calibration_generation"] != null && StereoPair.Integer(message["calibration_generation"]) != generation)
                                        throw new FormatException("generation changed; diagnostic session reconnect required");
                                    break;
                            }
                            if (sender != null && sender.IsFaulted) await sender.ConfigureAwait(false);
                        }
                        connectionCancel.Cancel();
                        if (sender != null) try { await sender.ConfigureAwait(false); } catch (OperationCanceledException) { }
                    }
                    catch (Exception error) when (error is WebSocketException || error is OperationCanceledException || error is FormatException || error is Newtonsoft.Json.JsonException)
                    { if (!cancel.IsCancellationRequested) Dispatch(() => UnityEngine.Debug.LogWarning("[QuestStereo] Control reconnect: " + error.Message)); }
                    finally
                    {
                        connectionCancel.Cancel(); ws.Abort(); ws.Dispose();
                        lock (pendingGate) { feedback = null; pause = false; }
                        if (!cancel.IsCancellationRequested) Dispatch(viewer.ClearSession);
                    }
                }
                try { await Task.Delay(1000, cancel.Token).ConfigureAwait(false); } catch (OperationCanceledException) { }
            }
        }
        public void Dispose() { cancel.Cancel(); socket?.Abort(); while (main.TryDequeue(out _)) { } }
    }
}

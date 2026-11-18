using System;
using System.Net.WebSockets;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using Newtonsoft.Json.Linq;

namespace AbstractStack.QuestStereo
{
    public sealed class StereoVideoReceiver : IDisposable
    {
        readonly LatestStereoPair latest;
        readonly CancellationTokenSource cancel = new CancellationTokenSource();
        readonly ClientWebSocket socket = new ClientWebSocket();
        public volatile bool Connected;
        public string Diagnostic { get; private set; } = "Disconnected";
        public long Received { get; private set; }
        public long Malformed { get; private set; }
        public StereoVideoReceiver(LatestStereoPair queue) { latest = queue; }
        public async Task Run(Uri uri, string session)
        {
            try
            {
                await socket.ConnectAsync(uri, cancel.Token).ConfigureAwait(false);
                var bind = new JObject { ["schema_version"] = 1, ["type"] = "video_bind", ["session_id"] = session };
                await SendJson(socket, bind, cancel.Token).ConfigureAwait(false);
                var buffer = new byte[StereoPair.MaxMessage];
                var ack = await Receive(socket, buffer, 16384, cancel.Token).ConfigureAwait(false);
                StereoPair.Require(ack.Item1 == WebSocketMessageType.Text, "video bind acknowledgement must be text");
                var header = JObject.Parse(new UTF8Encoding(false, true).GetString(buffer, 0, ack.Item2));
                StereoPair.Require((int?)header["schema_version"] == 1 && (string)header["type"] == "video_bind_ack" && (string)header["session_id"] == session, "video bind rejected");
                Connected = true; Diagnostic = "Receiving";
                while (!cancel.IsCancellationRequested)
                {
                    var message = await Receive(socket, buffer, buffer.Length, cancel.Token).ConfigureAwait(false);
                    if (message.Item1 == WebSocketMessageType.Close) break;
                    if (message.Item1 != WebSocketMessageType.Binary) { Malformed++; continue; }
                    try { var pair = StereoPair.Parse(buffer, message.Item2); if (latest.Offer(pair)) Received++; }
                    catch (Exception error) when (error is FormatException || error is Newtonsoft.Json.JsonException || error is DecoderFallbackException || error is OverflowException)
                    { Malformed++; Diagnostic = "Rejected pair: " + error.Message; }
                }
            }
            catch (Exception error) when (error is WebSocketException || error is OperationCanceledException || error is FormatException || error is Newtonsoft.Json.JsonException)
            { Diagnostic = cancel.IsCancellationRequested ? "Disconnected" : error.GetType().Name + ": " + error.Message; }
            finally { Connected = false; socket.Abort(); }
        }
        public static async Task<Tuple<WebSocketMessageType, int>> Receive(ClientWebSocket ws, byte[] buffer, int limit, CancellationToken token)
        {
            int count = 0; WebSocketMessageType? kind = null;
            while (true)
            {
                StereoPair.Require(count < limit, "WebSocket message too large");
                var result = await ws.ReceiveAsync(new ArraySegment<byte>(buffer, count, limit - count), token).ConfigureAwait(false);
                if (result.MessageType == WebSocketMessageType.Close) return Tuple.Create(WebSocketMessageType.Close, 0);
                if (kind.HasValue) StereoPair.Require(kind == result.MessageType, "mixed WebSocket fragments");
                kind = result.MessageType; count += result.Count;
                if (result.EndOfMessage) return Tuple.Create(result.MessageType, count);
            }
        }
        public static Task SendJson(ClientWebSocket ws, JObject value, CancellationToken token)
        {
            byte[] bytes = Encoding.UTF8.GetBytes(value.ToString(Newtonsoft.Json.Formatting.None));
            return ws.SendAsync(new ArraySegment<byte>(bytes), WebSocketMessageType.Text, true, token);
        }
        public void Dispose() { cancel.Cancel(); socket.Abort(); socket.Dispose(); }
    }
}

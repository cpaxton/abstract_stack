using Newtonsoft.Json.Linq;
using UnityEngine;

namespace QuestTeleop.Transport
{
    // A usable canonical observation input for synthetic fixtures and recorded canonical JSONL.
    // Never attach a raw RawTrackingRecorder file: its local OVR hand rotations aren't globals.
    public sealed class CanonicalReplaySource : MonoBehaviour
    {
        public QuestTransportClient Client;
        public TextAsset CanonicalJsonLines;
        public bool SyntheticRetiming;
        public float FramesPerSecond = 30;
        string[] lines;
        int index;
        double next;
        double initialObserved, replayStart;
        void OnEnable()
        {
            if (CanonicalJsonLines == null) return;
            lines = CanonicalJsonLines.text.Split(new[] {'\n'}, System.StringSplitOptions.RemoveEmptyEntries);
            index = 0; next = 0; replayStart = QuestTransportClient.MonotonicSeconds;
            if (lines.Length > 0) initialObserved = (double)JObject.Parse(lines[0])["sent_monotonic_s"];
        }
        void Update()
        {
            if (Client == null || !Client.Connected || lines == null || index >= lines.Length) return;
            var now = QuestTransportClient.MonotonicSeconds;
            if (now < next) return;
            var observation = JObject.Parse(lines[index++]);
            if (SyntheticRetiming)
            {
                // Explicit synthetic replay mode: preserve inter-source timing offsets; not live data.
                var shift = replayStart - initialObserved;
                foreach (var name in new[] {"head", "left_hand", "right_hand", "body"})
                    observation[name]["observed_monotonic_s"] = (double)observation[name]["observed_monotonic_s"] + shift;
            }
            Client.SubmitTracking(observation);
            next = now + 1.0 / Mathf.Max(1, FramesPerSecond);
        }
    }
}

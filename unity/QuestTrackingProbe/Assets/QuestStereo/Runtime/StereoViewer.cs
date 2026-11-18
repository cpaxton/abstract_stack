using System;
using System.Diagnostics;
using System.IO;
using Newtonsoft.Json.Linq;
using UnityEngine;
using UnityEngine.Rendering;
using UnityEngine.XR;

namespace AbstractStack.QuestStereo
{
    // ConfigureSession, UpdateClock and ClearSession are called on Unity's main thread.
    // Network receive owns no Unity objects; Update performs one whole-pair decode.
    public sealed class StereoViewer : MonoBehaviour
    {
        public Camera viewerCamera;
        public MeshRenderer window;
        public UnityEngine.UI.Text status;
        public float windowWidthMetres = 1.2f, windowDistanceMetres = 1.5f, windowHeightMetres = 1.4f;
        public bool synthetic = true;
        public bool allowEditorPresentationForTests;
        public double LastDecodeMilliseconds { get; private set; }
        public long Decoded { get; private set; }
        public long DecodeRejected { get; private set; }
        public long Submitted { get; private set; }
        public bool Stale { get; private set; } = true;
        public static double MonotonicSeconds => (double)Stopwatch.GetTimestamp() / Stopwatch.Frequency;
        public readonly LatestStereoPair Queue = new LatestStereoPair();
        StereoVideoReceiver receiver;
        StereoDiagnosticSession diagnostic;
        Material material;
        Texture2D left, right;
        StereoPair displayed;
        string session;
        long generation, sequence;
        double offset, uncertainty = double.PositiveInfinity, clockAt, presentedAt, presentedCaptureHostTime, feedbackAt, syntheticAt;
        long? presentedId;
        bool pauseRequestedForStale, active = true, focused = true, paused;
        Action<JObject> sendControl;
        Action requestPause;
        readonly System.Collections.Generic.List<XRDisplaySubsystem> displays = new System.Collections.Generic.List<XRDisplaySubsystem>();
        long syntheticId;
        string networkHost;
        bool placementInitialized;

        void Awake()
        {
            if (!window || !viewerCamera) { enabled = false; return; }
            material = window.material;
            RenderPipelineManager.endCameraRendering += OnCameraRendered;
        }
        void Start()
        {
            string path = Path.Combine(Application.persistentDataPath, "stereo-config.json");
            if (File.Exists(path))
            {
                try
                {
                    var config = JObject.Parse(File.ReadAllText(path));
                    synthetic = (string)config["mode"] != "network";
                    networkHost = (string)config["host"];
                    windowWidthMetres = Mathf.Clamp((float?)config["window_width_m"] ?? 1.2f, .3f, 2.5f);
                    windowDistanceMetres = Mathf.Clamp((float?)config["window_distance_m"] ?? 1.5f, .5f, 4f);
                    windowHeightMetres = Mathf.Clamp((float?)config["window_height_m"] ?? 1.4f, .3f, 2.5f);
                }
                catch (Exception error) { UnityEngine.Debug.LogError("[QuestStereo] Configuration: " + error.Message); synthetic = true; }
            }
            if (!synthetic && !string.IsNullOrEmpty(networkHost)) StartDiagnostic(networkHost);
            UnityEngine.Debug.Log("[QuestStereo] Started mode=" + (synthetic ? "local_synthetic" : "network") + " size_m=" + windowWidthMetres);
        }
        public void StartDiagnostic(string host)
        {
            diagnostic?.Dispose(); synthetic = false; networkHost = host;
            diagnostic = new StereoDiagnosticSession(this, new Uri(host)); diagnostic.Start();
        }
        public void ConfigureSession(Uri videoUri, string sessionId, long calibrationGeneration, Action<JObject> reliableControl, Action pauseRequest = null)
        {
            ClearSession(); synthetic = false; session = sessionId; generation = calibrationGeneration;
            sendControl = reliableControl; requestPause = pauseRequest; Queue.Reset(session, generation);
            receiver = new StereoVideoReceiver(Queue); _ = receiver.Run(videoUri, session);
        }
        public void UpdateClock(double hostMinusClientSeconds, double uncertaintySeconds)
        {
            if (double.IsNaN(hostMinusClientSeconds) || double.IsInfinity(hostMinusClientSeconds) || double.IsNaN(uncertaintySeconds) || uncertaintySeconds < 0 || uncertaintySeconds > .025) return;
            offset = hostMinusClientSeconds; uncertainty = uncertaintySeconds; clockAt = MonotonicSeconds;
        }
        public void ClearSession()
        {
            receiver?.Dispose(); receiver = null; Queue.Reset(null, 0); displayed = null; session = null;
            presentedId = null; presentedAt = 0; sequence = 0; uncertainty = double.PositiveInfinity; clockAt = 0;
            sendControl = null; requestPause = null; Stale = true; pauseRequestedForStale = false;
            if (material) { material.SetTexture("_LeftTex", Texture2D.blackTexture); material.SetTexture("_RightTex", Texture2D.blackTexture); material.SetFloat("_Stale", 1); }
            DestroyTextures();
        }
        public void SetGeneration(long value)
        {
            if (value == generation) return;
            // An external transport must call ConfigureSession after invalidation;
            // this immediately clears queued/displayed frames from the old generation.
            ClearSession(); generation = value;
        }
        void Update()
        {
            diagnostic?.Pump();
            if (!placementInitialized && viewerCamera.transform.position.y > .2f)
            {
                // Place once in room space, never parent to or continuously follow head.
                Vector3 forward = Vector3.ProjectOnPlane(viewerCamera.transform.forward, Vector3.up).normalized;
                if (forward.sqrMagnitude > .5f)
                {
                    Vector3 position = viewerCamera.transform.position + forward * windowDistanceMetres;
                    position.y = windowHeightMetres; transform.position = position;
                    transform.rotation = Quaternion.LookRotation(forward, Vector3.up); placementInitialized = true;
                }
            }
            window.transform.localScale = new Vector3(windowWidthMetres, windowWidthMetres * .75f, 1);
            if (synthetic)
            {
                if (MonotonicSeconds - syntheticAt >= 1.0 / 15)
                { syntheticAt = MonotonicSeconds; ShowSynthetic(++syntheticId); }
                SetStale(!active);
            }
            else
            {
                var pair = Queue.Take(); if (pair != null) Decode(pair);
                double now = MonotonicSeconds;
                double age = !presentedId.HasValue ? double.PositiveInfinity : now + offset - presentedCaptureHostTime;
                SetStale(!active || receiver == null || !receiver.Connected || displayed == null ||
                    uncertainty > .025 || now - clockAt > 5 || age + uncertainty > .300 || age < -uncertainty);
                if (sendControl != null && now - feedbackAt >= .05)
                {
                    feedbackAt = now;
                    sendControl(new JObject {
                        ["schema_version"] = 1, ["type"] = "video_status", ["session_id"] = session,
                        ["sequence"] = ++sequence, ["calibration_generation"] = generation,
                        ["presented_pair_id"] = presentedId.HasValue ? (JToken)presentedId.Value : JValue.CreateNull(),
                        ["presented_monotonic_s"] = presentedId.HasValue ? (JToken)presentedAt : JValue.CreateNull(),
                        ["reported_monotonic_s"] = now, ["stale"] = Stale });
                }
            }
            if (material) material.SetFloat("_Stale", Stale ? 1 : 0);
            if (status) status.text = synthetic ? "SYNTHETIC STEREO CHECK\nClose right eye: L / red. Close left: R / blue.\nCross should fuse in front of grid. Move head: window stays in room.\nPair " + syntheticId :
                "QUEST HEAD CAMERAS  |  " + (Stale ? "STALE / UNSYNCHRONIZED" : "LIVE") + "\n" + (receiver?.Diagnostic ?? "Disconnected") +
                "  Pair " + (displayed?.PairId.ToString() ?? "none") + "\nDecode " + LastDecodeMilliseconds.ToString("F1") + " ms  Received " + (receiver?.Received ?? 0) + "  Drops " + Queue.Replaced +
                "\nRejected " + ((receiver?.Malformed ?? 0) + DecodeRejected) + "  Submitted " + Submitted;
        }
        void SetStale(bool stale)
        {
            Stale = stale;
            if (!stale) { pauseRequestedForStale = false; return; }
            if (pauseRequestedForStale || requestPause == null) return;
            pauseRequestedForStale = true;
            requestPause.Invoke();
        }
        void Decode(StereoPair pair)
        {
            if (pair.SessionId != session || pair.Generation != generation) return;
            double start = MonotonicSeconds;
            // Decode into a new pair. Never mutate textures still bound to either eye.
            var nextLeft = new Texture2D(2, 2, TextureFormat.RGB24, false, false);
            var nextRight = new Texture2D(2, 2, TextureFormat.RGB24, false, false);
            if (!ImageConversion.LoadImage(nextLeft, pair.Left, true) || !ImageConversion.LoadImage(nextRight, pair.Right, true) ||
                nextLeft.width != pair.Width || nextLeft.height != pair.Height || nextRight.width != pair.Width || nextRight.height != pair.Height)
            { Destroy(nextLeft); Destroy(nextRight); DecodeRejected++; return; }
            DestroyTextures(); left = nextLeft; right = nextRight;
            material.SetTexture("_LeftTex", left); material.SetTexture("_RightTex", right); displayed = pair;
            LastDecodeMilliseconds = (MonotonicSeconds - start) * 1000; Decoded++;
            if (Decoded % 30 == 1) UnityEngine.Debug.Log("[QuestStereo] decoded pair=" + pair.PairId + " dimensions=" + pair.Width + "x" + pair.Height + " decode_ms=" + LastDecodeMilliseconds.ToString("F3"));
        }
        void OnCameraRendered(ScriptableRenderContext context, Camera camera)
        {
            if (camera != viewerCamera || !active || synthetic || displayed == null || !window.enabled || !window.gameObject.activeInHierarchy) return;
            SubsystemManager.GetSubsystems(displays);
            if (!allowEditorPresentationForTests && (!camera.stereoEnabled || !displays.Exists(d => d.running))) return;
            if (!GeometryUtility.TestPlanesAABB(GeometryUtility.CalculateFrustumPlanes(camera), window.bounds)) return;
            // This records an active XR camera render submission after drawing the
            // complete material pair. It does not measure compositor/optical photons.
            presentedId = displayed.PairId; presentedAt = MonotonicSeconds; presentedCaptureHostTime = displayed.CaptureHostTime; Submitted++;
            if (Submitted % 72 == 1) UnityEngine.Debug.Log("[QuestStereo] render_submission pair=" + presentedId + " presented_monotonic_s=" + presentedAt.ToString("F6") + " capture_host_monotonic_s=" + presentedCaptureHostTime.ToString("F6") + " mapped_capture_age_s=" + (presentedAt + offset - presentedCaptureHostTime).ToString("F6") + " clock_uncertainty_s=" + uncertainty.ToString("F6"));
        }
        void ShowSynthetic(long id)
        {
            DestroyTextures(); left = SyntheticStereo.Create(true, id); right = SyntheticStereo.Create(false, id);
            material.SetTexture("_LeftTex", left); material.SetTexture("_RightTex", right);
        }
        void DestroyTextures() { if (left) Destroy(left); if (right) Destroy(right); left = right = null; }
        void OnApplicationFocus(bool value) { focused = value; active = focused && !paused; if (!active) Suspend(); else Resume(); }
        void OnApplicationPause(bool value) { paused = value; active = focused && !paused; if (!active) Suspend(); else Resume(); }
        void Suspend()
        {
            // The control owner must see the pause request before its callback
            // and the session's presentation state are cleared.
            try { SetStale(true); }
            finally
            {
                diagnostic?.Dispose();
                diagnostic = null;
                ClearSession();
            }
        }
        void Resume() { if (!synthetic && !string.IsNullOrEmpty(networkHost) && diagnostic == null) StartDiagnostic(networkHost); }
        void OnDestroy() { RenderPipelineManager.endCameraRendering -= OnCameraRendered; diagnostic?.Dispose(); ClearSession(); if (material) Destroy(material); }
    }
}

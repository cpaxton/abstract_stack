using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading;
using UnityEngine;

// Read-only native SDK capture. No retargeting, simulation or robot connection.
public sealed class RawTrackingRecorder : MonoBehaviour
{
    const string Format = "quest_tracking_probe_v1";
    const string Permission = "com.oculus.permission.BODY_TRACKING";
    readonly ConcurrentQueue<string> pending = new ConcurrentQueue<string>();
    readonly AutoResetEvent wake = new AutoResetEvent(false);
    Thread writer;
    volatile bool stopping;
    volatile string writerError;
    int queued, dropped;
    ulong sequence;
    double startedAt, nextMarkerCheck;
    public static string Status = "starting";
    const double CaptureSeconds = 180;
    string session;
    double lastLeft, lastRight, lastBody;
    bool hadLeft, hadRight, hadBody, hadPresence, previousPresence;
    int previousRecenter;
    bool leftSkeleton, rightSkeleton;
    uint bodySkeleton;
    bool hadBodySkeleton;
    OVRPlugin.HandState left, right;
    OVRPlugin.BodyState body;

    [Serializable] class Header
    {
        public string format = Format, type = "session", session_id;
        public int schema_version = 1;
        public string unity_version, meta_core = "209.0.0", openxr = "1.19.0", ovrplugin_version, runtime_version;
        public string device_model, operating_system, package_id, application_version;
        public string pose_frame = "ovr_tracking_space_right_handed", quaternion_order = "xyzw", position_unit = "m";
        public string head_time_semantics = "runtime_pose_target_time_not_sensor_sample_time";
        public string body_time_semantics = "runtime_body_state_time_not_proof_of_sensor_observation";
        public string hand_time_semantics = "sdk_sample_timestamp_separate_from_requested_timestamp";
        public string hand_bone_semantics = "OVR_parent_local_rotations_bind_translations_sdk_basis_not_tracking_space";
        public string tracking_origin;
        public string utc_started;
    }
    [Serializable] class Pose
    {
        public float[] position_m, quaternion_xyzw;
        public Pose(OVRPlugin.Posef p)
        {
            position_m = new[] { p.Position.x, p.Position.y, p.Position.z };
            quaternion_xyzw = new[] { p.Orientation.x, p.Orientation.y, p.Orientation.z, p.Orientation.w };
        }
    }
    [Serializable] class Head
    {
        public Pose pose;
        public bool position_valid, orientation_valid, position_tracked, orientation_tracked;
        public double pose_target_time_s;
        public string timestamp_domain = "ovr_runtime", source_freshness = "unknown";
    }
    [Serializable] class Hand
    {
        public bool api_success, tracked;
        public string skeleton_profile, source_freshness, confidence, status;
        public double sample_time_s, requested_time_s;
        public string timestamp_domain = "ovr_runtime", per_joint_validity = "unavailable", per_joint_tracking = "unavailable";
        public Pose root_pose;
        public float scale;
        public string[] finger_confidences;
        public HandBone[] bones;
        public float[] pinch_strength;
        public int pinch_flags;
    }
    [Serializable] class HandBone
    {
        public int id;
        public string name;
        public float[] rotation_xyzw, position_m;
    }
    [Serializable] class Joint
    {
        public int id;
        public string name;
        public Pose pose;
        public bool position_valid, orientation_valid, position_tracked, orientation_tracked;
    }
    [Serializable] class Body
    {
        public bool api_success, permission_granted;
        public double state_time_s;
        public string timestamp_domain = "ovr_runtime", source_freshness = "unknown", state_time_change, joint_set, fidelity, calibration;
        public float confidence;
        public uint skeleton_changed_count;
        public Joint[] joints;
    }
    [Serializable] class Sample
    {
        public string format = Format, type = "sample", session_id;
        public ulong sequence;
        public double capture_monotonic_s;
        public int recenter_count, dropped_records;
        public bool user_present, focused;
        public Head head;
        public Hand left_hand, right_hand;
        public Body body;
    }
    [Serializable] class Event
    {
        public string format = Format, type = "event", session_id, name, detail;
        public double capture_monotonic_s;
    }
    [Serializable] class Bone
    {
        public int id, parent_index;
        public string name;
        public Pose bind_pose;
    }
    [Serializable] class Skeleton
    {
        public string format = Format, type = "skeleton", session_id, side, profile;
        public double capture_monotonic_s;
        public Bone[] bones;
    }

    void Start()
    {
        session = Guid.NewGuid().ToString("N");
        string path = Path.Combine(Application.persistentDataPath, "raw-" + session + ".jsonl");
        writer = new Thread(() => WriteLoop(path)) { IsBackground = true, Name = "Quest raw tracking writer" };
        writer.Start();
        startedAt = Time.realtimeSinceStartupAsDouble;
        Status = "recording (180s maximum)";
        previousRecenter = OVRPlugin.GetLocalTrackingSpaceRecenterCount();
        Enqueue(new Header {
            session_id = session, unity_version = Application.unityVersion,
            ovrplugin_version = OVRPlugin.version.ToString(), runtime_version = OVRPlugin.nativeSDKVersion.ToString(),
            device_model = SystemInfo.deviceModel, operating_system = SystemInfo.operatingSystem,
            package_id = Application.identifier, application_version = Application.version,
            tracking_origin = OVRManager.instance ? OVRManager.instance.trackingOriginType.ToString() : "unavailable",
            utc_started = DateTime.UtcNow.ToString("O")
        });
        Debug.Log("[RawTracking] Recording " + path);
    }

    void LateUpdate()
    {
        if (session == null || stopping) return;
        if (Time.realtimeSinceStartupAsDouble - startedAt >= CaptureSeconds) { StopRecording("duration_limit"); return; }
        if (Time.realtimeSinceStartupAsDouble >= nextMarkerCheck)
        {
            nextMarkerCheck = Time.realtimeSinceStartupAsDouble + 1;
            string marker = Path.Combine(Application.persistentDataPath, "marker.txt");
            if (File.Exists(marker)) { RecordEvent("operator_marker", File.ReadAllText(marker).Trim()); File.Delete(marker); }
        }
        if (writerError != null) { Status = "FAILED: " + writerError; Debug.LogError("[RawTracking] Capture failed: " + writerError); StopRecording("writer_failure"); enabled = false; return; }
        var node = OVRPlugin.Node.EyeCenter;
        var headPose = OVRPlugin.GetNodePoseStateRaw(node, OVRPlugin.Step.Render);
        int recenter = OVRPlugin.GetLocalTrackingSpaceRecenterCount();
        bool presence = OVRManager.instance && OVRManager.instance.isUserPresent;
        if (recenter != previousRecenter) { RecordEvent("recenter", previousRecenter + "->" + recenter); previousRecenter = recenter; }
        if (!hadPresence || presence != previousPresence) { RecordEvent("user_present", presence.ToString()); previousPresence = presence; hadPresence = true; }
        bool lg = OVRPlugin.GetHandState(OVRPlugin.Step.Render, OVRPlugin.Hand.HandLeft, ref left);
        bool rg = OVRPlugin.GetHandState(OVRPlugin.Step.Render, OVRPlugin.Hand.HandRight, ref right);
        if (!leftSkeleton) leftSkeleton = RecordSkeleton(OVRPlugin.SkeletonType.HandLeft, "left_hand");
        if (!rightSkeleton) rightSkeleton = RecordSkeleton(OVRPlugin.SkeletonType.HandRight, "right_hand");
        bool bg = OVRPlugin.GetBodyState4(OVRPlugin.Step.Render, OVRPlugin.BodyJointSet.UpperBody, ref body);
        if (bg && (!hadBodySkeleton || body.SkeletonChangedCount != bodySkeleton))
        {
            hadBodySkeleton = RecordSkeleton(OVRPlugin.SkeletonType.Body, "body");
            bodySkeleton = body.SkeletonChangedCount;
        }
        var joints = bg && body.JointLocations != null ? body.JointLocations.Select((j, i) => new Joint {
            id = i, name = BoneName(i, "Body_"), pose = new Pose(j.Pose), position_valid = j.PositionValid,
            orientation_valid = j.OrientationValid, position_tracked = j.PositionTracked, orientation_tracked = j.OrientationTracked
        }).ToArray() : new Joint[0];
        Enqueue(new Sample {
            session_id = session, sequence = sequence++, capture_monotonic_s = Time.realtimeSinceStartupAsDouble,
            recenter_count = recenter, user_present = presence, focused = Application.isFocused, dropped_records = dropped,
            head = new Head { pose = new Pose(headPose.Pose), pose_target_time_s = headPose.Time,
                position_valid = OVRPlugin.GetNodePositionValid(node), orientation_valid = OVRPlugin.GetNodeOrientationValid(node),
                position_tracked = OVRPlugin.GetNodePositionTracked(node), orientation_tracked = OVRPlugin.GetNodeOrientationTracked(node) },
            left_hand = CaptureHand(lg, left, ref lastLeft, ref hadLeft),
            right_hand = CaptureHand(rg, right, ref lastRight, ref hadRight),
            body = new Body { api_success = bg, permission_granted = PermissionGranted(), state_time_s = bg ? body.Time : 0,
                state_time_change = Change(bg, body.Time, ref lastBody, ref hadBody),
                confidence = bg ? body.Confidence : 0, joint_set = bg ? body.JointSet.ToString() : "unavailable",
                fidelity = bg ? body.Fidelity.ToString() : "unavailable", calibration = bg ? body.CalibrationStatus.ToString() : "unavailable",
                skeleton_changed_count = bg ? body.SkeletonChangedCount : 0, joints = joints }
        }, true);
    }

    static Hand CaptureHand(bool ok, OVRPlugin.HandState state, ref double previous, ref bool had)
    {
        bool tracked = ok && (state.Status & OVRPlugin.HandStatus.HandTracked) != 0;
        bool xr = OVRPlugin.HandSkeletonVersion == OVRHandSkeletonVersion.OpenXR;
        return new Hand {
            api_success = ok, tracked = tracked, skeleton_profile = xr ? "openxr_hand_26" : "ovr_hand_24",
            source_freshness = Change(tracked, state.SampleTimeStamp, ref previous, ref had),
            sample_time_s = ok ? state.SampleTimeStamp : 0, requested_time_s = ok ? state.RequestedTimeStamp : 0,
            confidence = ok ? state.HandConfidence.ToString() : "unavailable", status = ok ? state.Status.ToString() : "unavailable",
            root_pose = ok ? new Pose(state.RootPose) : null, scale = ok ? state.HandScale : 0,
            bones = ok && state.BoneRotations != null ? state.BoneRotations.Select((q, i) => new HandBone {
                id = i, name = BoneName(i, xr ? "XRHand_" : "Hand_"), rotation_xyzw = new[] { q.x, q.y, q.z, q.w },
                position_m = state.BonePositions != null && i < state.BonePositions.Length
                    ? new[] { state.BonePositions[i].x, state.BonePositions[i].y, state.BonePositions[i].z } : new float[0]
            }).ToArray() : new HandBone[0],
            finger_confidences = ok && state.FingerConfidences != null ? state.FingerConfidences.Select(c => c.ToString()).ToArray() : new string[0],
            pinch_strength = ok ? state.PinchStrength : new float[0], pinch_flags = ok ? (int)state.Pinches : 0
        };
    }

    // Never count an invalid API result or a repeat as a fresh source sample.
    static string Change(bool ok, double time, ref double previous, ref bool had)
    {
        if (!ok || time <= 0 || double.IsNaN(time) || double.IsInfinity(time)) return "unavailable";
        string result = !had ? "first" : time > previous ? "advanced" : time == previous ? "repeated" : "regressed";
        previous = had ? Math.Max(previous, time) : time; had = true;
        return result;
    }

    bool RecordSkeleton(OVRPlugin.SkeletonType type, string side)
    {
        OVRPlugin.Skeleton2 skeleton = default;
        if (!OVRPlugin.GetSkeleton2(type, ref skeleton) || skeleton.Bones == null) return false;
        return Enqueue(new Skeleton { session_id = session, side = side, profile = skeleton.Type.ToString(),
            capture_monotonic_s = Time.realtimeSinceStartupAsDouble,
            bones = skeleton.Bones.Take((int)skeleton.NumBones).Select(b => new Bone { id = (int)b.Id,
                name = BoneName((int)b.Id, side == "body" ? "Body_" : "Hand_"), parent_index = b.ParentBoneIndex, bind_pose = new Pose(b.Pose) }).ToArray() });
    }

    static readonly Dictionary<string, string[]> Names = new Dictionary<string, string[]>();
    static string BoneName(int id, string prefix)
    {
        if (!Names.TryGetValue(prefix, out string[] names))
        {
            names = new string[84];
            foreach (string name in Enum.GetNames(typeof(OVRPlugin.BoneId)))
            {
                int value = (int)Enum.Parse(typeof(OVRPlugin.BoneId), name);
                if (name.StartsWith(prefix, StringComparison.Ordinal) && !name.EndsWith("Start", StringComparison.Ordinal)
                    && !name.EndsWith("End", StringComparison.Ordinal) && !name.Contains("Max") && value >= 0 && value < names.Length)
                    names[value] = name;
            }
            Names[prefix] = names;
        }
        return id >= 0 && id < names.Length && names[id] != null ? names[id] : prefix + id;
    }

    static bool PermissionGranted()
    {
#if UNITY_ANDROID && !UNITY_EDITOR
        return UnityEngine.Android.Permission.HasUserAuthorizedPermission(Permission);
#else
        return false;
#endif
    }

    bool Enqueue(object value, bool sample = false)
    {
        // Serialize before reserving capacity: an exception cannot leak a queue slot.
        string line = JsonUtility.ToJson(value);
        if (sample && queued >= 128) { Interlocked.Increment(ref dropped); return false; }
        if (queued >= 256) { writerError = "critical record queue exceeded 256; capture incomplete"; return false; }
        Interlocked.Increment(ref queued); pending.Enqueue(line); wake.Set();
        return true;
    }

    void WriteLoop(string path)
    {
        try
        {
            using (var file = new StreamWriter(path, false))
            {
                while (!stopping || !pending.IsEmpty)
                {
                    while (pending.TryDequeue(out string line)) { Interlocked.Decrement(ref queued); file.WriteLine(line); }
                    file.Flush();
                    if (!stopping) wake.WaitOne(250);
                }
            }
        }
        catch (Exception error) { writerError = error.ToString(); }
    }

    void RecordEvent(string name, string detail)
    {
        if (session != null) Enqueue(new Event { session_id = session, name = name, detail = detail, capture_monotonic_s = Time.realtimeSinceStartupAsDouble });
    }
    void OnApplicationPause(bool pause) { RecordEvent("application_pause", pause.ToString()); }
    void OnApplicationFocus(bool focus) { RecordEvent("application_focus", focus.ToString()); }
    void StopRecording(string reason)
    {
        if (stopping) return;
        RecordEvent("recording_stopped", reason + "; dropped_records=" + dropped);
        stopping = true; wake.Set();
        Status = "stopped: " + reason + "; dropped=" + dropped;
        if (writer != null && !writer.Join(2000)) { Status = "FAILED writer drain"; Debug.LogError("[RawTracking] Writer did not drain before exit"); }
    }
    void OnDestroy() { StopRecording("recorder_destroyed"); }
}

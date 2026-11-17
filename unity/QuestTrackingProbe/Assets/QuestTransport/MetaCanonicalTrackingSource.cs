using System;
using Newtonsoft.Json.Linq;
using UnityEngine;

namespace QuestTeleop.Transport
{
    // Polls one SDK state per source. Global hand FK follows Meta XR Core209 OVRSkeleton:
    // FlipX bind translations/local rotations, wrist Y180, FlipZ root, hand scale.
    // Reference: Packages/com.meta.xr.sdk.core/Scripts/Util/OVRSkeleton.cs InitializeBones/Update.
    public sealed class MetaCanonicalTrackingSource : MonoBehaviour
    {
        public QuestTransportClient Client;
        public bool ManageBodyTracking = true;
        const string BodyPermission = "com.oculus.permission.BODY_TRACKING";
        bool bodyStarted;
        float nextBodyRetry;
        OVRPlugin.HandState left, right;
        OVRPlugin.BodyState body;
        OVRPlugin.Skeleton2 leftBind, rightBind;
        double lastLeft, lastRight, lastBody;
        int recenter;
        static readonly string[] Hands = ("WristRoot ForearmStub Thumb0 Thumb1 Thumb2 Thumb3 Index1 Index2 Index3 Middle1 Middle2 Middle3 Ring1 Ring2 Ring3 Pinky0 Pinky1 Pinky2 Pinky3 ThumbTip IndexTip MiddleTip RingTip PinkyTip").Split(' ');
        static readonly string[] Bodies = ("Root Hips SpineLower SpineMiddle SpineUpper Chest Neck Head LeftShoulder LeftScapula LeftArmUpper LeftArmLower LeftHandWristTwist RightShoulder RightScapula RightArmUpper RightArmLower RightHandWristTwist").Split(' ');
        void OnEnable()
        {
            recenter = OVRPlugin.GetLocalTrackingSpaceRecenterCount();
#if UNITY_ANDROID && !UNITY_EDITOR
            if (ManageBodyTracking && !UnityEngine.Android.Permission.HasUserAuthorizedPermission(BodyPermission))
                UnityEngine.Android.Permission.RequestUserPermission(BodyPermission);
#endif
        }
        void OnDisable() { if (ManageBodyTracking && bodyStarted) OVRPlugin.StopBodyTracking(); bodyStarted = false; }
        void EnsureBodyTracking()
        {
            if (!ManageBodyTracking || bodyStarted || Time.unscaledTime < nextBodyRetry) return;
            nextBodyRetry = Time.unscaledTime + 2;
#if UNITY_ANDROID && !UNITY_EDITOR
            if (!UnityEngine.Android.Permission.HasUserAuthorizedPermission(BodyPermission)) return;
#endif
            bodyStarted = OVRPlugin.StartBodyTracking2(OVRPlugin.BodyJointSet.UpperBody);
        }
        static JToken Null => JValue.CreateNull();
        static JArray Position(Vector3 p) => new JArray(p.z, -p.x, p.y);
        static JArray Rotation(Quaternion q)
        {
            // Reflection B=(z,-x,y): vector part transforms by det(B)*B.
            return new JArray(q.w, -q.z, q.x, -q.y);
        }
        static JObject Pose(Vector3 p, Quaternion q, bool pv, bool ov, bool? pt = null, bool? ot = null)
            => new JObject { ["position_m"] = pv ? Position(p) : Null,
                ["quaternion_wxyz"] = ov ? Rotation(q) : Null,
                ["position_valid"] = pv, ["orientation_valid"] = ov,
                ["position_tracked"] = pt.HasValue ? new JValue(pt.Value) : Null,
                ["orientation_tracked"] = ot.HasValue ? new JValue(ot.Value) : Null,
                ["confidence"] = Null };
        static JObject NativePose(OVRPlugin.Posef p, bool pv, bool ov, bool? pt = null, bool? ot = null)
            => Pose(p.Position.FromFlippedZVector3f(), p.Orientation.FromFlippedZQuatf(), pv, ov, pt, ot);
        static JObject Source(double observed, double? runtime, string kind, bool? fresh)
            => new JObject { ["observed_monotonic_s"] = observed,
                ["source_time_s"] = runtime.HasValue ? new JValue(runtime.Value) : Null,
                ["source_clock"] = runtime.HasValue ? "quest_runtime" : "unavailable",
                ["timestamp_kind"] = kind, ["fresh"] = fresh.HasValue ? new JValue(fresh.Value) : Null };
        static bool ValidTime(double time) => time > 0 && !double.IsNaN(time) && !double.IsInfinity(time);
        static JObject UnavailableHand(double observed)
        {
            var source = Source(observed, null, "unavailable", null);
            source["pose"] = Pose(default, Quaternion.identity, false, false);
            source["skeleton_profile"] = "ovr_hand_24";
            var joints = new JArray();
            foreach (var name in Hands)
            { var p = Pose(default, Quaternion.identity, false, false); p["name"] = "Hand_" + name; joints.Add(p); }
            source["joints"] = joints; return source;
        }
        static JObject CaptureHand(OVRPlugin.Hand side, ref OVRPlugin.HandState state,
            ref OVRPlugin.Skeleton2 bind, ref double highwater)
        {
            bool ok = OVRPlugin.GetHandState(OVRPlugin.Step.Render, side, ref state);
            double observed = QuestTransportClient.MonotonicSeconds;
            bool tracked = ok && (state.Status & OVRPlugin.HandStatus.HandTracked) != 0;
            if (!tracked || !ValidTime(state.SampleTimeStamp) || state.BoneRotations == null ||
                state.BoneRotations.Length < 24 || !float.IsFinite(state.HandScale) || state.HandScale <= 0)
                return UnavailableHand(observed);
            var skeletonType = side == OVRPlugin.Hand.HandLeft ? OVRPlugin.SkeletonType.HandLeft : OVRPlugin.SkeletonType.HandRight;
            if (bind.Bones == null && !OVRPlugin.GetSkeleton2(skeletonType, ref bind))
                return UnavailableHand(observed);
            if (bind.Bones == null || bind.NumBones != 24) return UnavailableHand(observed);
            var positions = new Vector3[24]; var rotations = new Quaternion[24]; var visited = new byte[24];
            var rootPosition = state.RootPose.Position.FromFlippedZVector3f();
            var rootRotation = state.RootPose.Orientation.FromFlippedZQuatf();
            // Copy retrieved state into local arrays; recursion only reads this exact sample.
            var localRotations = new Quaternion[24];
            for (int i = 0; i < 24; i++)
            {
                if ((int)bind.Bones[i].Id != i) return UnavailableHand(observed);
                localRotations[i] = state.BoneRotations[i].FromFlippedXQuatf();
                if (i == (int)OVRPlugin.BoneId.Hand_WristRoot)
                    localRotations[i] *= new Quaternion(0, 1, 0, 0);
            }
            var bones = bind.Bones; var scale = state.HandScale;
            Action<int> compose = null;
            compose = i => {
                if (visited[i] == 2) return;
                if (visited[i] == 1) throw new InvalidOperationException("cyclic hand skeleton");
                visited[i] = 1;
                int parent = bones[i].ParentBoneIndex;
                bool hasParent = parent >= 0 && parent < 24;
                if (hasParent) compose(parent);
                var pp = hasParent ? positions[parent] : rootPosition;
                var pq = hasParent ? rotations[parent] : rootRotation;
                positions[i] = pp + pq * (scale * bones[i].Pose.Position.FromFlippedXVector3f());
                rotations[i] = pq * localRotations[i]; visited[i] = 2;
            };
            for (int i = 0; i < 24; i++) compose(i);
            var source = Source(observed, state.SampleTimeStamp, "sensor_sample", state.SampleTimeStamp > highwater);
            highwater = Math.Max(highwater, state.SampleTimeStamp);
            // SDK exposes a whole-hand tracked flag, no independent per-joint validity/confidence.
            source["pose"] = Pose(positions[0], rotations[0], true, true, tracked, tracked);
            source["skeleton_profile"] = "ovr_hand_24";
            var joints = new JArray();
            for (int i = 0; i < 24; i++)
            { var p = Pose(positions[i], rotations[i], true, true); p["name"] = "Hand_" + Hands[i]; joints.Add(p); }
            source["joints"] = joints; return source;
        }
        void Update()
        {
            EnsureBodyTracking();
            if (Client == null || !Client.Connected) return;
            if (OVRPlugin.HandSkeletonVersion != OVRHandSkeletonVersion.OVR)
            { UnityEngine.Debug.LogError("Quest transport requires OVR24 hand skeleton"); enabled = false; return; }
            var currentRecenter = OVRPlugin.GetLocalTrackingSpaceRecenterCount();
            if (currentRecenter != recenter)
            { recenter = currentRecenter; Client.ReferenceSpaceChanged(); return; }
            var node = OVRPlugin.Node.EyeCenter;
            var headState = OVRPlugin.GetNodePoseStateRaw(node, OVRPlugin.Step.Render);
            var head = Source(QuestTransportClient.MonotonicSeconds,
                ValidTime(headState.Time) ? headState.Time : (double?)null,
                ValidTime(headState.Time) ? "predicted_target" : "unavailable", null);
            head["pose"] = NativePose(headState.Pose, OVRPlugin.GetNodePositionValid(node),
                OVRPlugin.GetNodeOrientationValid(node), OVRPlugin.GetNodePositionTracked(node), OVRPlugin.GetNodeOrientationTracked(node));
            var lh = CaptureHand(OVRPlugin.Hand.HandLeft, ref left, ref leftBind, ref lastLeft);
            var rh = CaptureHand(OVRPlugin.Hand.HandRight, ref right, ref rightBind, ref lastRight);
            bool ok = OVRPlugin.GetBodyState4(OVRPlugin.Step.Render, OVRPlugin.BodyJointSet.UpperBody, ref body);
            var bodySource = Source(QuestTransportClient.MonotonicSeconds,
                ok && ValidTime(body.Time) ? body.Time : (double?)null,
                ok && ValidTime(body.Time) ? "state_target" : "unavailable", null);
            bodySource["confidence"] = ok ? new JValue(body.Confidence) : Null;
            if (ok && ValidTime(body.Time)) bodySource["state_advanced"] = body.Time > lastBody;
            if (ok) lastBody = Math.Max(lastBody, body.Time);
            bodySource["skeleton_profile"] = "ovr_body_upper_18";
            var bodyJoints = new JArray();
            foreach (var name in Bodies)
            {
                int id = (int)(OVRPlugin.BoneId)Enum.Parse(typeof(OVRPlugin.BoneId), "Body_" + name);
                JObject p;
                if (ok && body.JointLocations != null && id < body.JointLocations.Length)
                { var j = body.JointLocations[id]; p = NativePose(j.Pose, j.PositionValid, j.OrientationValid, j.PositionTracked, j.OrientationTracked); }
                else p = Pose(default, Quaternion.identity, false, false);
                p["name"] = "Body_" + name; bodyJoints.Add(p);
            }
            bodySource["joints"] = bodyJoints;
            Client.SubmitTracking(new JObject { ["frame"] = "operator_forward_left_up",
                ["worn"] = OVRManager.instance != null && OVRManager.instance.isUserPresent,
                ["focused"] = Application.isFocused, ["head"] = head, ["left_hand"] = lh,
                ["right_hand"] = rh, ["body"] = bodySource });
        }
    }
}

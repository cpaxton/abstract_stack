using System;
using System.IO;
using UnityEngine;
using UnityEngine.XR;

public class TrackingProbe : MonoBehaviour
{
    public OVRCameraRig rig;
    public OVRHand leftHand, rightHand;
    public UnityEngine.UI.Text status;
    public Material jointMaterial;
    public Mesh jointMesh;
    const string BodyPermission = "com.oculus.permission.BODY_TRACKING";
    OVRPlugin.BodyState body;
    OVRPlugin.HandState leftRaw, rightRaw;
    OVRPlugin.BodyJointSet mode = OVRPlugin.BodyJointSet.UpperBody;
    bool started, active, highRequested, leftPinch, rightPinch;
    double lastBodyTime, lastLeftTime, lastRightTime;
    int bodyStateAdvances, leftFresh, rightFresh, leftPinches, rightPinches, bodyMotion, leftMotion, rightMotion;
    Vector3 lastBodyPosition, lastLeftPosition, lastRightPosition;
    bool hadBodyPosition, hadLeftPosition, hadRightPosition;
    float nextReport, nextRetry;
    Transform[] dots = new Transform[84];
    Transform mirror;
    bool displayPlaced;
    Vector3 displayOrigin;
    int recenterCount = -1;
    const double HeadingSettleSeconds = .5;
    const float HeadingToleranceDegrees = 5;
    double headingStableSince = -1;
    Vector3 settlingHeading;
    StreamWriter report;

    void Start()
    {
        Application.runInBackground = true;
        var file = Path.Combine(Application.persistentDataPath, "tracking-probe.jsonl");
        report = new StreamWriter(file, true) { AutoFlush = true };
        Debug.Log("[TrackingProbe] START Unity=" + Application.unityVersion + " MetaCore=209.0.0 OpenXR=1.19.0 OVRPlugin=" + OVRPlugin.version + " runtime=" + OVRPlugin.nativeSDKVersion);
        Debug.Log("[TrackingProbe] Diagnostics file=" + file);
        mirror = new GameObject("Body joint display (estimated pose)").transform;
        mirror.SetParent(rig.trackingSpace, false);
        for (var i = 0; i < dots.Length; i++)
        {
            var dot = new GameObject("BodyJoint" + i, typeof(MeshFilter), typeof(MeshRenderer));
            dot.name = "BodyJoint" + i;
            dot.GetComponent<MeshFilter>().sharedMesh = jointMesh;
            dot.transform.SetParent(mirror, false);
            dot.transform.localScale = Vector3.one * (i < 18 ? .035f : .014f);
            dot.GetComponent<Renderer>().sharedMaterial = jointMaterial;
            dots[i] = dot.transform;
            dot.SetActive(false);
        }
#if UNITY_ANDROID && !UNITY_EDITOR
        if (!UnityEngine.Android.Permission.HasUserAuthorizedPermission(BodyPermission))
            UnityEngine.Android.Permission.RequestUserPermission(BodyPermission);
#endif
    }

    void Update()
    {
        if (!rig || !rig.centerEyeAnchor) return;
        if (Time.unscaledTime >= nextRetry)
        {
            nextRetry = Time.unscaledTime + 5;
            if (!started && PermissionGranted()) StartBody();
        }
        var lg = OVRPlugin.GetHandState(OVRPlugin.Step.Render, OVRPlugin.Hand.HandLeft, ref leftRaw);
        var rg = OVRPlugin.GetHandState(OVRPlugin.Step.Render, OVRPlugin.Hand.HandRight, ref rightRaw);
        TrackHand(lg && leftHand.IsTracked, leftRaw, ref lastLeftTime, ref leftFresh, ref lastLeftPosition, ref hadLeftPosition, ref leftMotion);
        TrackHand(rg && rightHand.IsTracked, rightRaw, ref lastRightTime, ref rightFresh, ref lastRightPosition, ref hadRightPosition, ref rightMotion);
        bool lp = leftHand.IsTracked && leftHand.GetFingerIsPinching(OVRHand.HandFinger.Index);
        bool rp = rightHand.IsTracked && rightHand.GetFingerIsPinching(OVRHand.HandFinger.Index);
        if (lp && !leftPinch) leftPinches++;
        if (rp && !rightPinch) rightPinches++;
        leftPinch = lp; rightPinch = rp;
        active = started && OVRPlugin.GetBodyState4(OVRPlugin.Step.Render, mode, ref body);
        int valid = 0, tracked = 0;
        for (int i = 0; i < dots.Length; i++)
        {
            bool usable = displayPlaced && active && body.JointLocations != null && i < body.JointLocations.Length && body.JointLocations[i].PositionValid;
            dots[i].gameObject.SetActive(usable);
            if (!usable) continue;
            valid++;
            if (body.JointLocations[i].PositionTracked) tracked++;
            dots[i].localPosition = Position(body.JointLocations[i].Pose.Position) - displayOrigin;
        }
        if (active && body.Time > lastBodyTime)
        {
            bodyStateAdvances++; lastBodyTime = body.Time;
            // Elbow motion is more useful evidence than the root following the HMD.
            int elbow = (int)OVRPlugin.BoneId.Body_LeftArmLower;
            if (body.JointLocations != null && elbow < body.JointLocations.Length && body.JointLocations[elbow].PositionValid)
                CountMotion(Position(body.JointLocations[elbow].Pose.Position), ref lastBodyPosition, ref hadBodyPosition, ref bodyMotion);
        }
        if (Time.unscaledTime < nextReport) return;
        nextReport = Time.unscaledTime + 1;
        string text = "QUEST TRACKING PROBE  |  Meta XR 209\n" +
            $"Running {Time.unscaledTime:F0}s  |  XR: {XRSettings.isDeviceActive}  |  headset worn: {(OVRManager.instance != null && OVRManager.instance.isUserPresent)}\n" +
            $"LEFT  tracked: {leftHand.IsTracked}  confidence: {leftHand.HandConfidence}  pinches: {leftPinches}\n" +
            $"RIGHT tracked: {rightHand.IsTracked}  confidence: {rightHand.HandConfidence}  pinches: {rightPinches}\n" +
            $"Hand fresh samples L/R: {leftFresh}/{rightFresh}  motion L/R: {leftMotion}/{rightMotion}\n" +
            $"BODY {mode}: supported={OVRPlugin.bodyTrackingSupported} started={started} active={active}\n" +
            $"Permission: {PermissionGranted()}  confidence: {(active ? body.Confidence : 0):F2}  fidelity: {(active ? body.Fidelity.ToString() : "unavailable")}\n" +
            $"Joints valid/tracked: {valid}/{tracked}  state advances: {bodyStateAdvances}  elbow motion: {bodyMotion}\n" +
            "Room-anchored pose display. Body poses include estimates.\n" +
            "Put controllers down. Show both hands; pinch each index finger.\n" +
            "Raise/bend arms and turn torso. Raw capture runs while this app is active.";
        status.text = text + "\nRAW: " + RawTrackingRecorder.Status;
        var sample = new Sample {
            uptime = Time.unscaledTime, xrActive = XRSettings.isDeviceActive, worn = (OVRManager.instance != null && OVRManager.instance.isUserPresent),
            leftTracked = leftHand.IsTracked, rightTracked = rightHand.IsTracked,
            leftConfidence = leftHand.HandConfidence.ToString(), rightConfidence = rightHand.HandConfidence.ToString(),
            leftFresh = leftFresh, rightFresh = rightFresh, leftMotion = leftMotion, rightMotion = rightMotion,
            leftPinches = leftPinches, rightPinches = rightPinches, bodySupported = OVRPlugin.bodyTrackingSupported,
            bodyStarted = started, bodyActive = active, bodyMode = mode.ToString(), bodyPermission = PermissionGranted(),
            bodyConfidence = active ? body.Confidence : 0, validJoints = valid, trackedJoints = tracked,
            bodyStateAdvances = bodyStateAdvances, elbowMotion = bodyMotion, highFidelityRequestAccepted = highRequested,
            bodyFidelity = active ? body.Fidelity.ToString() : "unavailable", bodyTimestamp = active ? body.Time : 0
        };
        var json = JsonUtility.ToJson(sample);
        Debug.Log("[TrackingProbe] " + json);
        report?.WriteLine(json);
    }
    void LateUpdate()
    {
        if (!rig || !rig.centerEyeAnchor) return;
        int count = OVRPlugin.GetLocalTrackingSpaceRecenterCount();
        if (count != recenterCount)
        {
            recenterCount = count;
            displayPlaced = false;
            headingStableSince = -1;
            mirror.gameObject.SetActive(false);
        }
        if (!displayPlaced)
            TryPlaceDisplay();
    }

    void TryPlaceDisplay()
    {
        const OVRPlugin.Node eye = OVRPlugin.Node.EyeCenter;
        int chest = (int)OVRPlugin.BoneId.Body_Chest;
        if (!active || !Application.isFocused || !OVRManager.instance ||
            !OVRManager.instance.isUserPresent || !OVRManager.hasVrFocus || !OVRManager.hasInputFocus ||
            !OVRPlugin.GetNodePositionTracked(eye) || !OVRPlugin.GetNodeOrientationTracked(eye) ||
            body.JointLocations == null || chest >= body.JointLocations.Length ||
            !body.JointLocations[chest].PositionValid)
        {
            headingStableSince = -1;
            return;
        }

        // The camera rig has updated its anchors by LateUpdate. Tracking can still
        // report a valid heading in the old origin for a few frames.
        var headPosition = rig.trackingSpace.InverseTransformPoint(rig.centerEyeAnchor.position);
        var forward = rig.trackingSpace.InverseTransformDirection(rig.centerEyeAnchor.forward);
        var heading = Vector3.ProjectOnPlane(forward, Vector3.up);
        if (heading.sqrMagnitude < .01f)
        {
            headingStableSince = -1;
            return;
        }
        heading.Normalize();
        double now = Time.realtimeSinceStartupAsDouble;
        if (headingStableSince < 0 || Vector3.Angle(settlingHeading, heading) > HeadingToleranceDegrees)
        {
            settlingHeading = heading;
            headingStableSince = now;
            return;
        }
        if (now - headingStableSince < HeadingSettleSeconds) return;

        var yaw = Quaternion.LookRotation(heading, Vector3.up);
        var chestPosition = Position(body.JointLocations[chest].Pose.Position);
        displayOrigin = new Vector3(chestPosition.x, 0, chestPosition.z);
        var eyeOnFloor = new Vector3(headPosition.x, 0, headPosition.z);
        mirror.localPosition = eyeOnFloor + yaw * new Vector3(0, 0, 2.4f);
        mirror.localRotation = Quaternion.Euler(0, 180, 0);
        // Leave space for the centered body and its moving arms.
        var panel = status.canvas.GetComponent<RectTransform>();
        panel.SetParent(rig.trackingSpace, false);
        panel.localPosition = eyeOnFloor + yaw * new Vector3(-1.25f, Mathf.Max(headPosition.y, 1.2f), 2.4f);
        panel.localRotation = yaw;
        displayPlaced = true;
        mirror.gameObject.SetActive(true);
        Debug.Log($"[TrackingProbe] Centered body at time={now:F3}s recenter={recenterCount} yaw={yaw.eulerAngles.y:F1}");
    }
    void StartBody()
    {
        started = OVRPlugin.StartBodyTracking2(mode);
        highRequested = started && OVRPlugin.RequestBodyTrackingFidelity(OVRPlugin.BodyTrackingFidelity2.High);
        Debug.Log($"[TrackingProbe] StartBody mode={mode} accepted={started} highFidelityAccepted={highRequested}");
    }
    bool PermissionGranted()
    {
#if UNITY_ANDROID && !UNITY_EDITOR
        return UnityEngine.Android.Permission.HasUserAuthorizedPermission(BodyPermission);
#else
        return true;
#endif
    }
    static Vector3 Position(OVRPlugin.Vector3f p) => new Vector3(p.x, p.y, -p.z);
    static void TrackHand(bool valid, OVRPlugin.HandState state, ref double time, ref int fresh, ref Vector3 last, ref bool had, ref int motion)
    {
        if (!valid || state.SampleTimeStamp <= time) return;
        time = state.SampleTimeStamp; fresh++;
        CountMotion(Position(state.RootPose.Position), ref last, ref had, ref motion);
    }
    static void CountMotion(Vector3 position, ref Vector3 last, ref bool had, ref int count)
    {
        if (had && Vector3.Distance(last, position) > .003f) { count++; last = position; }
        if (!had) { last = position; had = true; }
    }
    void OnApplicationPause(bool pause) { report?.Flush(); }
    void OnDestroy() { if (started) OVRPlugin.StopBodyTracking(); report?.Dispose(); }
    [Serializable] class Sample
    {
        public float uptime, bodyConfidence;
        public bool xrActive, worn, leftTracked, rightTracked, bodySupported, bodyStarted, bodyActive, bodyPermission, highFidelityRequestAccepted;
        public string leftConfidence, rightConfidence, bodyMode, bodyFidelity;
        public int leftFresh, rightFresh, leftMotion, rightMotion, leftPinches, rightPinches, validJoints, trackedJoints, bodyStateAdvances, elbowMotion;
        public double bodyTimestamp;
    }
}

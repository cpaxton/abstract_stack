using System;
using System.IO;
using System.Linq;
using UnityEditor;
using UnityEditor.Android;
using System.Xml.Linq;
using UnityEditor.Build;
using UnityEditor.Build.Reporting;
using UnityEditor.SceneManagement;
using UnityEditor.XR.Management;
using UnityEditor.XR.Management.Metadata;
using UnityEngine;
using UnityEngine.Rendering;
using UnityEngine.XR.Management;
using UnityEngine.XR.OpenXR;

public static class ProbeBuilder
{
    public static void SelectAndroidTarget()
    {
        EditorUserBuildSettings.selectedBuildTargetGroup = BuildTargetGroup.Android;
        if (EditorUserBuildSettings.activeBuildTarget != BuildTarget.Android &&
            !EditorUserBuildSettings.SwitchActiveBuildTarget(BuildTargetGroup.Android, BuildTarget.Android))
            throw new Exception("Could not switch the active build target to Android");
    }

    public static void Configure()
    {
        EditorUserBuildSettings.selectedBuildTargetGroup = BuildTargetGroup.Android;
        Directory.CreateDirectory("Assets/Probe");
        Directory.CreateDirectory("Assets/XR");
        PlayerSettings.companyName = "Abstrakt Robots";
        PlayerSettings.productName = "Quest Tracking Probe";
        PlayerSettings.SetApplicationIdentifier(NamedBuildTarget.Android, "org.abstractstack.questtrackingprobe");
        PlayerSettings.SetScriptingBackend(NamedBuildTarget.Android, ScriptingImplementation.IL2CPP);
        PlayerSettings.Android.targetArchitectures = AndroidArchitecture.ARM64;
        PlayerSettings.Android.applicationEntry = AndroidApplicationEntry.GameActivity;
        PlayerSettings.Android.minSdkVersion = AndroidSdkVersions.AndroidApiLevel32;
        PlayerSettings.Android.targetSdkVersion = AndroidSdkVersions.AndroidApiLevelAuto;
        PlayerSettings.SetUseDefaultGraphicsAPIs(BuildTarget.Android, false);
        PlayerSettings.SetGraphicsAPIs(BuildTarget.Android, new[] { GraphicsDeviceType.Vulkan });
        PlayerSettings.colorSpace = ColorSpace.Linear;
        PlayerSettings.runInBackground = true;
        PlayerSettings.Android.forceInternetPermission = false;
        PlayerSettings.bundleVersion = "0.3.5";
        PlayerSettings.Android.bundleVersionCode = 8;
        PlayerSettings.defaultInterfaceOrientation = UIOrientation.LandscapeLeft;
        var input = new SerializedObject(AssetDatabase.LoadAllAssetsAtPath("ProjectSettings/ProjectSettings.asset")[0]);
        var activeInput = input.FindProperty("activeInputHandler");
        if (activeInput != null) { activeInput.intValue = 1; input.ApplyModifiedPropertiesWithoutUndo(); }

        if (!EditorBuildSettings.TryGetConfigObject(XRGeneralSettings.settingsKey, out XRGeneralSettingsPerBuildTarget perTarget))
        {
            perTarget = ScriptableObject.CreateInstance<XRGeneralSettingsPerBuildTarget>();
            AssetDatabase.CreateAsset(perTarget, "Assets/XR/XRGeneralSettings.asset");
            EditorBuildSettings.AddConfigObject(XRGeneralSettings.settingsKey, perTarget, true);
        }
        if (!perTarget.HasSettingsForBuildTarget(BuildTargetGroup.Android)) perTarget.CreateDefaultSettingsForBuildTarget(BuildTargetGroup.Android);
        if (!perTarget.HasManagerSettingsForBuildTarget(BuildTargetGroup.Android)) perTarget.CreateDefaultManagerSettingsForBuildTarget(BuildTargetGroup.Android);
        var general = perTarget.SettingsForBuildTarget(BuildTargetGroup.Android);
        general.InitManagerOnStart = true;
        if (!XRPackageMetadataStore.AssignLoader(general.Manager, "UnityEngine.XR.OpenXR.OpenXRLoader", BuildTargetGroup.Android))
            throw new Exception("Could not assign OpenXR loader");
        var xr = OpenXRSettings.GetSettingsForBuildTargetGroup(BuildTargetGroup.Android);
        if (!xr) throw new Exception("Missing Android OpenXR settings");
        foreach (var feature in xr.GetFeatures<UnityEngine.XR.OpenXR.Features.OpenXRFeature>())
        {
            string type = feature.GetType().Name;
            feature.enabled = type == "MetaXRFeature" || type == "OculusTouchControllerProfile";
            EditorUtility.SetDirty(feature);
            if (feature.enabled) Debug.Log("[ProbeSetup] Enabled " + type);
        }
        xr.renderMode = OpenXRSettings.RenderMode.SinglePassInstanced;
        EditorUtility.SetDirty(xr); EditorUtility.SetDirty(general); EditorUtility.SetDirty(general.Manager);
        var config = OVRProjectConfig.CachedProjectConfig;
        config.targetDeviceTypes = new System.Collections.Generic.List<OVRProjectConfig.DeviceType> { OVRProjectConfig.DeviceType.Quest3 };
        config.handTrackingSupport = OVRProjectConfig.HandTrackingSupport.ControllersAndHands;
        config.bodyTrackingSupport = OVRProjectConfig.FeatureSupport.Supported;
        OVRProjectConfig.CommitProjectConfig(config);
        var runtime = OVRRuntimeSettings.GetRuntimeSettings();
        // SDK 209 defaults newly serialized settings to OpenXR26. This probe's
        // declared raw bind hierarchy and T01 profile deliberately use OVR24.
        runtime.HandSkeletonVersion = OVRHandSkeletonVersion.OVR;
        runtime.BodyTrackingJointSet = OVRPlugin.BodyJointSet.UpperBody;
        runtime.BodyTrackingFidelity = OVRPlugin.BodyTrackingFidelity2.High;
        OVRRuntimeSettings.CommitRuntimeSettings(runtime);

        var scene = EditorSceneManager.NewScene(NewSceneSetup.EmptyScene, NewSceneMode.Single);
        var rigPrefab = AssetDatabase.LoadAssetAtPath<GameObject>("Packages/com.meta.xr.sdk.core/Prefabs/OVRCameraRig.prefab");
        var rig = ((GameObject)PrefabUtility.InstantiatePrefab(rigPrefab)).GetComponent<OVRCameraRig>();
        rig.GetComponent<OVRManager>().trackingOriginType = OVRManager.TrackingOrigin.FloorLevel;
        var camera = rig.centerEyeAnchor.GetComponent<Camera>();
        camera.clearFlags = CameraClearFlags.SolidColor;
        camera.backgroundColor = new Color(.025f,.04f,.075f);
        camera.nearClipPlane = .05f;
        camera.farClipPlane = 30;
        var left = AddHand(rig.leftHandAnchor, 0);
        var right = AddHand(rig.rightHandAnchor, 1);
        var light = new GameObject("Room light").AddComponent<Light>();
        light.type = LightType.Directional; light.intensity = 1.2f;
        light.transform.rotation = Quaternion.Euler(45,-30,0);
        var material = new Material(Shader.Find("Universal Render Pipeline/Unlit"));
        material.color = new Color(.1f,.9f,.7f);
        AssetDatabase.CreateAsset(material, "Assets/Probe/JointMaterial.mat");
        var floor = GameObject.CreatePrimitive(PrimitiveType.Plane);
        floor.name = "Floor reference"; floor.transform.localScale = Vector3.one * .8f;
        var fm = new Material(Shader.Find("Universal Render Pipeline/Lit"));
        fm.color = new Color(.07f,.10f,.14f);
        AssetDatabase.CreateAsset(fm,"Assets/Probe/FloorMaterial.mat"); floor.GetComponent<Renderer>().sharedMaterial = fm;

        var canvasObject = new GameObject("Tracking status",typeof(RectTransform),typeof(Canvas),typeof(UnityEngine.UI.CanvasScaler));
        canvasObject.transform.SetParent(rig.trackingSpace,false);
        var canvas = canvasObject.GetComponent<Canvas>(); canvas.renderMode = RenderMode.WorldSpace; canvas.worldCamera = camera;
        var cr = canvasObject.GetComponent<RectTransform>(); cr.sizeDelta = new Vector2(1100,660);
        cr.localScale = Vector3.one * .0011f; cr.localPosition = new Vector3(-.13f,.12f,1.4f);
        var background = canvasObject.AddComponent<UnityEngine.UI.Image>(); background.color = new Color(.015f,.025f,.045f,.94f);
        var textObject = new GameObject("Readings",typeof(RectTransform),typeof(UnityEngine.UI.Text));
        textObject.transform.SetParent(cr,false);
        var tr = textObject.GetComponent<RectTransform>(); tr.anchorMin = Vector2.zero; tr.anchorMax = Vector2.one;
        tr.offsetMin = new Vector2(28,20); tr.offsetMax = new Vector2(-28,-20);
        var text = textObject.GetComponent<UnityEngine.UI.Text>();
        text.font = Resources.GetBuiltinResource<Font>("LegacyRuntime.ttf"); text.fontSize = 28; text.color = Color.white;
        text.alignment = TextAnchor.UpperLeft; text.supportRichText = false;
        text.horizontalOverflow = HorizontalWrapMode.Wrap; text.verticalOverflow = VerticalWrapMode.Truncate;
        text.text = "QUEST TRACKING PROBE\nStarting tracking...";
        var probe = new GameObject("TrackingProbe").AddComponent<TrackingProbe>();
        probe.gameObject.AddComponent<RawTrackingRecorder>();
        probe.rig = rig; probe.leftHand = left; probe.rightHand = right; probe.status = text; probe.jointMaterial = material;
        var sphere = GameObject.CreatePrimitive(PrimitiveType.Sphere);
        probe.jointMesh = sphere.GetComponent<MeshFilter>().sharedMesh;
        UnityEngine.Object.DestroyImmediate(sphere);
        EditorSceneManager.SaveScene(scene,"Assets/Probe/TrackingProbe.unity");
        EditorBuildSettings.scenes = new[]{ new EditorBuildSettingsScene("Assets/Probe/TrackingProbe.unity",true) };
        AssetDatabase.SaveAssets();
        Debug.Log("[ProbeSetup] CONFIGURED Meta XR 209 / OpenXR / Android ARM64 / API32 minimum");
    }
    static OVRHand AddHand(Transform anchor,int side)
    {
        var prefab = AssetDatabase.LoadAssetAtPath<GameObject>("Packages/com.meta.xr.sdk.core/Prefabs/OVRHandPrefab.prefab");
        var go = (GameObject)PrefabUtility.InstantiatePrefab(prefab);
        go.transform.SetParent(anchor,false); go.name = side == 0 ? "LeftHand" : "RightHand";
        var hand = go.GetComponent<OVRHand>();
        var hs = new SerializedObject(hand); hs.FindProperty("HandType").intValue = side; hs.ApplyModifiedPropertiesWithoutUndo();
        var skeleton = new SerializedObject(go.GetComponent<OVRSkeleton>());
        skeleton.FindProperty("_skeletonType").intValue = side; skeleton.ApplyModifiedPropertiesWithoutUndo();
        hand.OnValidate();
        return hand;
    }
    public static void Build()
    {
        // Meta's build processors inspect selectedBuildTargetGroup independently
        // of BuildPlayerOptions.target; mismatch silently omits tracking permissions.
        EditorUserBuildSettings.selectedBuildTargetGroup = BuildTargetGroup.Android;
        if (EditorUserBuildSettings.activeBuildTarget != BuildTarget.Android)
            throw new Exception("Select Android first with ProbeBuilder.SelectAndroidTarget, then rebuild after import");
        if (OVRRuntimeSettings.GetRuntimeSettings().HandSkeletonVersion != OVRHandSkeletonVersion.OVR)
            throw new Exception("This probe requires the declared OVR24 hand profile; run Configure first");
        var general = XRGeneralSettingsPerBuildTarget.XRGeneralSettingsForBuildTarget(BuildTargetGroup.Android);
        var xr = OpenXRSettings.GetSettingsForBuildTargetGroup(BuildTargetGroup.Android);
        if (general?.Manager == null || !general.Manager.activeLoaders.Any(l => l is OpenXRLoader) ||
            xr == null || !xr.GetFeatures<UnityEngine.XR.OpenXR.Features.OpenXRFeature>().Any(f => f.enabled && f.GetType().Name == "MetaXRFeature"))
            throw new Exception("Android OpenXR loader and MetaXRFeature must be enabled before building");
        Directory.CreateDirectory("Builds");
        var report = BuildPipeline.BuildPlayer(new BuildPlayerOptions {
            scenes = new[]{ "Assets/Probe/TrackingProbe.unity" }, locationPathName = "Builds/QuestTrackingProbe.apk",
            target = BuildTarget.Android, options = BuildOptions.Development
        });
        Debug.Log("[ProbeBuild] " + report.summary.result + " errors=" + report.summary.totalErrors);
        if (report.summary.result != BuildResult.Succeeded) throw new Exception("Probe APK build failed");
    }
}

// Fail before Gradle packaging if SDK preprocessing was skipped. Never inject
// permissions here: the Meta SDK must generate its complete supported manifest.
public sealed class ProbeManifestValidation : IPostGenerateGradleAndroidProject
{
    public int callbackOrder => int.MaxValue;
    public void OnPostGenerateGradleAndroidProject(string path)
    {
        var manifest = XDocument.Load(Path.Combine(path, "src/main/AndroidManifest.xml"));
        XNamespace android = "http://schemas.android.com/apk/res/android";
        string[] permissions = { "com.oculus.permission.HAND_TRACKING", "com.oculus.permission.BODY_TRACKING" };
        string[] features = { "oculus.software.handtracking", "com.oculus.software.body_tracking" };
        foreach (string name in permissions)
            if (!manifest.Root.Elements("uses-permission").Any(e => (string)e.Attribute(android + "name") == name))
                throw new BuildFailedException("Quest tracking permission missing from generated manifest: " + name);
        foreach (string name in features)
            if (!manifest.Root.Elements("uses-feature").Any(e => (string)e.Attribute(android + "name") == name))
                throw new BuildFailedException("Quest tracking feature missing from generated manifest: " + name);
        if (!manifest.Descendants("category").Any(e => (string)e.Attribute(android + "name") == "com.oculus.intent.category.VR"))
            throw new BuildFailedException("Quest immersive VR activity category missing from generated manifest");
        Debug.Log("[ProbeBuild] Verified Meta hand/body manifest declarations and immersive VR category");
    }
}

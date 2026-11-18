using System;
using System.IO;
using System.Linq;
using AbstractStack.QuestStereo;
using UnityEditor;
using UnityEditor.Build;
using UnityEditor.Build.Reporting;
using UnityEditor.SceneManagement;
using UnityEditor.XR.Management;
using UnityEngine;
using UnityEngine.Rendering;
using UnityEngine.XR.Management;
using UnityEngine.XR.OpenXR;

public static class QuestStereoBuilder
{
    const string ScenePath = "Assets/QuestStereo/QuestStereo.unity";
    public static void Configure()
    {
        ProbeBuilder.SelectAndroidTarget();
        EditorUserBuildSettings.selectedBuildTargetGroup = BuildTargetGroup.Android;
        PlayerSettings.companyName = "Abstrakt Robots";
        PlayerSettings.productName = "Quest Stereo Window";
        PlayerSettings.SetApplicationIdentifier(NamedBuildTarget.Android, "org.abstractstack.queststereoviewer");
        PlayerSettings.bundleVersion = "0.1.0"; PlayerSettings.Android.bundleVersionCode = 1;
        PlayerSettings.SetScriptingBackend(NamedBuildTarget.Android, ScriptingImplementation.IL2CPP);
        PlayerSettings.Android.targetArchitectures = AndroidArchitecture.ARM64;
        PlayerSettings.Android.applicationEntry = AndroidApplicationEntry.GameActivity;
        PlayerSettings.Android.minSdkVersion = AndroidSdkVersions.AndroidApiLevel32;
        PlayerSettings.Android.targetSdkVersion = AndroidSdkVersions.AndroidApiLevelAuto;
        PlayerSettings.Android.forceInternetPermission = true;
        PlayerSettings.SetUseDefaultGraphicsAPIs(BuildTarget.Android, false);
        PlayerSettings.SetGraphicsAPIs(BuildTarget.Android, new[] { GraphicsDeviceType.Vulkan });
        PlayerSettings.colorSpace = ColorSpace.Linear; PlayerSettings.runInBackground = true;
        var scene = EditorSceneManager.NewScene(NewSceneSetup.EmptyScene, NewSceneMode.Single);
        var rigPrefab = AssetDatabase.LoadAssetAtPath<GameObject>("Packages/com.meta.xr.sdk.core/Prefabs/OVRCameraRig.prefab");
        var rig = ((GameObject)PrefabUtility.InstantiatePrefab(rigPrefab)).GetComponent<OVRCameraRig>();
        rig.GetComponent<OVRManager>().trackingOriginType = OVRManager.TrackingOrigin.FloorLevel;
        var camera = rig.centerEyeAnchor.GetComponent<Camera>();
        camera.clearFlags = CameraClearFlags.SolidColor; camera.backgroundColor = new Color(.025f, .035f, .055f);
        camera.nearClipPlane = .05f; camera.farClipPlane = 20;
        var root = new GameObject("Room Anchored Stereo Window");
        root.transform.position = new Vector3(0, 1.4f, 1.5f);
        var quad = GameObject.CreatePrimitive(PrimitiveType.Quad); quad.name = "Stereo Image"; quad.transform.SetParent(root.transform, false);
        UnityEngine.Object.DestroyImmediate(quad.GetComponent<Collider>());
        var material = new Material(Shader.Find("QuestTeleop/RoomStereo"));
        string materialPath = "Assets/QuestStereo/RoomStereo.mat";
        if (AssetDatabase.LoadAssetAtPath<Material>(materialPath)) AssetDatabase.DeleteAsset(materialPath);
        AssetDatabase.CreateAsset(material, materialPath); quad.GetComponent<MeshRenderer>().sharedMaterial = material;
        var viewer = root.AddComponent<StereoViewer>(); viewer.viewerCamera = camera; viewer.window = quad.GetComponent<MeshRenderer>();
        var canvasObject = new GameObject("Video status", typeof(RectTransform), typeof(Canvas), typeof(UnityEngine.UI.CanvasScaler));
        canvasObject.transform.SetParent(root.transform, false);
        var canvas = canvasObject.GetComponent<Canvas>(); canvas.renderMode = RenderMode.WorldSpace; canvas.worldCamera = camera;
        var rect = canvasObject.GetComponent<RectTransform>(); rect.sizeDelta = new Vector2(1200, 250);
        rect.localScale = Vector3.one * .001f; rect.localPosition = new Vector3(0, -.60f, -.005f);
        var background = canvasObject.AddComponent<UnityEngine.UI.Image>(); background.color = new Color(.01f, .02f, .035f, 1); background.raycastTarget = false;
        // Follow the existing probe's built-in font so this diagnostic has no
        // generated TMP font dependency and remains readable on first launch.
        var textObject = new GameObject("Video telemetry", typeof(RectTransform), typeof(UnityEngine.UI.Text));
        textObject.transform.SetParent(rect, false);
        var tr = textObject.GetComponent<RectTransform>(); tr.anchorMin = Vector2.zero; tr.anchorMax = Vector2.one; tr.offsetMin = new Vector2(20, 12); tr.offsetMax = new Vector2(-20, -12);
        var text = textObject.GetComponent<UnityEngine.UI.Text>(); text.font = Resources.GetBuiltinResource<Font>("LegacyRuntime.ttf"); text.fontSize = 30; text.color = Color.white;
        text.alignment = TextAnchor.UpperLeft; text.supportRichText = false; text.raycastTarget = false; text.text = "QUEST STEREO WINDOW\nStarting..."; viewer.status = text;
        EditorSceneManager.SaveScene(scene, ScenePath);
        EditorBuildSettings.scenes = new[] { new EditorBuildSettingsScene(ScenePath, true) };
        AssetDatabase.SaveAssets();
        Debug.Log("[QuestStereoBuild] Configured separate viewer / Android OpenXR / internet enabled");
    }
    public static void Build()
    {
        EditorUserBuildSettings.selectedBuildTargetGroup = BuildTargetGroup.Android;
        if (EditorUserBuildSettings.activeBuildTarget != BuildTarget.Android) throw new Exception("Select Android before building");
        var general = XRGeneralSettingsPerBuildTarget.XRGeneralSettingsForBuildTarget(BuildTargetGroup.Android);
        var xr = OpenXRSettings.GetSettingsForBuildTargetGroup(BuildTargetGroup.Android);
        if (general?.Manager == null || !general.Manager.activeLoaders.Any(l => l is OpenXRLoader) || xr == null ||
            !xr.GetFeatures<UnityEngine.XR.OpenXR.Features.OpenXRFeature>().Any(f => f.enabled && f.GetType().Name == "MetaXRFeature"))
            throw new Exception("Android OpenXR loader / Meta XR feature required");
        if (!PlayerSettings.Android.forceInternetPermission) throw new Exception("Internet permission required");
        if (PlayerSettings.GetApplicationIdentifier(NamedBuildTarget.Android) != "org.abstractstack.queststereoviewer") throw new Exception("Run QuestStereoBuilder.Configure before building");
        Directory.CreateDirectory("Builds");
        var report = BuildPipeline.BuildPlayer(new BuildPlayerOptions { scenes = new[] { ScenePath }, locationPathName = "Builds/QuestStereoWindow.apk", target = BuildTarget.Android, options = BuildOptions.Development });
        Debug.Log("[QuestStereoBuild] " + report.summary.result + " errors=" + report.summary.totalErrors);
        if (report.summary.result != BuildResult.Succeeded) throw new Exception("Stereo viewer APK failed");
    }
}

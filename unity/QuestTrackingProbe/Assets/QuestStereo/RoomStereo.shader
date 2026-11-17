Shader "QuestTeleop/RoomStereo"
{
    Properties
    {
        _LeftTex("Left eye", 2D) = "black" {}
        _RightTex("Right eye", 2D) = "black" {}
        _Stale("Stale", Float) = 1
        _DebugEye("Editor test eye (-1 uses XR)", Float) = -1
    }
    SubShader
    {
        Tags { "RenderType"="Opaque" "RenderPipeline"="UniversalPipeline" }
        Pass
        {
            Cull Off ZWrite On
            HLSLPROGRAM
            #pragma vertex Vert
            #pragma fragment Frag
            #pragma multi_compile_instancing
            #include "Packages/com.unity.render-pipelines.universal/ShaderLibrary/Core.hlsl"
            struct Attributes { float4 positionOS: POSITION; float2 uv: TEXCOORD0; UNITY_VERTEX_INPUT_INSTANCE_ID };
            struct Varyings { float4 positionCS: SV_POSITION; float2 uv: TEXCOORD0; UNITY_VERTEX_OUTPUT_STEREO };
            TEXTURE2D(_LeftTex); SAMPLER(sampler_LeftTex);
            TEXTURE2D(_RightTex); SAMPLER(sampler_RightTex);
            CBUFFER_START(UnityPerMaterial)
            float _Stale, _DebugEye;
            CBUFFER_END
            Varyings Vert(Attributes input)
            {
                Varyings output;
                UNITY_SETUP_INSTANCE_ID(input);
                UNITY_INITIALIZE_VERTEX_OUTPUT_STEREO(output);
                output.positionCS = TransformObjectToHClip(input.positionOS.xyz);
                output.uv = input.uv;
                return output;
            }
            half4 Frag(Varyings input): SV_Target
            {
                UNITY_SETUP_STEREO_EYE_INDEX_POST_VERTEX(input);
                float eye = _DebugEye >= 0 ? _DebugEye : unity_StereoEyeIndex;
                half4 color = eye < .5 ? SAMPLE_TEXTURE2D(_LeftTex, sampler_LeftTex, input.uv) : SAMPLE_TEXTURE2D(_RightTex, sampler_RightTex, input.uv);
                // The last complete pair remains visible with an unmistakable stale border.
                if (_Stale > .5 && (input.uv.x < .015 || input.uv.x > .985 || input.uv.y < .02 || input.uv.y > .98)) return half4(1, .05, .01, 1);
                return half4(color.rgb, 1);
            }
            ENDHLSL
        }
    }
}

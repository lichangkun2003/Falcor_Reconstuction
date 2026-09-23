#pragma once
#include "VoxelizationBase.h"
#include <Rendering/Lights/EnvMapSampler.h>
#include <Core/Pass/FullScreenPass.h>
#include <fstream>
#include <filesystem>

using namespace Falcor;

class RayMarchingPass : public RenderPass
{
public:
    FALCOR_PLUGIN_CLASS(RayMarchingPass, "RayMarchingPass", "Insert pass description here.");

    static ref<RayMarchingPass> create(ref<Device> pDevice, const Properties& props) { return make_ref<RayMarchingPass>(pDevice, props); }

    RayMarchingPass(ref<Device> pDevice, const Properties& props);

    virtual RenderPassReflection reflect(const CompileData& compileData) override;
    virtual void execute(RenderContext* pRenderContext, const RenderData& renderData) override;
    virtual void compile(RenderContext* pRenderContext, const CompileData& compileData) override;
    virtual void renderUI(Gui::Widgets& widget) override;
    virtual void setScene(RenderContext* pRenderContext, const ref<Scene>& pScene) override;
    virtual bool onMouseEvent(const MouseEvent& mouseEvent) override;

private:
    ref<Scene> mpScene;
    ref<FullScreenPass> mpFullScreenPass;
    ref<FullScreenPass> mpDisplayNDFPass;
    ref<Sampler> mpPointSampler;
    ref<Buffer> mSelectedVoxel;
    std::unique_ptr<EnvMapSampler> mpEnvMapSampler;

    GridData& gridData;
    uint mDrawMode;
    uint mMaxBounce;
    float mShadowBias100;
    float mMinPdf100;
    float mTrasmittanceThreshold100;
    bool mCheckEllipsoid;
    bool mCheckVisibility;
    bool mCheckCoverage;
    bool mUseMipmap;
    bool mUseEmissiveLight;
    bool mDebug;
    bool mRenderBackGround;
    float3 mClearColor;

    bool mDisplayNDF;
    float2 mSelectedUV;
    uint2 mSelectedPixel;
    uint mSelectedResolution;

    bool mOptionsChanged;
    uint mFrameIndex;
    uint2 mOutputResolution;

    // ------------------------------------------------------------------
    // 烘焙：把 Voxelization 的 ABSDF 表达转成 VoxelReconstructionNoLightTransport 的表达。
    // 假设纯白天光、只算直接光。输出重建端 v1 格式的 .bin。
    // ------------------------------------------------------------------
    struct BakeState
    {
        bool requested = false;
        // logit 之前把 coverage 夹到的边界，避免 log(0)
        float coverageEps = 1e-3f;
        std::string status = "Not baked";
        std::string outputDirectory;
        ref<ComputePass> pass;
        ref<Buffer> output;
    };

    void bakeReconstruction(RenderContext* pRenderContext, const RenderData& renderData);
    void renderBakeUI(Gui::Widgets& widget);

    BakeState mBake;
};

#pragma once
#include "Falcor.h"
#include "RenderGraph/RenderPass.h"

using namespace Falcor;

/** One-shot RGB PNG export of the exact two FLIP inputs and its display output. */
class ReconstructionResultPass : public RenderPass
{
public:
    FALCOR_PLUGIN_CLASS(ReconstructionResultPass, "ReconstructionResultPass", "Save reconstruction, reference and FLIP images together.");
    static ref<ReconstructionResultPass> create(ref<Device> device, const Properties& props)
    {
        return make_ref<ReconstructionResultPass>(device, props);
    }
    ReconstructionResultPass(ref<Device> device, const Properties& props);
    Properties getProperties() const override;
    RenderPassReflection reflect(const CompileData& compileData) override;
    void execute(RenderContext* context, const RenderData& data) override;
    void renderUI(Gui::Widgets& widget) override;
    void requestSave() { mSaveRequested = true; }
    const std::string& getStatus() const { return mStatus; }

private:
    void saveImages(RenderContext* context, const RenderData& data);
    std::filesystem::path mOutputDirectory;
    std::string mReconstructionPass = "VoxelReconstruction";
    std::string mFlipPass = "FLIPPass";
    std::string mSceneName;
    uint32_t mCameraIndex = 0;
    bool mReady = false;
    bool mSaveRequested = false;
    std::string mStatus;
    ref<Texture> mpPngTexture;
};

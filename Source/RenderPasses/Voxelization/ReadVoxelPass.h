#pragma once
#include "VoxelizationBase.h"
#include "Math/SphericalHarmonics.slang"
#include <fstream>
#include <filesystem>

using namespace Falcor;

class ReadVoxelPass : public RenderPass
{
public:
    FALCOR_PLUGIN_CLASS(ReadVoxelPass, "ReadVoxelPass", "Insert pass description here.");

    static ref<ReadVoxelPass> create(ref<Device> pDevice, const Properties& props) { return make_ref<ReadVoxelPass>(pDevice, props); }

    ReadVoxelPass(ref<Device> pDevice, const Properties& props);
    virtual RenderPassReflection reflect(const CompileData& compileData) override;
    virtual void execute(RenderContext* pRenderContext, const RenderData& renderData) override;
    virtual void compile(RenderContext* pRenderContext, const CompileData& compileData) override;
    virtual void renderUI(Gui::Widgets& widget) override;
    virtual bool onMouseEvent(const MouseEvent& mouseEvent) override { return false; }
    virtual bool onKeyEvent(const KeyboardEvent& keyEvent) override { return false; }
    virtual void setScene(RenderContext* pRenderContext, const ref<Scene>& pScene) override;

private:
    bool tryRead(std::ifstream& f, size_t& offset, size_t bytes, void* dst, size_t fileSize);
    GridData& gridData;

    ref<ComputePass> mPreparePass;
    ref<Device> mpDevice;
    ref<Scene> mpScene;
    ref<Buffer> mpVoxelDataBuffer;
    std::vector<std::filesystem::path> filePaths;

    uint selectedFile;

    bool mComplete;
    bool mOptionsChanged;

    //Test

};

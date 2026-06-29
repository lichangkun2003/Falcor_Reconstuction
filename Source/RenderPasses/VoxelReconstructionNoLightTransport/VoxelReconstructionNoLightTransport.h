/***************************************************************************
 # Copyright (c) 2015-23, NVIDIA CORPORATION. All rights reserved.
 #
 # Redistribution and use in source and binary forms, with or without
 # modification, are permitted provided that the following conditions
 # are met:
 #  * Redistributions of source code must retain the above copyright
 #    notice, this list of conditions and the following disclaimer.
 #  * Redistributions in binary form must reproduce the above copyright
 #    notice, this list of conditions and the following disclaimer in the
 #    documentation and/or other materials provided with the distribution.
 #  * Neither the name of NVIDIA CORPORATION nor the names of its
 #    contributors may be used to endorse or promote products derived
 #    from this software without specific prior written permission.
 #
 # THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS "AS IS" AND ANY
 # EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 # IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 # PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 # CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 # EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 # PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 # PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 # OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 # (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 # OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 **************************************************************************/
#pragma once
#include "Falcor.h"
#include "RenderGraph/RenderPass.h"
#include "Utils/Debug/PixelDebug.h"
#include "Core/Pass/FullScreenPass.h"
#include "RenderGraph/RenderPassStandardFlags.h"
#include "iostream"
#include <fstream>
#include <filesystem>
#include <iomanip>
#include <sstream>

#include "Defines.h"
#include "Voxel/VoxelData.slang"
#include "Voxel/VoxelGrid.slang"
#include "Voxel/ABSDF.slang"

#include "PathRecord.slang"
#include "GradRecord.slang"

using namespace Falcor;

namespace 
{
const std::string ReflectTypesShaderFilePath = "RenderPasses/VoxelReconstructionNoLightTransport/Shader/ReflectTypes.cs.slang";
const std::string ProcessXuDataShaderFilePath = "RenderPasses/VoxelReconstructionNoLightTransport/Shader/ProcessXuData.cs.slang";
const std::string RayMarchingShaderFilePath = "RenderPasses/VoxelReconstructionNoLightTransport/Shader/RayMarchingPass.ps.slang";
const std::string LossPassShaderFilePath = "RenderPasses/VoxelReconstructionNoLightTransport/Shader/LossPass.cs.slang";
const std::string GradientPassShaderFilePath = "RenderPasses/VoxelReconstructionNoLightTransport/Shader/GradientPass.cs.slang";
const std::string UpdatePassShaderFilePath = "RenderPasses/VoxelReconstructionNoLightTransport/Shader/UpdatePass.cs.slang";
const std::string ReduceTexturePassShaderFilePath = "RenderPasses/VoxelReconstructionNoLightTransport/Shader/ReduceTexturePass.cs.slang";
const std::string ReduceBufferPassShaderFilePath = "RenderPasses/VoxelReconstructionNoLightTransport/Shader/ReduceBufferPass.cs.slang";

inline std::string kGBuffer = "gBuffer";
inline std::string kVBuffer = "vBuffer";
inline std::string kPBuffer = "pBuffer";
inline std::string kBlockMap = "blockMap";
inline std::string kOutputColor = "color";

inline std::string ReferenceImageDir = "D:/lck/vs/Reconstruction_Input/lego_white_256";
inline std::string ReferenceCameraFile = "D:/lck/vs/Reconstruction_Input/lego_white_256/camera_params.txt";
inline std::string ReconstructionDataDir = "D:/lck/vs/Reconstruction_Output";
} // namespace VoxelPrime

class VoxelReconstructionNoLightTransport : public RenderPass
{
public:
    FALCOR_PLUGIN_CLASS(VoxelReconstructionNoLightTransport, "VoxelReconstructionNoLightTransport", "Insert pass description here.");

    static ref<VoxelReconstructionNoLightTransport> create(ref<Device> pDevice, const Properties& props)
    {
        return make_ref<VoxelReconstructionNoLightTransport>(pDevice, props);
    }

    VoxelReconstructionNoLightTransport(ref<Device> pDevice, const Properties& props);

    virtual Properties getProperties() const override;
    virtual RenderPassReflection reflect(const CompileData& compileData) override;
    virtual void compile(RenderContext* pRenderContext, const CompileData& compileData) override {}
    virtual void execute(RenderContext* pRenderContext, const RenderData& renderData) override;
    virtual void renderUI(Gui::Widgets& widget) override;
    virtual void setScene(RenderContext* pRenderContext, const ref<Scene>& pScene) override;
    virtual bool onMouseEvent(const MouseEvent& mouseEvent) override { return false; }
    virtual bool onKeyEvent(const KeyboardEvent& keyEvent) override { return false; }

    void beginFrame(RenderContext* pRenderContext, bool forceReset = false);
    void endFrame(RenderContext* pRenderContext);


    void setupGridResouce(RenderContext* pRenderContext, bool forceReset);
    void proccessXuData(RenderContext* pRenderContext, const RenderData& renderData);
    void UpdateVoxelGrid(ref<Scene> scene, uint voxelResolution);

    void createRayMarchingPassResource(RenderContext* pRenderContext);
    void rayMarchingPass(RenderContext* pRenderContext, const RenderData& renderData);

    void createLossPassResource(RenderContext* pRenderContext);
    void runLossPass(RenderContext* pRenderContext, const RenderData& renderData);

    void createGradientPassResource(RenderContext* pRenderContext);
    void runGradientPass(RenderContext* pRenderContext, const RenderData& renderData);

    void createUpdatePassResource(RenderContext* pRenderContext);
    void runUpdatePass(RenderContext* pRenderContext, const RenderData& renderData);
    void renderUIUpdatePass(Gui::Widgets& widget);

    void createReducePassResource(RenderContext* pRenderContext);
    void runReducePass(RenderContext* pRenderContext, const RenderData& renderData);

    void loadReferenceImages();
    bool loadReferenceCamerasFromFile(const std::string& cameraFile);

    void startReconstruction();
    void stopReconstruction();

    std::filesystem::path getDefaultReconstructionSavePath() const;
    std::string getOptimizedParamTag() const;
    void saveReconstruction(RenderContext* pRenderContext);
    void loadReconstruction(RenderContext* pRenderContext, const std::filesystem::path& path);
    void refreshReconstructionFileList();

    struct GridResources
    {
        ref<Buffer> gridDataBuffer;
        ref<Texture> blockOM;
        GridData gridData;
    };

    struct RayMarchingPass
    {
        bool mOptionsChanged;
        uint mFrameIndex;
        uint2 mOutputResolution;
        float3 mClearColor;
        bool mCheckPrimitive;
        float mShadowBias100;
        uint mDrawMode;
        bool mRenderBackGround;

        ref<FullScreenPass> mpFullScreenPass;
        ref<FullScreenPass> mpDisplayNDFPass;
        ref<Sampler> mpPointSampler;
        void init()
        {
            mpFullScreenPass = nullptr;
            mpDisplayNDFPass = nullptr;
            mpPointSampler = nullptr;

            mOptionsChanged = false;
            mFrameIndex = 0;
            mOutputResolution = uint2(1920, 1080);
            mClearColor = float3(0);
            mCheckPrimitive = true;
            mShadowBias100 = 0.01f;
            mDrawMode = 0;
            mRenderBackGround = true;
        }
    };

    struct LossPass
    {
        uint mView;
        ref<ComputePass> mpComputePass;
        ref<Texture> lossBuffer;
        ref<Texture> dL_dColor;
        void init()
        {
            mView = 0;
            mpComputePass = nullptr;
            dL_dColor = nullptr;
            lossBuffer = nullptr;
        }
    };

    struct GradientPass
    {
        ref<ComputePass> mpComputePass;
        ref<Buffer> gradBuffer;
        float geometryGradClamp;
        float geometryTau;

        void init()
        {
            gradBuffer = nullptr;
            mpComputePass = nullptr;
            geometryTau = 0.05f;
            geometryGradClamp = 5.0f;
        }
    };

    struct UpdatePass
    {
        ref<ComputePass> mpComputePass;

        // 是否按当前 voxel 命中的 pixel 数做平均
        bool mUseGradCountNormalize;

        // 全局梯度缩放，第一版可以设为 1.0
        float mGradScale;

        float mLrRadiance;
        float mLrCenter;
        float mLrB;

        void init()
        {
            mpComputePass = nullptr;

            mUseGradCountNormalize = true;
            mGradScale = 1.0f;

            mLrRadiance = 0.005f;
            
            mLrCenter = 0.0f;
            mLrB = 0.0f;
        }
    };

    struct OptimizerParams
    {
        bool isRunning = false;

        // 控制一次优化过程
        uint32_t maxIteration = 100;
        uint32_t currentIteration = 0;

        // 每次 iteration 使用多少个 camera/view
        uint32_t viewsPerIteration = 300;
        uint32_t currentView = 0;

        void reset()
        {
            currentIteration = 0;
            currentView = 0;
            isRunning = false;
        }
    };

    struct ReduceLossPass
    {
        ref<Buffer> mpReduceBufferA;
        ref<Buffer> mpReduceBufferB;
        ref<Buffer> mpTotalLossReadback;
        ref<ComputePass> mpReduceTexturePass;
        ref<ComputePass> mpReduceBufferPass;

        float meanLoss = 0.0f;
    };

private:
    ref<Device> mpDevice;
    ref<Scene> mpScene;
    std::unique_ptr<PixelDebug> mpPixelDebug;

    // Parameters
    uint mFrameCount = 0;
    uint2 mFrameDim;
    float2 mInvFrameDim;
    uint mVoxelResolution = GRID_RESOLUTION; // X,Y,Z三个方向中，最长的边被划分的体素数量

    OptimizerParams mOptimizerParams;

    // Passes
    ref<ComputePass> mpReflectTypes;
    ref<ComputePass> mpProcessXuDataPass;
    GradientPass mGradientPass;
    UpdatePass mUpdatePass;
    LossPass mLossPass;
    ReduceLossPass mReduceLossPass;
    

    // Grid
    GridResources mGridResources;    // cpu中的对应gpu中的资源，变量赋值，buffer绑定
    ref<ParameterBlock> mpGridBlock; // gpu的block

    // RayMarchingPass
    RayMarchingPass mRayMarchingPass;
    uint3 MinFactor = uint3(1, 1, 1);

    // Voxel Optimization
    std::vector<ref<Texture>> mReferenceImages;
    std::vector<ref<Camera>> mReferenceCameras;
    ref<Buffer> mpPathRecordBuffer;


    // UI
    bool mOptionsChanged = false;
    bool mEnableReconstruction = false;
    bool mInitVoxelData = false;
    uint testIndex = 0;
    bool mSaveReconstructionRequested = false;
    bool mLoadReconstructionRequested = false;
    bool mReconstructionFileListDirty = true;
    float mLrCenterScale = 0.0f; 
    float mLrBScale = 0.0f; 
    std::vector<std::filesystem::path> mReconstructionFilePaths;
    uint32_t mSelectedReconstructionFile = 0;
    std::string mReconstructionNameTag = "";
};


inline std::string ToString(float3 v)
{
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(4);
    oss << "(" << v.x << ", " << v.y << ", " << v.z << ")";
    return oss.str();
}
inline std::string ToString(int2 v)
{
    std::ostringstream oss;
    oss << "(" << v.x << ", " << v.y << ")";
    return oss.str();
}
inline std::string ToString(int3 v)
{
    std::ostringstream oss;
    oss << "(" << v.x << ", " << v.y << ", " << v.z << ")";
    return oss.str();
}

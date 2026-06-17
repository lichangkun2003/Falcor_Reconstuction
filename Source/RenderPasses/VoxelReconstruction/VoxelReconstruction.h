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
#include <Rendering/Lights/EnvMapSampler.h>
#include "iostream"
#include <fstream>
#include <filesystem>
#include <iomanip>
#include <sstream>

namespace fs = std::filesystem;

#include "Defines.h"
#include "Voxel/VoxelData.slang"
#include "Voxel/VoxelGrid.slang"
#include "Voxel/ABSDF.slang"

#include "Voxel/Shading.slang"
#include "PathRecord.slang"
#include "GradRecord.slang"

using namespace Falcor;

// 常用命名（shader路径）
namespace VoxelPrime
{
const std::string ReflectTypesShaderFilePath = "RenderPasses/VoxelReconstruction/Shader/ReflectTypes.cs.slang";
const std::string ProcessXuDataShaderFilePath = "RenderPasses/VoxelReconstruction/Shader/ProcessXuData.cs.slang";
const std::string RayMarchingShaderFilePath = "RenderPasses/VoxelReconstruction/Shader/RayMarchingPass.ps.slang";
const std::string LossPassShaderFilePath = "RenderPasses/VoxelReconstruction/Shader/LossPass.cs.slang";
const std::string GradientPassShaderFilePath = "RenderPasses/VoxelReconstruction/Shader/GradientPass.cs.slang";
const std::string UpdatePassShaderFilePath = "RenderPasses/VoxelReconstruction/Shader/UpdatePass.cs.slang";
const std::string ReduceTexturePassShaderFilePath = "RenderPasses/VoxelReconstruction/Shader/ReduceTexturePass.cs.slang";
const std::string ReduceBufferPassShaderFilePath = "RenderPasses/VoxelReconstruction/Shader/ReduceBufferPass.cs.slang";


inline std::string kGBuffer = "gBuffer";
inline std::string kVBuffer = "vBuffer";
inline std::string kPBuffer = "pBuffer";
inline std::string kBlockMap = "blockMap";
inline std::string kOutputColor = "color";


inline std::string ReferenceImageDir = "D:/lck/vs/Reconstruction_Input/lego_white_256";
inline std::string ReferenceCameraFile = "D:/lck/vs/Reconstruction_Input/lego_white_256/camera_params.txt";
inline std::string ReconstructionDataDir = "D:/lck/vs/Reconstruction_Output";
} // namespace VoxelPrime

class VoxelReconstruction : public RenderPass
{
public:
    FALCOR_PLUGIN_CLASS(VoxelReconstruction, "VoxelReconstruction", "Insert pass description here.");

    static ref<VoxelReconstruction> create(ref<Device> pDevice, const Properties& props)
    {
        return make_ref<VoxelReconstruction>(pDevice, props);
    }

    VoxelReconstruction(ref<Device> pDevice, const Properties& props);

    virtual Properties getProperties() const override;
    virtual RenderPassReflection reflect(const CompileData& compileData) override;
    virtual void compile(RenderContext* pRenderContext, const CompileData& compileData) override {}
    virtual void execute(RenderContext* pRenderContext, const RenderData& renderData) override;
    virtual void renderUI(Gui::Widgets& widget) override;
    virtual void setScene(RenderContext* pRenderContext, const ref<Scene>& pScene) override;
    virtual bool onMouseEvent(const MouseEvent& mouseEvent) override;
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
    void VoxelReconstruction::refreshReconstructionFileList();

    struct GridResources
    {
        ref<Buffer> gridDataBuffer;
        ref<Texture> blockOM;
        GridData gridData;
    };

    struct RayMarchingPassResouce
    {
        ref<FullScreenPass> mpFullScreenPass;
        ref<FullScreenPass> mpDisplayNDFPass;
        ref<Sampler> mpPointSampler;
        std::unique_ptr<EnvMapSampler> mpEnvMapSampler;
        void init() {
            mpFullScreenPass = nullptr;
            mpDisplayNDFPass = nullptr;
            mpPointSampler = nullptr;
        }
    };
    struct RayMarchingPassParams
    {
        bool mOptionsChanged;
        uint mFrameIndex;
        uint2 mOutputResolution;
        bool mDisplayNDF;
        bool mCheckVisibility;
        bool mCheckCoverage;
        uint mDrawMode;
        bool mUseMipmap;
        uint mMaxBounce;
        bool mRenderBackGround;
        float3 mClearColor;
        //bool mCheckEllipsoid;
        bool mCheckPrimitive;
        float mTrasmittanceThreshold100;
        float mShadowBias100;
        float mMinPdf100;

        void init() {
            mOptionsChanged = false;
            mFrameIndex = 0;
            mOutputResolution = uint2(1920, 1080);
            mDisplayNDF = false;
            mCheckVisibility = false;
            mCheckCoverage = false;
            mDrawMode = 0;
            mUseMipmap = true;
            mMaxBounce = 0;
            mRenderBackGround = true;
            mClearColor = float3(0);
            //mCheckEllipsoid = false;
            mCheckPrimitive = true;
            mTrasmittanceThreshold100 = 5.f;
            mMinPdf100 = 0.1f;
            mShadowBias100 = 0.01f;
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

    struct GradientPass{
        ref<ComputePass> mpComputePass;
        ref<Buffer> gradBuffer;
        //ref<Buffer> mpVoxelGradAccumBuffer;

        void init()
        {
            gradBuffer = nullptr;
            mpComputePass = nullptr;
        }

    };

    struct UpdatePass
    {
        ref<ComputePass> mpComputePass;

            // 是否按当前 voxel 命中的 pixel 数做平均
        bool mUseGradCountNormalize;

        // 全局梯度缩放，第一版可以设为 1.0
        float mGradScale;

        float mLrDiffuse;
        float mLrSpecular;
        float mLrRough;
        float mLrWeight;
        float mLrNormal;
        float mLrCoverageFunc;
        float mLrVisibilityFunc;
        float mLrCenter;
        float mLrB;


        void init() {
            mpComputePass = nullptr;

            mUseGradCountNormalize = true;
            mGradScale = 1.0f;

            mLrDiffuse = 0.005f;
            mLrSpecular = 0.0f;
            mLrRough = 0.0f;
            mLrWeight = 0.0f;
            mLrNormal = 0.0f;
            mLrCoverageFunc = 0.0f;
            mLrVisibilityFunc = 0.0f;
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
        //ref<Buffer> mpTotalLoss;
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
    LossPass mLossPass;
    GradientPass mGradientPass;
    UpdatePass mUpdatePass;

    // Grid
    GridResources mGridResources;  // cpu中的对应gpu中的资源，变量赋值，buffer绑定
    ref<ParameterBlock> mpGridBlock; // gpu的block

    // RayMarchingPass
    RayMarchingPassResouce mRayMarchingPassResouce;
    RayMarchingPassParams mRayMarchingPassParams;
    uint3 MinFactor = uint3(1, 1, 1);

    // Voxel Optimization
    std::vector<ref<Texture>> mReferenceImages;
    std::vector<ref<Camera>> mReferenceCameras;
    ref<Buffer> mpPathRecordBuffer;

    // Reduce Pass
    ReduceLossPass mReduceLossPass;


    // UI
    bool mOptionsChanged = false;
    bool mEnableReconstruction = false;
    bool mInitVoxelData = false;
    //bool mVisualizeDataset = false;
    //bool test = false;
    uint testIndex = 0;
    bool mSaveReconstructionRequested = false;
    bool mLoadReconstructionRequested = false;
    bool mReconstructionFileListDirty = true;
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

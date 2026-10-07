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
#include "Core/Platform/OS.h"
#include "RenderGraph/RenderPassStandardFlags.h"
#include "iostream"
#include <fstream>
#include <filesystem>
#include <iomanip>
#include <sstream>
#include <random>
#include <nlohmann/json.hpp>


#include "Defines.h"
#include "Voxel/VoxelData.slang"
#include "Voxel/VoxelGrid.slang"
#include "Voxel/ABSDF.slang"

#include "PathRecord.slang"
#include "GradRecord.slang"
#include "GeometryAdamState.slang"
#include "TopologyEvidence.slang"

using namespace Falcor;

static_assert(sizeof(GaussianEllipsoid) == 40, "GaussianEllipsoid host/device layout changed.");
static_assert(sizeof(VoxelData) == 44 + 12 * SH_COUNT + 4 * SH_OPACITY_COUNT, "VoxelData host/device layout changed.");
static_assert(sizeof(GradRecord) == 48 + 12 * SH_COUNT + 4 * SH_OPACITY_COUNT, "GradRecord host/device layout changed.");
static_assert(sizeof(TopologyEvidence) == (2 * TOPOLOGY_EVIDENCE_VIEW_WORDS + 4) * sizeof(uint32_t),
    "TopologyEvidence host/device layout changed.");

namespace
{
const std::string ReflectTypesShaderFilePath = "RenderPasses/VoxelReconstructionNoLightTransport/Shader/ReflectTypes.cs.slang";
const std::string RayMarchingShaderFilePath = "RenderPasses/VoxelReconstructionNoLightTransport/Shader/RayMarchingPass.ps.slang";
const std::string LossPassShaderFilePath = "RenderPasses/VoxelReconstructionNoLightTransport/Shader/LossPass.cs.slang";
const std::string GradientPassShaderFilePath = "RenderPasses/VoxelReconstructionNoLightTransport/Shader/GradientPass.cs.slang";
const std::string UpdatePassShaderFilePath = "RenderPasses/VoxelReconstructionNoLightTransport/Shader/UpdatePass.cs.slang";
const std::string TopologyPassShaderFilePath = "RenderPasses/VoxelReconstructionNoLightTransport/Shader/TopologyPass.cs.slang";
const std::string DeleteCompactPassShaderFilePath = "RenderPasses/VoxelReconstructionNoLightTransport/Shader/DeleteCompactPass.cs.slang";
const std::string GrowthPassShaderFilePath = "RenderPasses/VoxelReconstructionNoLightTransport/Shader/GrowthPass.cs.slang";
const std::string ReduceTexturePassShaderFilePath = "RenderPasses/VoxelReconstructionNoLightTransport/Shader/ReduceTexturePass.cs.slang";
const std::string ReduceBufferPassShaderFilePath = "RenderPasses/VoxelReconstructionNoLightTransport/Shader/ReduceBufferPass.cs.slang";

inline std::string kGBuffer = "gBuffer";
inline std::string kPBuffer = "pBuffer";
inline std::string kOutputColor = "color";
inline std::string kAccumulateOutputColor = "AccuColor";

// Relative paths are rooted at the Falcor source project, independently of the process working directory.
inline std::string ReconstructionDataDir = "Reconstruction_Output";
inline std::string ReferenceImageDir = "Reconstruction_Input/lego";
inline std::string ReferenceCameraFile = "Reconstruction_Input/lego/transforms_train.json";
// Use the sparse point cloud that was supplied to 3DGS. The optimized 3DGS
// output remains available as "point_cloud.ply" for comparison experiments.
//inline std::string InitializationPointCloudFile = "init_points.ply";
inline std::string InitializationPointCloudFile = "point_cloud.ply";

inline std::filesystem::path resolveReconstructionPath(const std::filesystem::path& path)
{
    return (path.is_absolute() ? path : Falcor::getProjectDirectory() / path).lexically_normal();
}
} // namespace VoxelPrime

class VoxelReconstructionNoLightTransport : public RenderPass
{
    friend struct NeighborGrowthTestAccess;
    friend struct CoarseToFineTestAccess;
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
    virtual bool onMouseEvent(const MouseEvent& mouseEvent) override;
    virtual bool onKeyEvent(const KeyboardEvent& keyEvent) override { return false; }

    void beginFrame(RenderContext* pRenderContext, bool forceReset = false);
    void endFrame(RenderContext* pRenderContext);


    void setupGridResouce(RenderContext* pRenderContext, bool forceReset);
    void UpdateVoxelGrid(uint voxelResolution);

    void createRayMarchingPassResource(RenderContext* pRenderContext);
    void rayMarchingPass(RenderContext* pRenderContext, const RenderData& renderData);

    void createLossPassResource(RenderContext* pRenderContext);
    void runLossPass(RenderContext* pRenderContext, const RenderData& renderData);

    void createGradientPassResource(RenderContext* pRenderContext);
    void runGradientPass(RenderContext* pRenderContext, const RenderData& renderData);
    float getGeometryTauWorld() const;

    void createUpdatePassResource(RenderContext* pRenderContext);
    void runUpdatePass(RenderContext* pRenderContext, const RenderData& renderData);
    void renderUIUpdatePass(Gui::Widgets& widget);
    void renderUITopology(Gui::Widgets& widget);
    float getEffectiveOpacityLearningRate() const;
    void createTopologyPassResource(RenderContext* pRenderContext);
    void evaluateDeletionEvidence(RenderContext* pRenderContext);
    void resetDeletionEvidence(RenderContext* pRenderContext, bool preserveGrowthProtection = false);
    uint32_t getDeletionEvidenceStartIteration() const;
    bool shouldCollectDeletionEvidence() const;
    void createDeletionPassResources();
    void deleteAndCompactCandidates(RenderContext* pRenderContext);

    void createReducePassResource(RenderContext* pRenderContext);
    void runReducePass(RenderContext* pRenderContext, const RenderData& renderData);

    void loadReferenceImages();
    bool loadReferenceCamerasFromFile(const std::string& cameraFile);

    void startReconstruction();
    void stopReconstruction();
    void shuffleTrainingViews();
    uint32_t getTrainingViewIndex() const;

    std::filesystem::path getDefaultReconstructionSavePath();
    std::string getReconstructionExperimentPrefix();
    std::string getOptimizedParamTag() const;
    void saveReconstruction(RenderContext* pRenderContext);
    void loadReconstruction(RenderContext* pRenderContext, const std::filesystem::path& path);
    void refreshReconstructionFileList();
    void refreshExperimentLevels();
    void saveLossHistory(const std::filesystem::path& reconstructionPath) const;
    std::filesystem::path getReconstructionModeDirectory() const;

    struct GridResources
    {
        GridData gridData;
        std::vector<ref<Texture>> indexPages;
        std::vector<ref<Buffer>> voxelPages;
        std::vector<ref<Buffer>> gradPages;
        std::vector<ref<Buffer>> topologyEvidencePages;
        std::vector<ref<Buffer>> adamPages;
        std::vector<ref<Buffer>> radianceAdamIndexPages;
        std::vector<ref<Buffer>> radianceAdamPages;
        ref<Buffer> radianceAdamCounter;
        std::vector<ref<Buffer>> cellIndexPages;
    };

    struct RayMarchingPass
    {
        bool mOptionsChanged;
        uint mFrameIndex;
        uint2 mOutputResolution;
        float3 mClearColor;
        bool mCheckPrimitive;
        bool mUsePixelFrustum = false;
        float mShadowBias100;
        uint mDrawMode;
        bool mRenderBackGround;
        uint mMaxContributingVoxelCount;
        float mTransmittanceThreshold;

        // loss 的前缀平均需要 N >= 2 才有意义（N == 1 时前缀为空，会退回旧行为）.
        uint mSpp = 8;
        uint mSampleIndex = 0;

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
            mOutputResolution = uint2(800, 800);
            mClearColor = float3(0);
            mCheckPrimitive = true;
            mShadowBias100 = 0.01f;
            mDrawMode = 0;
            mRenderBackGround = true;
            mMaxContributingVoxelCount = MAX_CONTRIBUTING_VOXELS_PER_RAY;
            mTransmittanceThreshold = 0.01f;
        }
    };

    struct LossPass
    {
        uint mView;
        float alphaLossWeight = 0.5f;
        ref<ComputePass> mpComputePass;
        ref<Texture> lossBuffer;
        ref<Texture> dL_dColor;
        ref<Texture> mpBackGroundMask;
        void init()
        {
            mView = 0;
            mpComputePass = nullptr;
            dL_dColor = nullptr;
            lossBuffer = nullptr;
            mpBackGroundMask = nullptr;
        }
    };

    struct GradientPass
    {
        ref<ComputePass> mpComputePass;
        float geometryGradClamp;
        float geometryTauVoxelFraction = 0.12f;
        float alphaGeometryWeight = 0.1f; // Extra multiplier on the existing alpha-loss geometry proxy only.

        void init()
        {
            mpComputePass = nullptr;
            // Keep the configured tau and alpha multiplier when recreating GPU resources.
            geometryTauVoxelFraction = 0.12f;
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
        float mLrShape;
        float mLrRotation;
        float mLrOpacity;
        float mBackgroundCarveAdamMultiplier;
        uint32_t mOpacityWarmupIterations;
        uint32_t mOpacityRampIterations;

        void init()
        {
            mpComputePass = nullptr;

            mUseGradCountNormalize = true;
            mGradScale = 1.0f;

            // Separate Adam states for radiance, opacity, center,
            // log semi-axes, and tangent-space rotation.
            mLrRadiance = 1e-3f;
            mLrOpacity = 5e-4f;
            mLrCenter = 1e-3f;
            mLrShape = 1e-3f;
            mLrRotation = 5e-4f;
            mBackgroundCarveAdamMultiplier = 5.0f;
            mOpacityWarmupIterations = 20u;
            mOpacityRampIterations = 30u;

        }
    };

    struct OptimizerParams
    {
        bool isRunning = false;

        // 控制一次优化过程
        uint32_t maxIteration = 200;
        uint32_t currentIteration = 0;

        // 每次 iteration 使用多少个 camera/view
        uint32_t viewsPerIteration = 100;
        uint32_t currentView = 0;

        void reset()
        {
            currentIteration = 0;
            currentView = 0;
            isRunning = false;
        }
    };

    // Confirmed evidence triggers periodic deletion and in-place pool
    // compaction at complete iteration boundaries.
    struct TopologySettings
    {
        uint32_t debugLayer = uint32_t(TopologyDebugLayer::Occupied);
        bool showOccupiedContext = true;
        bool collectDeletionEvidence = true;
        bool enableGrowth = true;
        // Minimum outward depth into the face neighbor, in that axis's voxel widths.
        float growthFacePenetration = 1.0f;
        uint32_t growthInterval = 10;
        float growthShrink = 0.7f;
        float growthContactOffset = 0.2f;
        float growthInitialOpacity = 0.1f;
        uint32_t growthWaitIterations = 5;
        uint32_t growthProtectionIterations = 5;
        uint32_t deletionCooldownIterations = 5;
        uint32_t lastGrowthCount = 0;
        uint32_t lastGrowthPages = 0;
        std::string growthStatus = "Waiting for opacity warm-up + ramp";
        float foregroundAlphaMin = 0.95f;
        float backgroundAlphaMax = 1e-4f;
        float minRemovalLossDelta = 1e-4f;
        float minEvidenceTransmittance = 0.05f;
        uint32_t minDeletionConflictViews = 5;
        uint32_t deletionConflictSupportRatio = 2;
        uint32_t evidenceInterval = 5;
        uint32_t deletionInterval = 10;
        uint32_t candidateCount = 0;
        uint32_t oneWindowCount = 0;
        uint32_t protectedCount = 0;
        uint32_t weakConflictCount = 0;
        uint32_t completedWindows = 0;
        uint32_t lastDeletedCount = 0;
        uint32_t lastReleasedPoolPages = 0;
        uint32_t lastReleasedRadiancePages = 0;
        std::string deletionStatus = "No compaction performed";
    };

    struct CoarseToFineSettings
    {
        uint32_t startResolution = COARSE_TO_FINE_START_RESOLUTION;
        uint32_t targetResolution = GRID_RESOLUTION;
        uint32_t totalIterations = COARSE_TO_FINE_TOTAL_ITERATIONS;
        float parentOpacityThreshold = 0.0f;
        float opacityOpticalDepthScale = 0.65f;
        float childFaceOverlap = 0.1f;
        bool enableCoarseGrowth = true;
        float coarsestGrowthFacePenetration = COARSEST_GROWTH_FACE_PENETRATION;
        float growthFinerLevelMultiplier = FINER_GROWTH_THRESHOLD_MULTIPLIER;
        uint32_t coarseGrowthInterval = 20u;
        bool saveEachLevel = true;
        uint32_t scheduleStartResolution = 0;
        uint32_t levelStartIteration = 0;
        uint32_t levelIterationBudget = 0;
        bool initializationPending = false;
        std::string status = "Initialize from PLY to start the resolution pyramid";
    };

    struct ReduceLossPass
    {
        ref<Buffer> mpReduceBufferA;
        ref<Buffer> mpReduceBufferB;
        ref<Buffer> mpTotalLossReadback;
        ref<ComputePass> mpReduceTexturePass;
        ref<ComputePass> mpReduceBufferPass;

        float meanLoss = 0.0f;

        // 当前 iteration 内所有 view 的 loss 累加
        float iterationLossSum = 0.0f;
        uint32_t iterationLossCount = 0;

        // 每个 iteration 一个点
        std::vector<float> iterationLossHistory;
    };

private:
    bool isCoarseToFine() const { return mReconstructionMode == 1u; }
    void renderUICoarseToFine(Gui::Widgets& widget);
    void validateCoarseToFineSettings() const;
    bool prepareReconstruction(RenderContext* pRenderContext);
    void advanceCoarseToFine(RenderContext* pRenderContext);
    void refineCoarseToFineGrid(RenderContext* pRenderContext);
    uint32_t coarseToFineRemainingLevels() const;
    uint32_t coarseToFineLevelBudget(uint32_t resolution) const;
    GridData makeVoxelGrid(uint32_t resolution) const;
    static DefineList getReconstructionDefines();
    void updateOutputResolution();
    void resetLoadedReconstruction(RenderContext* pRenderContext);


    struct PointCloudState
    {
        bool initialized = false;
        bool startRequested = false;
        bool clearAccumulation = true;
        std::string status = "Not initialized";

        // 高斯占位的透明度阈值: 峰值 alpha = sigmoid(opacity) 低于它就连中心格都不占,
        // 高于它的高斯按 sqrt(2 ln(alpha / threshold)) * sigma 的半径扩张.
        // 只在点 Init / Reset from PLY 时读取一次.
        float opacityThreshold = 0.05f;
    };
    PointCloudState mPointCloud;
    ref<ComputePass> mpInitializePointCloudPass;
    ref<ComputePass> mpBuildSparseIndexPass;
    ref<ComputePass> mpTopologyPass;
    ref<ComputePass> mpResetTopologyEvidencePass;
    ref<Buffer> mpTopologySummary;
    ref<ComputePass> mpBuildDeletionListsPass;
    ref<ComputePass> mpClearDeletionIndicesPass;
    ref<ComputePass> mpMoveDeletionSurvivorsPass;
    ref<ComputePass> mpClearDeletionTailPass;
    ref<ComputePass> mpProposeGrowthPass;
    ref<ComputePass> mpInitializeGrowthPass;
    ref<ComputePass> mpCommitGrowthPass;
    ref<ComputePass> mpClearGrowthClaimsPass;
    ref<ComputePass> mpRollbackGrowthPass;
    ref<ComputePass> mpClearGrowthCooldownPass;
    bool mGrowthCooldownPresent = false;
    void createGrowthPassResources();
    void resetGrowthCooldown(RenderContext* pRenderContext);
    void growNeighborVoxels(RenderContext* pRenderContext);
    float getEffectiveGrowthFacePenetration() const;
    bool initializePointCloudVoxelData(RenderContext* pRenderContext, const std::filesystem::path& source = {});
    void resetPointCloudOptimization(RenderContext* pRenderContext);
    GridResources allocateSparseGrid(RenderContext* pRenderContext, const GridData& grid, uint32_t capacity);
    ref<ParameterBlock> createSparseGridBlock(const GridResources& resources);
    void commitSparseGrid(GridResources&& resources, const ref<ParameterBlock>& block, uint32_t resolution);
    void reserveSparseVoxelCapacity(RenderContext* pRenderContext, uint32_t minimumCapacity);
    void reserveRadianceAdamCapacity(RenderContext* pRenderContext, uint32_t minimumCapacity);
    void uploadSparseBatch(RenderContext* pRenderContext, const ref<ParameterBlock>& block,
        uint32_t offset, const uint32_t* cells, uint32_t count, const VoxelData* data = nullptr);
    void clearSparseGradients(RenderContext* pRenderContext);
    void barrierSparseVoxels(RenderContext* pRenderContext);
    void barrierSparseGradients(RenderContext* pRenderContext);
    void barrierTopologyEvidence(RenderContext* pRenderContext);
    void saveSparseReconstruction(RenderContext* pRenderContext, const std::filesystem::path& path);
    void loadSparseReconstruction(RenderContext* pRenderContext, const std::filesystem::path& path);

    ref<Device> mpDevice;
    ref<Scene> mpScene;
    std::unique_ptr<PixelDebug> mpPixelDebug;

    // Parameters
    uint mFrameCount = 0;
    uint2 mFrameDim;
    float2 mInvFrameDim;
    uint mVoxelResolution = GRID_RESOLUTION; // X,Y,Z三个方向中，最长的边被划分的体素数量

    OptimizerParams mOptimizerParams;
    uint32_t mReconstructionMode = RECONSTRUCTION_MODE;
    CoarseToFineSettings mCoarseToFine;
    TopologySettings mTopologySettings;

    // Passes
    ref<ComputePass> mpReflectTypes;
    GradientPass mGradientPass;
    UpdatePass mUpdatePass;
    LossPass mLossPass;
    ReduceLossPass mReduceLossPass;


    // Grid
    GridResources mGridResources;    // cpu中的对应gpu中的资源，变量赋值，buffer绑定
    ref<ParameterBlock> mpGridBlock; // gpu的block

    // RayMarchingPass
    RayMarchingPass mRayMarchingPass;
    uint32_t mViewingResolution = 0; // 0: 800x800, 1: 1920x1080; training always uses 800x800.
    uint3 MinFactor = uint3(1, 1, 1);

    // Voxel Optimization
    std::vector<ref<Texture>> mReferenceImages;
    std::vector<ref<Camera>> mReferenceCameras;
    // currentView is the traversal position; camera/image/evidence IDs use this permutation.
    std::vector<uint32_t> mTrainingViewOrder;
    std::mt19937 mTrainingViewRng{std::random_device{}()};
    std::vector<std::filesystem::path> mReferenceImagePaths;
    ref<Buffer> mpPathRecordBuffer;

    // UI
    bool mOptionsChanged = false;
    bool mEnableReconstruction = false;
    bool mInitVoxelData = false;
    bool mUseReferenceCamera = false;
    uint testIndex = 0;
    bool mSaveReconstructionRequested = false;
    bool mLoadReconstructionRequested = false;
    bool mReconstructionFileListDirty = true;
    bool mLoadedReconstructionForViewing = false;
    std::string mReconstructionIOStatus;
    std::string mReferenceDataError;
    std::vector<std::filesystem::path> mReconstructionFilePaths;
    uint32_t mSelectedReconstructionFile = 0;
    std::string mReconstructionNameTag = "";
    std::filesystem::path mReconstructionOutputRoot = ReconstructionDataDir;
    std::string mReconstructionExperimentKey;
    std::string mReconstructionExperimentPrefix;
    // Viewing group is anchored to the last successfully saved/loaded result, not a file-list selection.
    std::filesystem::path mViewedReconstructionPath;
    std::string mViewedExperimentLabel;
    std::vector<uint32_t> mExperimentLevelFileIndices;
    std::vector<uint32_t> mExperimentLevelResolutions;
    uint32_t mSelectedExperimentLevel = 0u;

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

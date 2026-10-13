#include "../VoxelReconstructionNoLightTransport.h"
#include "../../ReconstructionResultPass/ResultMetadata.h"
#include "RenderGraph/ResourceCache.h"
#include <chrono>
#include <iostream>

struct DatasetSelectionTestAccess
{
    static void require(bool value, const char* message) { if (!value) throw std::runtime_error(message); }
    static void run(VoxelReconstructionNoLightTransport& pass, RenderContext* ctx)
    {
        namespace fs = std::filesystem;
        const auto root = getProjectDirectory() / "build" / "dataset_selection_tests" /
            std::to_string(std::chrono::system_clock::now().time_since_epoch().count());
        const auto first = root / "first";
        const auto second = root / "second";
        const auto invalid = root / "invalid";
        fs::create_directories(invalid);
        const auto& grid = pass.mGridResources.gridData;
        const float3 position = grid.gridMin + 1.5f * grid.voxelSize;
        for (const auto& path : {first, second})
        {
            fs::create_directories(path);
            std::ofstream ply(path / "init_points.ply");
            ply << "ply\nformat ascii 1.0\nelement vertex 1\nproperty float x\nproperty float y\nproperty float z\nend_header\n";
            ply << position.x << ' ' << -position.z << ' ' << position.y << '\n';
            std::ofstream json(path / "transforms_train.json");
            json << R"({"camera_angle_x":0.69,"frames":[
                {"file_path":"./r_0","transform_matrix":[[1,0,0,0],[0,1,0,0],[0,0,1,4],[0,0,0,1]]},
                {"file_path":"./r_1","transform_matrix":[[1,0,0,1],[0,1,0,0],[0,0,1,4],[0,0,0,1]]}]})";
            uint8_t pixel[] = {128, 64, 32, 255};
            auto image = pass.mpDevice->createTexture2D(1, 1, ResourceFormat::RGBA8UnormSrgb, 1, 1, pixel);
            for (const auto* file : {"r_0.png", "r_1.png"})
                image->captureToFile(0, 0, path / file, Bitmap::FileFormat::PngFile, Bitmap::ExportFlags::ExportAlpha, false);
        }
        pass.mReconstructionMode = 0;
        pass.mCoarseToFine.initializationPending = false;
        pass.mRayMarchingPass.mOutputResolution = uint2(1);
        pass.mRayMarchingPass.mDrawMode = uint32_t(ReconstructionDrawMode::Default);
        pass.mDatasetDirectories = {first, second, invalid};
        pass.mSelectedPointCloudFile = "init_points.ply";
        auto hostScene = pass.mpScene;
        const auto flags = ResourceBindFlags::ShaderResource | ResourceBindFlags::RenderTarget | ResourceBindFlags::UnorderedAccess;
        ResourceCache cache;
        Dictionary dict;
        const std::string name = "dataset-test";
        for (const auto* channel : {"color", "AccuColor", "dummy"})
            cache.registerExternalResource(name + "." + channel,
                pass.mpDevice->createTexture2D(1, 1, ResourceFormat::RGBA32Float, 1, 1, nullptr, flags));
        struct TestRenderData : RenderData
        {
            TestRenderData(const std::string& name, ResourceCache& cache, Dictionary& dictionary)
                : RenderData(name, cache, dictionary, uint2(1), ResourceFormat::RGBA32Float) {}
        } data(name, cache, dict);
        const std::string readyKey = pass.getName() + ReconstructionResult::kReady;
        for (uint32_t selection : {0u, 1u})
        {
            pass.mSelectedDataset = selection;
            pass.mDatasetLoadRequested = true;
            pass.execute(ctx, data);
            require(!dict.getValue(readyKey, false), "Partially loaded dataset advertised a valid result");
            require(pass.mReferenceImages.size() == 1 && pass.mReferenceCameras.size() == 2,
                "Dataset selection did not load new cameras and first image");
            pass.execute(ctx, data);
            require(dict.getValue(readyKey, false), "Complete dataset initialization did not become ready");
            require(pass.mpScene == hostScene, "Dataset loading changed Falcor's host scene");
            require(pass.mGridResources.gridData.activeVoxelCount == 1 && pass.mPointCloud.initialized,
                "Dataset initialization did not replace the previous point cloud");
            require(pass.mReferenceImagePaths[0].parent_path() == pass.mDatasetDirectories[selection], "Retained old image paths");
            require(pass.mOptimizerParams.currentIteration == 0 && pass.mTrainingViewOrder.empty() && pass.mContourViews.empty(),
                "Dataset switch retained optimizer or sampling history");
            require(pass.mUseReferenceCamera && pass.testIndex == 0 && !pass.mEnableReconstruction,
                "Dataset load should enter reference-camera preview, not start training");
            require(dict.getValue(pass.getName() + ReconstructionResult::kScene, std::string{}) ==
                pass.mDatasetDirectories[selection].filename().string(), "Published the wrong scene name");
        }
        auto oldImage = pass.mReferenceImages[0];
        pass.mSelectedDataset = 2;
        pass.mDatasetLoadRequested = true;
        pass.execute(ctx, data);
        require(pass.mReferenceImageDir == second && pass.mReferenceImages[0] == oldImage && pass.mDatasetReady,
            "An invalid directory selection destroyed the active dataset");
        require(pass.mDatasetStatus.find("failed") != std::string::npos, "Invalid input was not reported");
        std::cout << "PASS: input switching, staged reference loading, PLY initialization, metadata and unchanged host scene\n";
    }
};

void runDatasetSelectionTests(VoxelReconstructionNoLightTransport& pass, RenderContext* ctx)
{
    DatasetSelectionTestAccess::run(pass, ctx);
}

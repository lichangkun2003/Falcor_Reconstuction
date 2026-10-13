#include "VoxelReconstructionNoLightTransport.h"
#include <algorithm>

void VoxelReconstructionNoLightTransport::refreshDatasetList()
{
    const auto previous = mSelectedDataset < mDatasetDirectories.size() ? mDatasetDirectories[mSelectedDataset] :
        resolveReconstructionPath("Reconstruction_Input/ship");
    mDatasetDirectories.clear();
    const auto root = resolveReconstructionPath("Reconstruction_Input");
    try
    {
        if (std::filesystem::is_directory(root))
            for (const auto& entry : std::filesystem::directory_iterator(root))
                if (entry.is_directory()) mDatasetDirectories.push_back(entry.path());
        std::sort(mDatasetDirectories.begin(), mDatasetDirectories.end());
        const auto it = std::find(mDatasetDirectories.begin(), mDatasetDirectories.end(), previous);
        mSelectedDataset = it == mDatasetDirectories.end() ? 0u : uint32_t(it - mDatasetDirectories.begin());
    }
    catch (const std::exception& e)
    {
        mDatasetStatus = "Cannot list input directories: " + std::string(e.what());
    }
    mDatasetListDirty = false;
}

void VoxelReconstructionNoLightTransport::renderUIDataset(Gui::Widgets& widget)
{
    if (auto group = widget.group("Input Dataset", true))
    {
        if (mDatasetListDirty) refreshDatasetList();
        if (group.button("Refresh Input Directories")) refreshDatasetList();
        Gui::DropdownList directories;
        for (uint32_t i = 0; i < mDatasetDirectories.size(); ++i)
            directories.push_back({i, mDatasetDirectories[i].filename().string()});
        if (!directories.empty())
        {
            group.dropdown("Scene Directory", directories, mSelectedDataset);
            group.textbox("Initialization PLY", mSelectedPointCloudFile);
            group.text("Cameras: transforms_train.json");
            group.tooltip("Reference images use file_path entries from this JSON, relative to the selected directory.");
            if (group.button("Load / Initialize Selected Dataset"))
            {
                // Compile the graph at reference resolution before next frame's loading begins.
                mViewingResolution = 0;
                updateOutputResolution();
                mDatasetLoadRequested = true;
            }
        }
        else group.text("No directories found in Reconstruction_Input.");
        group.text("Active dataset: " + (mReferenceImageDir.empty() ? std::string("none") : mReferenceImageDir.filename().string()));
        group.text(mDatasetStatus);
        if (!mpScene) group.text("Load the Falcor host scene first; this button only changes reconstruction input data.");
    }
}

void VoxelReconstructionNoLightTransport::loadSelectedDataset(RenderContext* pRenderContext)
{
    mDatasetLoadRequested = false;
    try
    {
        FALCOR_CHECK(mSelectedDataset < mDatasetDirectories.size(), "No input directory selected.");
        const auto directory = mDatasetDirectories[mSelectedDataset];
        const auto cameraFile = directory / "transforms_train.json";
        FALCOR_CHECK(std::filesystem::is_regular_file(cameraFile), "Missing camera file: {}", cameraFile.string());
        FALCOR_CHECK(!mSelectedPointCloudFile.empty() && std::filesystem::is_regular_file(directory / mSelectedPointCloudFile),
            "Missing initialization PLY: {}", (directory / mSelectedPointCloudFile).string());

        // Switch only reconstruction-owned state. Falcor's setScene() is deliberately not involved.
        resetPointCloudOptimization(pRenderContext);
        mDatasetReady = false;
        mReferenceImageDir = directory;
        mReferenceCameraFile = cameraFile;
        mInitializationPointCloudFile = mSelectedPointCloudFile;
        mReferenceImages.clear();
        mReferenceCameras.clear();
        mReferenceImagePaths.clear();
        mTrainingViewOrder.clear();
        mReferenceDataError.clear();
        mViewedReconstructionPath.clear();
        mViewedExperimentLabel.clear();
        mExperimentLevelFileIndices.clear();
        mExperimentLevelResolutions.clear();
        mReconstructionExperimentPrefix.clear();
        mLoadReconstructionRequested = false;
        mSaveReconstructionRequested = false;
        mLoadedReconstructionForViewing = false;
        mPointCloud.initialized = false;
        mInitVoxelData = true; // Execute initializes once all images and camera poses have loaded.
        mUseReferenceCamera = true;
        testIndex = 0;
        mDatasetStatus = "Loading references for " + directory.filename().string() + "...";
    }
    catch (const std::exception& e)
    {
        mDatasetStatus = "Dataset load failed: " + std::string(e.what());
        logWarning(mDatasetStatus);
    }
}

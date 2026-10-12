#include "../VoxelReconstructionNoLightTransport.h"
#include "Scene/SceneBuilder.h"
#include "RenderGraph/ResourceCache.h"
#include <cmath>
#include <cstring>
#include <iostream>

struct ContourSamplingTestAccess
{
    static void require(bool value, const char* message) { if (!value) throw std::runtime_error(message); }
    static void near(float a, float b, const char* message) { require(std::abs(a - b) < 2e-5f, message); }
    static float4 pixel(RenderContext* ctx, const ref<Texture>& texture)
    {
        auto bytes = ctx->readTextureSubresource(texture.get(), 0);
        float4 result;
        std::memcpy(&result, bytes.data(), sizeof(result));
        return result;
    }
    static ContourHistory history(VoxelReconstructionNoLightTransport& pass, uint32_t view, uint32_t slot)
    {
        ContourHistory result;
        pass.mContourViews[view].history->getBlob(&result, slot * sizeof(result), sizeof(result));
        return result;
    }

    static void lifecycle(VoxelReconstructionNoLightTransport& pass, RenderContext* ctx)
    {
        pass.resetContourSampling();
        pass.mEnableReconstruction = pass.mOptimizerParams.isRunning = true;
        pass.mContourSampling = {};
        pass.mRayMarchingPass.mOutputResolution = uint2(4, 1);
        float4 colors[] = {float4(0, 0, 0, 0), float4(0, 0, 0, 0.25f), float4(0, 0, 0, 1), float4(0, 0, 0, 0.75f)};
        auto reference = pass.mpDevice->createTexture2D(4, 1, ResourceFormat::RGBA32Float, 1, 1, colors);
        pass.mReferenceImages = {reference, reference};
        const auto vp = float4x4::identity();
        pass.beginContourSampling(ctx, 0, vp);
        ctx->submit(true);
        require(pass.mContourViews[0].count == 2u && pass.mContourBatchActive, "Contour mask is not strictly 0 < alpha < 1");
        auto bytes = ctx->readTextureSubresource(pass.mpContourLookup.get(), 0);
        uint32_t lookup[4];
        std::memcpy(lookup, bytes.data(), sizeof(lookup));
        require(lookup[0] == 0 && lookup[2] == 0 && lookup[1] > 0 && lookup[3] > 0, "Sparse contour lookup includes pure pixels");
        const uint32_t slot = lookup[1] - 1u;
        const auto before = history(pass, 0, slot);
        near(before.meanAlpha[0], 0.25f, "Incorrect uniform history prior");
        std::vector<ContourObservation> observations(32);
        observations[slot * 16] = {1.0f, 2u};
        observations[slot * 16 + 1] = {0.0f, 1u};
        pass.mpContourObservations->setBlob(observations.data(), 0, observations.size() * sizeof(ContourObservation));
        pass.mContourSampling.historyRate = 1.0f; // The current batch must retain its snapshot (0.3).
        ctx->submit(true);
        auto unchanged = history(pass, 0, slot);
        require(std::memcmp(&before, &unchanged, sizeof(before)) == 0, "SPP observations modified the active proposal");
        pass.finishContourSampling(ctx);
        ctx->submit(true);
        auto after = history(pass, 0, slot);
        near(after.meanAlpha[0], 0.325f, "EMA did not use raw alpha mean or batch rate");
        near(after.meanAlpha[1], 0.175f, "Background evidence did not reduce importance");
        near(after.meanAlpha[2], 0.25f, "Unvisited bin lost history");
        pass.beginContourSampling(ctx, 1, vp);
        ctx->submit(true);
        auto other = history(pass, 1, 0);
        near(other.meanAlpha[0], colors[other.pixelIndex].w, "Cameras share sampling history");
        pass.finishContourSampling(ctx);
        pass.beginContourSampling(ctx, 0, vp);
        ctx->submit(true);
        auto revisit = history(pass, 0, slot);
        require(std::memcmp(&after, &revisit, sizeof(after)) == 0, "Revisiting physical camera lost updated history");
        pass.mpContourObservations->getBlob(observations.data(), 0, observations.size() * sizeof(ContourObservation));
        for (const auto& observation : observations) require(observation.count == 0u, "Scratch was not cleared between batches");
        auto moved = vp;
        moved[0][3] = 1.0f;
        pass.beginContourSampling(ctx, 0, moved);
        ctx->submit(true);
        auto reset = history(pass, 0, 0);
        near(reset.meanAlpha[0], colors[reset.pixelIndex].w, "Camera changes retained stale history");
        pass.mContourSampling.enabled = false;
        pass.beginContourSampling(ctx, 0, moved);
        require(!pass.mContourBatchActive, "Disabled sampling retained an active proposal");
        pass.resetContourSampling();
        require(pass.mContourViews.empty() && !pass.mpContourLookup, "History reset retained resources");
        std::cout << "PASS: contour mask, per-camera history, frozen batch, EMA, revisit and reset\n";
    }

    static void distribution(VoxelReconstructionNoLightTransport& pass, RenderContext* ctx)
    {
        ProgramDesc desc;
        desc.addShaderLibrary("RenderPasses/VoxelReconstructionNoLightTransport/Tests/ContourSampling.cs.slang").csEntry("main");
        auto test = ComputePass::create(pass.mpDevice, desc, DefineList(), true);
        constexpr uint32_t count = 262144;
        const auto flags = ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess;
        auto samples = pass.mpDevice->createStructuredBuffer(sizeof(float4), count, flags);
        auto estimates = pass.mpDevice->createStructuredBuffer(sizeof(float4), count, flags);
        auto table = pass.mpDevice->createStructuredBuffer(sizeof(ContourHistory), 1, flags);
        auto var = test->getRootVar();
        var["gSamples"] = samples;
        var["gEstimates"] = estimates;
        var["gHistory"] = table;
        var["CB"]["gCount"] = count;
        std::vector<float4> points(count), values(count);
        uint32_t hits[3] = {};
        for (uint32_t trial = 0; trial < 3; ++trial)
        {
            ContourHistory h = {};
            if (trial != 2) h.meanAlpha[0] = 1;
            const float epsilon = trial == 1 ? 1.0f : 0.5f;
            table->setBlob(&h, 0, sizeof(h));
            var["CB"]["gUniformFraction"] = epsilon;
            test->execute(ctx, uint3(count, 1, 1));
            ctx->submit(true);
            samples->getBlob(points.data(), 0, count * sizeof(float4));
            estimates->getBlob(values.data(), 0, count * sizeof(float4));
            uint32_t bins[16] = {};
            double sum[4] = {}, squares[4] = {};
            for (uint32_t i = 0; i < count; ++i)
            {
                const auto p = points[i];
                require(p.x >= 0 && p.x < 1 && p.y >= 0 && p.y < 1, "Sample left unit pixel");
                const auto bin = uint32_t(p.w);
                require(bin == uint32_t(p.x * 4) + 4 * uint32_t(p.y * 4), "CDF selected wrong bin");
                ++bins[bin];
                hits[trial] += p.x < 0.025f && p.y < 0.25f;
                const float probability = trial == 0 ? (bin == 0 ? 0.53125f : 0.03125f) : 0.0625f;
                near(p.z, 1.0f / (16.0f * probability), "PDF omitted mixture or bin area");
                for (uint32_t c = 0; c < 4; ++c) { sum[c] += values[i][c]; squares[c] += double(values[i][c]) * values[i][c]; }
            }
            for (uint32_t b = 0; b < 16; ++b)
            {
                const double probability = trial == 0 ? (b == 0 ? 0.53125 : 0.03125) : 0.0625;
                require(std::abs(double(bins[b]) / count - probability) < 6 * std::sqrt(probability * (1 - probability) / count),
                    "Empirical bin frequency disagrees with proposal");
            }
            const double area = 0.025 * 0.25;
            const double target[] = {1 - 0.8 * area, 0.4 * area, 0.5 + (0.6 - 0.0125) * area, area};
            for (uint32_t c = 0; c < 4; ++c)
            {
                const double mean = sum[c] / count;
                const double variance = std::max(0.0, squares[c] / count - mean * mean);
                require(std::abs(mean - target[c]) < 6 * std::sqrt(variance / count) + 1e-5,
                    "PDF-corrected color/alpha differs from known area integral");
            }
        }
        require(hits[0] > 6 * hits[1], "Learned bin did not improve thin-structure hit rate");
        std::cout << "PASS: contour distribution and weighted RGB/alpha integrals; thin hits " << hits[0] << " vs " << hits[1] << " uniform\n";
    }

    static void prefix(VoxelReconstructionNoLightTransport& pass, RenderContext* ctx)
    {
        ProgramDesc desc;
        desc.addShaderLibrary(LossPassShaderFilePath).csEntry("main");
        auto loss = ComputePass::create(pass.mpDevice, desc, DefineList(), true);
        auto var = loss->getRootVar();
        const float4 previous(-0.2f, 0.3f, 1.4f, 1.6f), current(0.4f, 0.5f, 0.6f, 0.2f);
        const float4 accumulated = (previous + current) / 8.0f;
        const float4 reference(0.2f, 0.4f, 0.6f, 0.5f);
        auto texture = [&](const float4& value) { return pass.mpDevice->createTexture2D(1, 1, ResourceFormat::RGBA32Float, 1, 1, &value); };
        var["gRenderedColor"] = texture(accumulated);
        var["gCurrentFrameColor"] = texture(current);
        var["gReferenceImage"] = texture(reference);
        const auto flags = ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess;
        auto derivatives = pass.mpDevice->createTexture2D(1, 1, ResourceFormat::RGBA32Float, 1, 1, nullptr, flags);
        var["gDL_dColorBuffer"] = derivatives;
        var["gLossBuffer"] = pass.mpDevice->createTexture2D(1, 1, ResourceFormat::R32Float, 1, 1, nullptr, flags);
        var["gBackgroundMaskBuffer"] = pass.mpDevice->createTexture2D(1, 1, ResourceFormat::R32Uint, 1, 1, nullptr, flags);
        auto cb = var["CB"];
        cb["gResolution"] = uint2(1);
        cb["gSpp"] = 8u;
        cb["gAlphaLossWeight"] = 2.0f;
        cb["gForegroundAlphaMin"] = 0.95f;
        cb["gBackgroundAlphaMax"] = 0.0001f;
        for (uint32_t k : {2u, 1u})
        {
            cb["gSampleCount"] = k;
            loss->execute(ctx, uint3(1));
            auto d = pixel(ctx, derivatives);
            const auto color = k == 2 ? previous : current;
            for (uint32_t c = 0; c < 3; ++c) near(d[c], 2 * (color[c] - (reference[c] * 0.5f + 0.5f)), "Prefix RGB uses wrong samples or clamps weights");
            near(d.w, 4 * (color.w - reference.w), "Prefix alpha uses wrong samples or clamps weights");
        }
        std::cout << "PASS: corrected prefix excludes current path, keeps out-of-range estimates and 1 SPP fallback\n";
    }

    static void paths(VoxelReconstructionNoLightTransport& pass, RenderContext* ctx)
    {
        GridData grid = {};
        grid.voxelCount = uint3(8);
        grid.voxelSize = float3(1);
        grid.activeVoxelCount = grid.solidVoxelCount = 1u;
        auto resources = pass.allocateSparseGrid(ctx, grid, 1u);
        auto block = pass.createSparseGridBlock(resources);
        pass.commitSparseGrid(std::move(resources), block, 8u);
        const uint32_t cell = 3u + 8u * 3u + 64u * 3u;
        VoxelData voxel = {};
        voxel.occupied = 1;
        voxel.ellipsoid.center = float3(0.5f);
        voxel.ellipsoid.rotation = float4(1, 0, 0, 0);
        voxel.radiance.coefficients[0] = float3(0.2f, 0.4f, 0.6f) / 0.28209479177387814f;
        pass.uploadSparseBatch(ctx, block, 0, &cell, 1, &voxel);
        pass.barrierSparseVoxels(ctx);
        for (const auto& page : pass.mGridResources.indexPages) ctx->uavBarrier(page.get());

        auto scene = SceneBuilder(pass.mpDevice, Settings()).getScene();
        ProgramDesc desc;
        desc.addShaderLibrary(RayMarchingShaderFilePath).psEntry("main");
        desc.setShaderModel(ShaderModel::SM6_5);
        desc.addTypeConformances(scene->getTypeConformances());
        auto defines = scene->getSceneDefines();
        defines.add(pass.getReconstructionDefines());
        defines.add("CHECK_PRIMITIVE", "1");
        defines.add("USE_ENV_MAP", "0");
        auto forward = FullScreenPass::create(pass.mpDevice, desc, defines);
        const auto flags = ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess;
        auto pathBuffer = pass.mpDevice->createStructuredBuffer(sizeof(PathRecord), 1, flags);
        auto accumulation = pass.mpDevice->createTexture2D(1, 1, ResourceFormat::RGBA32Float, 1, 1, nullptr, flags);
        auto output = pass.mpDevice->createTexture2D(1, 1, ResourceFormat::RGBA32Float, 1, 1, nullptr,
            ResourceBindFlags::ShaderResource | ResourceBindFlags::RenderTarget);
        auto fbo = Fbo::create(pass.mpDevice);
        fbo->attachColorTarget(output, 0);
        auto var = forward->getRootVar();
        scene->bindShaderData(var["gScene"]);
        var["gGridDataParamBlock"] = block;
        var["gPathRecordBuffer"] = pathBuffer;
        var["gAccuColor"] = accumulation;
        var["GridData"]["gridMin"] = float3(0);
        var["GridData"]["voxelSize"] = float3(1);
        var["GridData"]["voxelCount"] = uint3(8);
        var["GridData"]["solidVoxelCount"] = 1u;
        auto cb = var["CB"];
        cb["pixelCount"] = uint2(1);
        cb["enableReconstruction"] = true;
        cb["usePixelFrustum"] = false;
        cb["drawMode"] = uint32_t(ABSDFDrawMode::Default);
        cb["renderBackGround"] = false;
        cb["clearColor"] = float4(0.7f, 0.8f, 0.9f, 0);
        cb["maxContributingVoxelCount"] = 16u;
        cb["transmittanceThreshold"] = 0.001f;
        cb["contourUniformFraction"] = 0.5f;
        cb["invSpp"] = 1.0f / 32;
        ContourHistory h = {};
        h.meanAlpha[0] = 1;
        const uint32_t slot = 1;
        auto table = pass.mpDevice->createStructuredBuffer(sizeof(ContourHistory), 1, flags, MemoryType::DeviceLocal, &h);
        auto observations = pass.mpDevice->createStructuredBuffer(sizeof(ContourObservation), 16, flags);
        var["gContourLookup"] = pass.mpDevice->createTexture2D(1, 1, ResourceFormat::R32Uint, 1, 1, &slot);
        var["gContourHistory"] = table;
        var["gContourObservations"] = observations;
        auto invVP = float4x4::identity();
        invVP[0][0] = invVP[1][1] = 0.001f;
        invVP[2][2] = 10;
        invVP[0][3] = invVP[1][3] = 3.5f;
        invVP[2][3] = -1;
        PathRecord path = {};
        for (uint32_t trial = 0; trial < 3; ++trial)
        {
            const bool guided = trial != 1;
            cb["useContourSampling"] = guided;
            invVP[0][3] = trial == 2 ? -2.0f : 3.5f; // The last trial misses the entire scene.
            cb["invVP"] = invVP;
            ctx->clearUAV(observations->getUAV().get(), uint4(0));
            ctx->clearUAV(accumulation->getUAV().get(), float4(0));
            float4 sum(0);
            for (uint32_t s = 0; s < 32; ++s)
            {
                cb["frameIndex"] = s;
                forward->execute(ctx, fbo);
                ctx->uavBarrier(observations.get());
                ctx->uavBarrier(accumulation.get());
                auto result = pixel(ctx, output);
                pathBuffer->getBlob(&path, 0, sizeof(path));
                require(path.valid == (trial == 2 ? 0u : 1u), "Production forward hit fixture failed");
                const float4 raw = trial == 2 ? float4(0.7f, 0.8f, 0.9f, 0) : float4(0.45f, 0.6f, 0.75f, 0.5f);
                const float weight = path.inverseSamplingPdf;
                if (!guided) near(weight, 1, "Uniform path has non-unit PDF weight");
                for (uint32_t c = 0; c < 3; ++c) near(result[c], 1 + (raw[c] - 1) * weight, "Production forward RGB is not PDF corrected");
                near(result.w, raw.w * weight, "Production forward alpha is not PDF corrected");
                if (path.valid) near(path.voxels[0].opacity, 0.5f, "Path stores weighted instead of raw opacity");
                sum += result / 32.0f;
            }
            auto accumulated = pixel(ctx, accumulation);
            require(length(accumulated - sum) < 2e-5f, "Accumulation did not retain corrected samples");
            ContourObservation observed[16];
            observations->getBlob(observed, 0, sizeof(observed));
            uint32_t total = 0;
            for (const auto& o : observed) { total += o.count; near(o.alphaSum, float(o.count) * (trial == 2 ? 0.0f : 0.5f), "History collected PDF-weighted alpha"); }
            require(total == (guided ? 32u : 0u), "History dropped misses or collected uniform-disabled samples");
        }
        std::cout << "PASS: production forward hit/miss, raw path records, corrected accumulation and disabled fallback\n";

        ProgramDesc gradDesc;
        gradDesc.addShaderLibrary(GradientPassShaderFilePath).csEntry("main");
        auto gradient = ComputePass::create(pass.mpDevice, gradDesc, pass.getReconstructionDefines(), true);
        var = gradient->getRootVar();
        var["gGridDataParamBlock"] = block;
        var["gPathRecordBuffer"] = pathBuffer;
        var["dummy"] = accumulation;
        cb = var["CB"];
        cb["gResolution"] = uint2(1);
        cb["gVoxelCount"] = uint3(8);
        cb["gGeometryTauWorld"] = 0.12f;
        cb["gGeometryGradClamp"] = 0.001f; // Ensure weighting survives a saturated proxy clamp.
        cb["gAlphaGeometryWeight"] = 1.0f;
        cb["gBackgroundCarveWeight"] = 0.01f;
        cb["gCollectDeletionEvidence"] = false;
        for (uint32_t branch = 0; branch < 3; ++branch)
        {
            path = {};
            path.viewDir = float3(0, 0, 1);
            path.backgroundColor = float3(1);
            path.valid = branch == 1 ? 0u : 1u;
            path.contributingVoxelCount = path.valid;
            path.voxels[0].opacity = 0.5f;
            path.voxels[0].radiance = float3(0.2f, 0.4f, 0.6f);
            path.voxels[0].transmittanceBefore = 1;
            path.voxels[0].gmin = -0.01f;
            path.voxels[0].xStar = float3(4.49f, 3.5f, 3.5f);
            if (branch == 1)
            {
                path.candidateCount = 1;
                path.candidates[0].valid = 1;
                path.candidates[0].gmin = 0.01f;
                path.candidates[0].xStar = float3(4.51f, 3.5f, 3.5f);
                path.candidates[0].transmittanceBefore = 1;
            }
            const uint32_t classification = branch == 2 ? 1u : 0u;
            const float4 derivative(0.1f, 0.2f, 0.3f, branch == 1 ? -10.0f : 10.0f);
            var["gDL_dColorBuffer"] = pass.mpDevice->createTexture2D(1, 1, ResourceFormat::RGBA32Float, 1, 1, &derivative);
            var["gBackgroundMaskBuffer"] = pass.mpDevice->createTexture2D(1, 1, ResourceFormat::R32Uint, 1, 1, &classification);
            GradRecord gradients[3];
            for (uint32_t i = 0; i < 3; ++i)
            {
                path.inverseSamplingPdf = i == 0 ? 1.0f : (i == 1 ? 2.0f : 0.0f);
                pathBuffer->setBlob(&path, 0, sizeof(path));
                pass.clearSparseGradients(ctx);
                gradient->execute(ctx, uint3(1));
                pass.barrierSparseGradients(ctx);
                ctx->submit(true);
                pass.mGridResources.gradPages[0]->getBlob(&gradients[i], 0, sizeof(GradRecord));
            }
            require(length(gradients[0].center) > 0, "Geometry weight test has zero gradient");
            for (uint32_t i = 1; i < 3; ++i)
            {
                const float weight = i == 1 ? 2.0f : 1.0f;
                require(length(gradients[i].center - weight * gradients[0].center) < 1e-6f, "Geometry PDF correction lost after clamp");
                require(length(gradients[i].logScale - weight * gradients[0].logScale) < 1e-6f, "Scale PDF correction incorrect");
                require(gradients[i].geometryValid == gradients[0].geometryValid, "PDF changed valid gradient count");
                for (uint32_t j = 0; j < SH_COUNT; ++j)
                {
                    require(length(gradients[i].radianceGrad.coefficients[j] - weight * gradients[0].radianceGrad.coefficients[j]) < 1e-5f,
                        "Radiance PDF correction missing or doubled");
                    near(gradients[i].opacityGrad.coefficients[j], weight * gradients[0].opacityGrad.coefficients[j], "Opacity PDF correction missing or doubled");
                }
            }
        }
        std::cout << "PASS: appearance, hard-hit, near-miss and carve gradients weighted exactly once after proxy clamp\n";
    }

    static void singleSppTraining(VoxelReconstructionNoLightTransport& pass, RenderContext* ctx)
    {
        // Call the production execute() loop, including scheduling, loss,
        // backward, Adam, reduction and history commit, with one contour pixel.
        pass.mpScene = SceneBuilder(pass.mpDevice, Settings()).getScene();
        pass.createRayMarchingPassResource(ctx);
        pass.mRayMarchingPass.mOutputResolution = uint2(1);
        pass.mRayMarchingPass.mClearColor = float3(1);
        pass.createLossPassResource(ctx);
        pass.createGradientPassResource(ctx);
        pass.createUpdatePassResource(ctx);
        pass.mUpdatePass.mpComputePass->getRootVar(); // Commit binds optimizer pages.
        pass.createReducePassResource(ctx);
        pass.mUpdatePass.mLrCenter = pass.mUpdatePass.mLrShape = pass.mUpdatePass.mLrRotation = 0.0f;
        pass.mUpdatePass.mLrRadiance = pass.mUpdatePass.mLrOpacity = 0.001f;
        pass.mUpdatePass.mOpacityWarmupIterations = pass.mUpdatePass.mOpacityRampIterations = 0u;
        pass.mTopologySettings.collectDeletionEvidence = pass.mTopologySettings.enableGrowth = false;
        pass.mTopologySettings.candidateCount = 0u;
        pass.mReconstructionMode = 0u;
        pass.mInitVoxelData = pass.mPointCloud.startRequested = false;
        pass.mLoadReconstructionRequested = pass.mSaveReconstructionRequested = false;
        pass.mReferenceDataError.clear();
        auto camera = Camera::create("single-spp-test");
        camera->setPosition(float3(3.5f, 3.5f, -2.0f));
        camera->setTarget(float3(3.5f, 3.5f, 3.5f));
        camera->setUpVector(float3(0, 1, 0));
        camera->setAspectRatio(1.0f);
        camera->setFocalLength(1000.0f); // Every jitter hits the test cell.
        camera->setNearPlane(0.1f);
        camera->setFarPlane(20.0f);
        const float4 reference(0.1f, 0.2f, 0.3f, 0.25f);
        auto refImage = pass.mpDevice->createTexture2D(1, 1, ResourceFormat::RGBA32Float, 1, 1, &reference);
        pass.mReferenceCameras = {camera, camera};
        pass.mReferenceImages = {refImage, refImage};
        pass.mReferenceImagePaths = {"single-spp-view-0", "single-spp-view-1"};
        const auto flags = ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess;
        pass.mpPathRecordBuffer = pass.mpDevice->createStructuredBuffer(sizeof(PathRecord), 1u, flags);
        auto output = pass.mpDevice->createTexture2D(1, 1, ResourceFormat::RGBA32Float, 1, 1, nullptr,
            flags | ResourceBindFlags::RenderTarget);
        auto accumulation = pass.mpDevice->createTexture2D(1, 1, ResourceFormat::RGBA32Float, 1, 1, nullptr, flags);
        auto dummy = pass.mpDevice->createTexture2D(1, 1, ResourceFormat::RGBA32Float, 1, 1, nullptr, flags);
        ResourceCache cache;
        Dictionary dictionary;
        const std::string name = "singleSpp";
        cache.registerExternalResource(name + "." + kOutputColor, output);
        cache.registerExternalResource(name + "." + kAccumulateOutputColor, accumulation);
        cache.registerExternalResource(name + ".dummy", dummy);
        struct TestRenderData : RenderData
        {
            TestRenderData(const std::string& name, ResourceCache& cache, Dictionary& dictionary)
                : RenderData(name, cache, dictionary, uint2(1), ResourceFormat::RGBA32Float) {}
        } data(name, cache, dictionary);

        for (bool guided : {false, true})
        {
            pass.resetContourSampling();
            pass.mContourSampling = {};
            pass.mContourSampling.enabled = guided;
            pass.mOptimizerParams.reset();
            pass.mEnableReconstruction = pass.mOptimizerParams.isRunning = true;
            pass.mOptimizerParams.viewsPerIteration = 2u;
            pass.mOptimizerParams.maxIteration = 100u;
            pass.mRayMarchingPass.mSpp = 1u;
            pass.mRayMarchingPass.mSampleIndex = 0u;
            pass.mReduceLossPass.iterationLossSum = 0.0f;
            pass.mReduceLossPass.iterationLossCount = 0u;
            pass.mReduceLossPass.iterationLossHistory.clear();
            GridData grid = {};
            grid.voxelCount = uint3(8);
            grid.voxelSize = float3(1);
            grid.solidVoxelCount = grid.activeVoxelCount = 1u;
            auto resources = pass.allocateSparseGrid(ctx, grid, 1u);
            auto block = pass.createSparseGridBlock(resources);
            pass.commitSparseGrid(std::move(resources), block, 8u);
            VoxelData initial = {};
            initial.occupied = 1u;
            initial.ellipsoid.center = float3(0.5f);
            initial.ellipsoid.rotation = float4(1, 0, 0, 0);
            initial.radiance.coefficients[0] = float3(0.3f) / 0.28209479177387814f;
            const uint32_t cell = 3u + 8u * 3u + 64u * 3u;
            pass.uploadSparseBatch(ctx, block, 0u, &cell, 1u, &initial);
            pass.barrierSparseVoxels(ctx);
            for (const auto& page : pass.mGridResources.indexPages) ctx->uavBarrier(page.get());
            bool sawNonUniform = false;
            for (uint32_t frame = 0; frame < 6; ++frame)
            {
                if (frame == 4u)
                {
                    // Start an 8-SPP batch, then reproduce the UI's restart
                    // when switching to 1 SPP after a backward sample exists.
                    pass.mRayMarchingPass.mSpp = 8u;
                    pass.execute(ctx, data);
                    pass.execute(ctx, data);
                    require(pass.mRayMarchingPass.mSampleIndex == 2u && pass.mViewGradientSampleCount == 1u,
                        "Multi-SPP setup did not accumulate one backward sample");
                    GeometryAdamState adam;
                    pass.mGridResources.adamPages[0]->getBlob(&adam, 0, sizeof(adam));
                    require(adam.opacitySteps == frame, "Partial multi-SPP batch updated the optimizer");
                    pass.mRayMarchingPass.mSpp = 1u;
                    pass.mRayMarchingPass.mSampleIndex = 0u;
                }
                pass.execute(ctx, data);
                ctx->submit(true);
                require(pass.mRayMarchingPass.mSampleIndex == 0u && pass.mViewGradientSampleCount == 1u,
                    "1 SPP skipped backward or failed to complete its batch");
                require(pass.mOptimizerParams.currentView == (frame + 1u) % 2u &&
                    pass.mOptimizerParams.currentIteration == (frame + 1u) / 2u, "1 SPP stalled view/iteration scheduling");
                require(!pass.mContourBatchActive, "1 SPP failed to commit its sampling history");
                const float4 current = pixel(ctx, output);
                require(length(pixel(ctx, accumulation) - current) < 1e-6f, "1 SPP retained an old accumulated image");
                const float4 d = pixel(ctx, pass.mLossPass.dL_dColor);
                float expectedLoss = 0.0f;
                for (uint32_t channel = 0; channel < 4; ++channel)
                {
                    const float target = channel == 3 ? reference.w : reference[channel] * reference.w + 1 - reference.w;
                    const float lossWeight = channel == 3 ? pass.mLossPass.alphaLossWeight : 1.0f;
                    const float residual = current[channel] - target;
                    near(d[channel], 2 * lossWeight * residual, "1 SPP residual is not the current PDF-corrected sample");
                    expectedLoss += lossWeight * residual * residual;
                }
                near(pass.mReduceLossPass.meanLoss, expectedLoss, "1 SPP reduced loss is stale or non-finite");
                GeometryAdamState adam;
                pass.mGridResources.adamPages[0]->getBlob(&adam, 0, sizeof(adam));
                require(adam.opacitySteps == frame + 1u, "1 SPP did not perform exactly one optimizer update");
                GradRecord grad;
                pass.mGridResources.gradPages[0]->getBlob(&grad, 0, sizeof(grad));
                require(grad.appearanceValid == 1u, "Restart retained gradients from the previous batch");
                if (guided)
                {
                    ContourObservation observations[16];
                    pass.mpContourObservations->getBlob(observations, 0, sizeof(observations));
                    uint32_t count = 0;
                    for (const auto& observation : observations) count += observation.count;
                    require(count == 1u, "1 SPP history mixed different batches");
                    const auto updated = history(pass, pass.mContourBatchView, 0u);
                    bool changed = false;
                    for (float alpha : updated.meanAlpha) changed |= std::abs(alpha - reference.w) > 1e-5f;
                    require(changed, "1 SPP failed to learn from its sole observation");
                    PathRecord path;
                    pass.mpPathRecordBuffer->getBlob(&path, 0, sizeof(path));
                    sawNonUniform |= std::abs(path.inverseSamplingPdf - 1.0f) > 1e-5f;
                }
            }
            require(!guided || sawNonUniform, "1 SPP did not use learned history on revisits");
            VoxelData optimized;
            pass.mGridResources.voxelPages[0]->getBlob(&optimized, 0, sizeof(optimized));
            require(std::abs(optimized.opacity.coefficients[0]) > 1e-6f, "1 SPP did not change voxel parameters");
        }
        std::cout << "PASS: full 1-SPP training with guiding on/off, learned revisits, per-frame loss/backward/Adam/history and 8 -> 1 restart\n";
    }

    static void run(VoxelReconstructionNoLightTransport& pass, RenderContext* ctx)
    {
        lifecycle(pass, ctx);
        distribution(pass, ctx);
        prefix(pass, ctx);
        paths(pass, ctx);
        singleSppTraining(pass, ctx);
    }
};

void runContourSamplingTests(VoxelReconstructionNoLightTransport& pass, RenderContext* ctx)
{
    ContourSamplingTestAccess::run(pass, ctx);
}

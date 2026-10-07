#include "Falcor.h"
#include "Core/Pass/FullScreenPass.h"
#include "Scene/SceneBuilder.h"
#include "Scene/Lights/EnvMap.h"
#include "../Voxel/VoxelData.slang"
#include "../PathRecord.slang"
#include <cmath>
#include <cstring>
#include <iostream>

using namespace Falcor;

namespace
{
constexpr float pi = 3.14159265359f;
const char* shaderRoot = "RenderPasses/VoxelReconstructionNoLightTransport/";

void require(bool condition, const char* message)
{
    if (!condition) throw std::runtime_error(message);
}

float4x4 orthographicVP()
{
    auto vp = float4x4::identity();
    vp[0][0] = vp[1][1] = 2.0f;
    vp[0][3] = vp[1][3] = -1.0f;
    vp[2][2] = 0.1f;
    return vp;
}

GaussianEllipsoid ellipsoid(float3 center, float3 axes)
{
    GaussianEllipsoid e = {};
    e.center = center;
    e.logScale = log(axes);
    e.rotation = float4(1, 0, 0, 0);
    return e;
}

// Independent dense ray integral; fixtures use identity rotations.
float coverageOracle(const GaussianEllipsoid& e, int3 cell, const float4x4& invVP)
{
    constexpr uint32_t n = 1536;
    uint32_t hits = 0;
    const float3 axes = exp(e.logScale);
    for (uint32_t y = 0; y < n; ++y)
        for (uint32_t x = 0; x < n; ++x)
        {
            const float sx = 2.0f * (float(x) + 0.5f) / float(n) - 1.0f;
            const float sy = 1.0f - 2.0f * (float(y) + 0.5f) / float(n);
            float4 a = mul(invVP, float4(sx, sy, 0, 1));
            float4 b = mul(invVP, float4(sx, sy, 1, 1));
            const float3 from = float3(a.x, a.y, a.z) / a.w - float3(cell);
            const float3 delta = float3(b.x, b.y, b.z) / b.w - float3(a.x, a.y, a.z) / a.w;
            double entry = 0, exit = 1;
            bool valid = true;
            for (uint32_t k = 0; k < 3; ++k)
            {
                if (std::abs(delta[k]) < 1e-15f)
                {
                    valid = valid && from[k] >= 0 && from[k] <= 1;
                    continue;
                }
                double u0 = -double(from[k]) / double(delta[k]);
                double u1 = (1.0 - double(from[k])) / double(delta[k]);
                if (u0 > u1) std::swap(u0, u1);
                entry = std::max(entry, u0);
                exit = std::min(exit, u1);
            }
            if (!valid || entry > exit) continue;
            double qa = 0, qb = 0, qc = -1;
            for (uint32_t k = 0; k < 3; ++k)
            {
                const double offset = double(from[k] - e.center[k]) / double(axes[k]);
                const double step = double(delta[k]) / double(axes[k]);
                qa += step * step;
                qb += 2 * offset * step;
                qc += offset * offset;
            }
            const double discriminant = qb * qb - 4 * qa * qc;
            if (discriminant < 0) continue;
            const double root = std::sqrt(discriminant);
            entry = std::max(entry, (-qb - root) / (2 * qa));
            exit = std::min(exit, (-qb + root) / (2 * qa));
            if (entry <= exit) ++hits;
        }
    return float(hits) / float(n * n);
}

void testCoverage(const ref<Device>& device)
{
    auto ctx = device->getRenderContext();
    auto pass = ComputePass::create(device, std::string(shaderRoot) + "Tests/PixelCoverage.cs.slang", "main");
    auto result = device->createStructuredBuffer(sizeof(float), 1u, ResourceBindFlags::UnorderedAccess);
    auto var = pass->getRootVar();
    var["result"] = result;
    const auto run = [&](const GaussianEllipsoid& e, const float4x4& vp)
    {
        auto cb = var["CB"];
        cb["ellipsoid"]["center"] = e.center;
        cb["ellipsoid"]["logScale"] = e.logScale;
        cb["ellipsoid"]["rotation"] = e.rotation;
        cb["vp"] = vp;
        pass->execute(ctx, uint3(1));
        ctx->submit(true);
        float area;
        result->getBlob(&area, 0, sizeof(area));
        require(std::isfinite(area), "Coverage produced non-finite area");
        return area;
    };
    const auto check = [&](const char* name, const GaussianEllipsoid& e, const float4x4& vp, float expected, float tolerance)
    {
        const float area = run(e, vp);
        std::cout << name << ": " << area << " expected " << expected << '\n';
        require(std::abs(area - expected) <= tolerance, name);
    };
    const auto vp = orthographicVP();
    check("sphere area", ellipsoid(float3(0.5f), float3(0.1f)), vp, pi * 0.01f, 2e-8f);
    check("empty cell intersection", ellipsoid(float3(2), float3(0.1f)), vp, 0, 1e-7f);
    check("whole-cell coverage", ellipsoid(float3(0.5f), float3(2)), vp, 1, 1e-5f);
    check("cell-clipped half sphere", ellipsoid(float3(0, 0.5f, 0.5f), float3(0.2f)), vp, pi * 0.02f, 3e-8f);
    check("off-center subpixel structure", ellipsoid(float3(0.2f, 0.3f, 0.5f), float3(0.002f, 0.1f, 0.1f)), vp,
        pi * 0.0002f, 2e-9f);
    check("tiny projected sphere", ellipsoid(float3(0.5f), float3(1e-5f)), vp, pi * 1e-10f, 3e-16f);
    auto rotated = ellipsoid(float3(0.5f), float3(0.2f, 0.05f, 0.1f));
    rotated.rotation = float4(std::cos(0.37f), 0, 0, std::sin(0.37f));
    check("rotated ellipse", rotated, vp, pi * 0.01f, 3e-8f);
    const auto sphere = ellipsoid(float3(0.5f), float3(0.1f));
    check("cell-clipped quarter sphere", ellipsoid(float3(0, 0, 0.5f), float3(0.2f)), vp, pi * 0.01f, 3e-8f);
    auto pixelCrop = vp;
    pixelCrop[0][0] = pixelCrop[1][1] = 4.0f;
    pixelCrop[0][3] = -3.2f;
    pixelCrop[1][3] = -2.0f;
    const float cap = 0.01f * std::acos(0.5f) - 0.05f * std::sqrt(0.0075f);
    check("pixel-clipped circular segment", sphere, pixelCrop, 4.0f * cap, 5e-8f);
    pixelCrop[0][3] = pixelCrop[1][3] = -3.0f;
    check("pixel-clipped quarter circle", sphere, pixelCrop, pi * 0.01f, 2e-8f);
    auto narrowPixel = vp;
    narrowPixel[0][0] = narrowPixel[1][1] = 1600.0f;
    narrowPixel[0][3] = -960.0f;
    narrowPixel[1][3] = -800.0f;
    const double h = 1.0 / 1600.0;
    const double radius = double(exp(sphere.logScale).x);
    const double expectedNarrow = (h * std::sqrt(radius * radius - h * h) + radius * radius * std::asin(h / radius)
        - 2.0 * h * (0.1 - h)) / (4.0 * h * h);
    check("small pixel at curved silhouette", sphere, narrowPixel, float(expectedNarrow), 2e-5f);
    auto widePixel = vp;
    widePixel[0][0] = widePixel[1][1] = 1.0f;
    widePixel[0][3] = widePixel[1][3] = -0.5f;
    check("cube silhouette edges", ellipsoid(float3(0.5f), float3(2)), widePixel, 0.25f, 1e-8f);
    // Both separate 2D projections cover the pixel, but their ray depth
    // intervals are disjoint. Actual 3D-truncated coverage must be zero.
    auto grazingInvVP = float4x4::zeros();
    grazingInvVP[0][2] = 10.0f;
    grazingInvVP[1][0] = 0.001f;
    grazingInvVP[1][2] = 1.0f;
    grazingInvVP[1][3] = -0.095f;
    grazingInvVP[2][1] = 0.001f;
    grazingInvVP[2][3] = 0.5f;
    grazingInvVP[3][3] = 1.0f;
    check("disjoint voxel/ellipsoid depth", ellipsoid(float3(0.5f), float3(0.6f)), inverse(grazingInvVP), 0, 1e-8f);
    auto perspective = float4x4::zeros();
    perspective[0][0] = perspective[1][1] = 2;
    perspective[0][3] = perspective[1][3] = -1;
    perspective[3][2] = 1;
    perspective[3][3] = 2;
    perspective[2][2] = 10.0f / 9.9f;
    perspective[2][3] = 1.9f * 10.0f / 9.9f;
    check("perspective sphere", sphere, perspective, pi * 0.01f / (2.5f * 2.5f - 0.01f), 5e-9f);
    const float near = 2.55f;
    perspective[2][2] = 10.0f / (10.0f - near);
    perspective[2][3] = (2.0f - near) * 10.0f / (10.0f - near);
    check("near-plane clipped sphere", sphere, perspective, pi * 0.0075f / (near * near), 3e-8f);
    // Cap circles can project to hyperbolas or approach a parabola when their
    // planes are close to the camera. Validate finite pixel-clipped arcs.
    const auto clipped = ellipsoid(float3(0.5f), float3(0.6f));
    for (float cameraZ : {0.5f, 0.5f - std::sqrt(0.11f)})
    {
        auto camera = Camera::create();
        const float3 eye(-0.11f, 0.5f, cameraZ);
        camera->setPosition(eye);
        camera->setTarget(eye + float3(0, 0, 1));
        camera->setUpVector(float3(0, 1, 0));
        camera->setAspectRatio(1.0f);
        camera->setFocalLength(12.0f);
        camera->setDepthRange(0.05f, 10.0f);
        const auto nearCameraVP = camera->getViewProjMatrixNoJitter();
        const float expected = coverageOracle(clipped, int3(0), inverse(nearCameraVP));
        const float actual = run(clipped, nearCameraVP);
        std::cout << "hyperbolic/parabolic cap, cameraZ=" << cameraZ << ": " << actual << " expected " << expected << '\n';
        require(std::abs(actual - expected) < 8e-5f, "Near-camera clipped conic area failed");
    }
}

void testForward(const ref<Device>& device)
{
    auto ctx = device->getRenderContext();
    auto scene = SceneBuilder(device, Settings()).getScene();
    const float4 whitePixels[2] = {float4(1), float4(1)};
    auto environment = EnvMap::create(device, device->createTexture2D(2, 1, ResourceFormat::RGBA32Float,
        1, 1, whitePixels, ResourceBindFlags::ShaderResource));
    environment->setTint(float3(0, 0, 1));
    scene->setEnvMap(environment);
    ProgramDesc desc;
    desc.addShaderLibrary(std::string(shaderRoot) + "Shader/RayMarchingPass.ps.slang").psEntry("main");
    desc.setShaderModel(ShaderModel::SM6_5);
    desc.addTypeConformances(scene->getTypeConformances());
    auto defines = scene->getSceneDefines();
    defines.add("CHECK_PRIMITIVE", "1");
    defines.add("USE_ENV_MAP", "1");
    defines.add("SPARSE_POOL_PAGE_SIZE", "32u");
    defines.add("SPARSE_POOL_MAX_PAGES", "2");
    auto pass = FullScreenPass::create(device, desc, defines);
    auto var = pass->getRootVar();
    scene->bindShaderData(var["gScene"]);
    environment->bindShaderData(var["gScene"]["envMap"]);
    auto block = ParameterBlock::create(device, pass->getProgram()->getReflector()->getParameterBlock("gGridDataParamBlock"));
    auto grid = block->getRootVar();
    std::vector<int32_t> indices(512, -1);
    // IDs intentionally reverse depth order; this catches enumeration-order compositing.
    indices[0 + 8 * 0 + 64 * 2] = 1;
    indices[0 + 8 * 0 + 64 * 4] = 0;
    const auto flags = ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess;
    auto index = device->createTexture3D(8, 8, 8, ResourceFormat::R32Int, 1u, indices.data(), flags);
    std::vector<VoxelData> voxels(32);
    for (uint32_t i = 0; i < 2; ++i)
    {
        voxels[i].occupied = 1;
        voxels[i].ellipsoid = ellipsoid(float3(0.2f, 0.3f, 0.5f), float3(0.02f, 0.1f, 0.1f));
        // Opacity logit zero -> 0.5; SH DC produces unit red/green.
        voxels[i].radiance.coefficients[0] = i == 1 ? float3(1.0f / 0.28209479177f, 0, 0)
                                                   : float3(0, 1.0f / 0.28209479177f, 0);
    }
    auto data = device->createStructuredBuffer(sizeof(VoxelData), 32u, flags, MemoryType::DeviceLocal, voxels.data());
    grid["voxelCount"] = uint3(8);
    grid["activeVoxelCount"] = 2u;
    grid["indexPageCount"] = uint3(1);
    grid["indexPages"][0] = index;
    grid["voxelPages"][0] = data;
    var["gGridDataParamBlock"] = block;
    var["gPathRecordBuffer"] = device->createStructuredBuffer(sizeof(PathRecord), 1u, flags);
    auto accu = device->createTexture2D(1, 1, ResourceFormat::RGBA32Float, 1, 1, nullptr, flags);
    var["gAccuColor"] = accu;
    var["GridData"]["gridMin"] = float3(0);
    var["GridData"]["voxelSize"] = float3(1);
    var["GridData"]["voxelCount"] = uint3(8);
    auto cb = var["CB"];
    auto vp = orthographicVP();
    cb["pixelCount"] = uint2(1);
    cb["invVP"] = inverse(vp);
    cb["viewProjection"] = vp;
    cb["cameraPosition"] = float3(0.5f, 0.5f, 0);
    cb["cameraForward"] = float3(0, 0, 1);
    cb["cameraDepthRange"] = float2(0, 10);
    cb["usePixelFrustum"] = true;
    cb["drawMode"] = 0u;
    cb["enableReconstruction"] = false;
    cb["renderBackGround"] = true;
    cb["clearColor"] = float4(0, 0, 1, 0);
    cb["transmittanceThreshold"] = 0.0f;
    cb["maxContributingVoxelCount"] = 16u;
    cb["invSpp"] = 1.0f;
    auto output = device->createTexture2D(1, 1, ResourceFormat::RGBA32Float, 1, 1, nullptr,
        ResourceBindFlags::ShaderResource | ResourceBindFlags::RenderTarget);
    auto fbo = Fbo::create(device);
    fbo->attachColorTarget(output, 0);
    ctx->clearUAV(accu->getUAV().get(), float4(0));
    pass->execute(ctx, fbo);
    auto bytes = ctx->readTextureSubresource(output.get(), 0);
    float4 color;
    std::memcpy(&color, bytes.data(), sizeof(color));
    const float alpha = 0.5f * pi * 0.002f;
    const float3 expected(alpha, (1 - alpha) * alpha, (1 - alpha) * (1 - alpha));
    std::cout << "Forward RGBA: " << color.x << ", " << color.y << ", " << color.z << ", " << color.w << '\n';
    require(length(float3(color.x, color.y, color.z) - expected) < 0.00012f,
        "Candidate traversal, depth ordering, uniqueness or coverage-alpha compositing failed");
    require(std::abs(color.w - (1 - expected.z)) < 0.00012f, "Forward alpha failed");
    const auto readForward = [&]()
    {
        pass->execute(ctx, fbo);
        auto resultBytes = ctx->readTextureSubresource(output.get(), 0);
        float4 result;
        std::memcpy(&result, resultBytes.data(), sizeof(result));
        return result;
    };
    // A white environment contributes residual T both on early termination
    // and when the traversal exits the scene with mostly transparent pixels.
    environment->setTint(float3(1));
    scene->bindShaderData(var["gScene"]);
    environment->bindShaderData(var["gScene"]["envMap"]);
    color = readForward();
    const float residual = (1 - alpha) * (1 - alpha);
    require(length(float3(color.x, color.y, color.z) - float3(alpha + residual, (1 - alpha) * alpha + residual, residual)) < 0.00012f,
        "White environment contribution after scene exit failed");
    cb["transmittanceThreshold"] = 1.0f - 0.5f * alpha;
    color = readForward();
    require(length(float3(color.x, color.y, color.z) - float3(1, 1 - alpha, 1 - alpha)) < 0.00012f,
        "Early termination lost residual environment or processed a later voxel");
    cb["transmittanceThreshold"] = 0.0f;
    // Move the near plane through the red cell. Skip that whole cell, even
    // though the red ellipsoid itself remains partly in front of the plane.
    auto crossingVP = vp;
    crossingVP[2][3] = -2.5f * crossingVP[2][2];
    cb["viewProjection"] = crossingVP;
    cb["invVP"] = inverse(crossingVP);
    cb["cameraDepthRange"] = float2(2.5f, 12.5f);
    color = readForward();
    require(length(float3(color.x, color.y, color.z) - float3(1 - alpha, 1, 1 - alpha)) < 0.00012f,
        "Near-plane intersecting cell was not skipped");
    cb["viewProjection"] = vp;
    cb["invVP"] = inverse(vp);
    cb["cameraDepthRange"] = float2(0, 10);
    cb["renderBackGround"] = false;
    color = readForward();
    require(length(float3(color.x, color.y, color.z) - float3(alpha, (1 - alpha) * alpha, 0)) < 0.00012f,
        "Disabled environment contributed clearColor");
    cb["renderBackGround"] = true;
    environment->setTint(float3(0, 0, 1));
    scene->bindShaderData(var["gScene"]);
    environment->bindShaderData(var["gScene"]["envMap"]);
    // Diagonal perspective traversal, compared to independent per-voxel ray
    // coverage, using the same c*alpha model for final compositing.
    auto camera = Camera::create();
    camera->setPosition(float3(-2, -1, -2));
    camera->setTarget(float3(0.5f, 0.5f, 3.5f));
    camera->setUpVector(float3(0, 1, 0));
    camera->setAspectRatio(1.0f);
    camera->setFocalLength(12.0f);
    camera->setDepthRange(0.1f, 30.0f);
    vp = camera->getViewProjMatrixNoJitter();
    cb["invVP"] = inverse(vp);
    cb["viewProjection"] = vp;
    cb["cameraPosition"] = camera->getPosition();
    cb["cameraForward"] = normalize(camera->getTarget() - camera->getPosition());
    cb["cameraDepthRange"] = float2(camera->getNearPlane(), camera->getFarPlane());
    const float frontCoverage = coverageOracle(voxels[1].ellipsoid, int3(0, 0, 2), inverse(vp));
    const float backCoverage = coverageOracle(voxels[0].ellipsoid, int3(0, 0, 4), inverse(vp));
    require(frontCoverage > 0 && backCoverage > 0, "Perspective oracle fixture is outside the pixel");
    const float frontAlpha = 0.5f * frontCoverage, backAlpha = 0.5f * backCoverage;
    const float3 expectedPerspective(frontAlpha, (1 - frontAlpha) * backAlpha, (1 - frontAlpha) * (1 - backAlpha));
    pass->execute(ctx, fbo);
    bytes = ctx->readTextureSubresource(output.get(), 0);
    std::memcpy(&color, bytes.data(), sizeof(color));
    std::cout << "Diagonal perspective: red=" << color.x << " expected=" << expectedPerspective.x
              << " green=" << color.y << " expected=" << expectedPerspective.y << '\n';
    require(length(float3(color.x, color.y, color.z) - expectedPerspective) < 0.00003f,
        "Diagonal perspective candidate traversal or coverage failed");
    pass->getProgram()->addDefine("USE_ENV_MAP", "0");
    color = readForward();
    require(length(float3(color.x, color.y, color.z) - float3(expectedPerspective.x, expectedPerspective.y, 0)) < 0.00003f,
        "Unloaded environment contributed clearColor");
    // Ray mode and topology mode remain executable with the new uniforms present.
    cb["usePixelFrustum"] = false;
    cb["frameIndex"] = 0u;
    pass->execute(ctx, fbo);
    bytes = ctx->readTextureSubresource(output.get(), 0);
    std::memcpy(&color, bytes.data(), sizeof(color));
    require(std::isfinite(color.x) && std::isfinite(color.w), "Ray mode produced non-finite output");
}
}

int main()
{
    try
    {
        setErrorDiagnosticFlags(ErrorDiagnosticFlags::None);
        Device::Desc desc;
        desc.type = Device::Type::D3D12;
        desc.enableDebugLayer = true;
        auto device = make_ref<Device>(desc);
        testCoverage(device);
        testForward(device);
        device->wait();
        std::cout << "All pixel-frustum GPU tests passed.\n";
        return 0;
    }
    catch (const std::exception& e)
    {
        std::cerr << "FAIL: " << e.what() << '\n';
        return 1;
    }
}

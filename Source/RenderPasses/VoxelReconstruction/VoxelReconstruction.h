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

#include "Defines.h"
#include "Voxel/VoxelData.slang"
#include "Voxel/VoxelGrid.slang"
#include "Voxel/ABSDF.slang"

using namespace Falcor;

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
    virtual bool onMouseEvent(const MouseEvent& mouseEvent) override { return false; }
    virtual bool onKeyEvent(const KeyboardEvent& keyEvent) override { return false; }

    void beginFrame(RenderContext* pRenderContext, bool forceReset = false);
    void endFrame(RenderContext* pRenderContext);


    void setupGridResouce(RenderContext* pRenderContext, bool forceReset);
    void proccessXuData(RenderContext* pRenderContext, const RenderData& renderData);
    void UpdateVoxelGrid(ref<Scene> scene, uint voxelResolution);

    void createRayMarchingPassResource(RenderContext* pRenderContext);
    void rayMarchingPass(RenderContext* pRenderContext, const RenderData& renderData);

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
        //bool mCheckVisibility;
        uint mDrawMode;
        bool mUseMipmap;
        uint mMaxBounce;
        bool mRenderBackGround;
        float3 mClearColor;
        bool mCheckEllipsoid;

        void init() {
            mOptionsChanged = false;
            mFrameIndex = 0;
            mOutputResolution = uint2(1920, 1080);
            mDisplayNDF = false;
            //mCheckVisibility = false;
            mDrawMode = 0;
            mUseMipmap = true;
            mMaxBounce = 1;
            mRenderBackGround = false;
            mClearColor = float3(0);
            mCheckEllipsoid = false;
        }
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


    // Passes
    ref<ComputePass> mpReflectTypes;
    ref<ComputePass> mpProcessXuDataPass;
    


    // Grid
    GridResources mGridResources;  // cpu中的对应gpu中的资源，变量赋值，buffer绑定
    ref<ParameterBlock> mpGridBlock; // gpu的block

    // RayMarchingPass
    RayMarchingPassResouce mRayMarchingPassResouce;
    RayMarchingPassParams mRayMarchingPassParams;
    uint3 MinFactor = uint3(1, 1, 1);

    // UI
    

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

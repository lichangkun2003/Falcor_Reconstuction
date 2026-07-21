import DiffRendering.SceneGradientInfo;
import DiffRendering.SceneGradients;
import DiffRendering.GradientIOWrapper;

#include "../Voxel/VoxelGrid.slang"

struct DiffVoxelIO
{
    [Differentiable]
    float3 loadSHCoeff(SceneGradientInfo gradInfo, no_diff uint solidVoxelID, no_diff uint shID)
    {
        // 实际 SH buffer 读取方式
        float3 coeff = no_diff gGridDataParamBlock.gridDataBuffer[solidVoxelID] .radiance.coefficients[shID];

        uint hashIndex = hashFunction(
            gradInfo.pixelID,
            gSceneGradients.getHashSize(GradientType::VoxelSH)
        );

        uint baseOffset = solidVoxelID * SH_COUNT * 3 + shID * 3;

        GradientIOWrapper wrapper = GradientIOWrapper(
            GradientType::VoxelSH,
            baseOffset,
            hashIndex
        );

        return wrapper.getFloat(coeff, 0);
    }
};

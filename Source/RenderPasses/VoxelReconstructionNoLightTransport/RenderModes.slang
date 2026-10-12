#pragma once
#include "HostDeviceShared.slang"

BEGIN_NAMESPACE_FALCOR

enum class ReconstructionDrawMode : uint32_t
{
    Default,
    TopologyDebug,
};

FALCOR_ENUM_INFO(
    ReconstructionDrawMode,
    {
        { ReconstructionDrawMode::Default, "Default" },
        { ReconstructionDrawMode::TopologyDebug, "TopologyDebug" },
    }
);
FALCOR_ENUM_REGISTER(ReconstructionDrawMode);

enum class TopologyDebugLayer : uint32_t
{
    Occupied = 0,
    Growth = 1,
    // Preserve the existing value after removing the unimplemented Split layer.
    Deletion = 3,
};

FALCOR_ENUM_INFO(
    TopologyDebugLayer,
    {
        { TopologyDebugLayer::Occupied, "Occupied grid cells" },
        { TopologyDebugLayer::Growth, "Grown voxels" },
        { TopologyDebugLayer::Deletion, "Deletion candidates" },
    }
);
FALCOR_ENUM_REGISTER(TopologyDebugLayer);

END_NAMESPACE_FALCOR

#pragma once
#include "../HostDeviceShared.slang"
#include "../Defines.h"
BEGIN_NAMESPACE_FALCOR

static const uint SVO_CHILD_MASK = 0xffu;
static const uint SVO_LEAF_FLAG = 1u << 8u;
static const uint INVALID_INDEX = 0xffffffffu;

struct SvoNode
{
    // 内部节点：第一个有效子节点的索引。
    // 叶节点：gridDataBuffer 中的 payload index。
    uint data;

    // 内部节点：低 8 bit 是 childMask。
    // 叶节点：SVO_LEAF_FLAG。
    uint metadata;
};

struct SvoBuildItem
{
    // 当前节点在 gSvoNodes 中的索引。
    uint nodeIndex;

    // 当前节点在当前 OM level 中的坐标。
    uint3 coord;
};


#if !defined(HOST_CODE)

#endif





END_NAMESPACE_FALCOR

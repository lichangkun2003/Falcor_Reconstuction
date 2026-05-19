#pragma once
#include "HostDeviceShared.slang"

BEGIN_NAMESPACE_FALCOR

struct GradRecord
{
    float3 diffuseGrad;

    MUTATING
    void init() {

    }
};

#if !defined(HOST_CODE)


#endif


END_NAMESPACE_FALCOR

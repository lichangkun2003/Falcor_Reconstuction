#pragma once
#include "Utils/HostDeviceShared.slangh"


#include "Defines.h"


#if !defined(HOST_CODE)
#define MUTATING [mutating]
#else
#define MUTATING
#endif

#pragma once

// Shared by host code and shaders. Reconstruction uses point-cloud initialization.

#ifndef GRID_RESOLUTION
#define GRID_RESOLUTION 512
#endif

#define REFERENCE_IMAGES_COUNT 100



// Maximum hard-hit records retained for differentiable compositing along one ray.
#define MAX_CONTRIBUTING_VOXELS_PER_RAY 16


#define LOBE_COUNT 4

// Radiance
#define SH_COUNT 9


// Opacity
#define SH_OPACITY_COUNT 9

#define MAX_CANDIDATES 16

// Keep the complete spatial map, but allocate attributes only for occupied cells.
// Each pool page stays below Falcor's 4 GiB buffer limit, including SH counts up to 16.
#define SPARSE_POOL_PAGE_SIZE (1u << 18)
#define SPARSE_POOL_MAX_PAGES 256
#define SPARSE_INDEX_PAGE_EDGE 512
#define SPARSE_INDEX_MAX_PAGES 8

#if SH_COUNT < 1 || SH_COUNT > 16 || SH_OPACITY_COUNT < 1 || SH_OPACITY_COUNT > 16
#error "SH counts must be between 1 and 16."
#endif

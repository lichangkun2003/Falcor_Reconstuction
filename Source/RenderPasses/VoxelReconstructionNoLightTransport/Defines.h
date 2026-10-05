#pragma once

// Shared by host code and shaders. Reconstruction uses point-cloud initialization.

#ifndef GRID_RESOLUTION
#define GRID_RESOLUTION 512
#endif

// 0: fixed-resolution reconstruction; 1: conservative coarse-to-fine experiment.
#ifndef RECONSTRUCTION_MODE
#define RECONSTRUCTION_MODE 1
#endif
#define COARSE_TO_FINE_START_RESOLUTION 16
#define COARSE_TO_FINE_TOTAL_ITERATIONS 700

#define REFERENCE_IMAGES_COUNT 100



// Maximum hard-hit records retained for differentiable compositing along one ray.
#define MAX_CONTRIBUTING_VOXELS_PER_RAY 16


#define LOBE_COUNT 4

// Radiance
#define SH_COUNT 9


// Opacity
#define SH_OPACITY_COUNT 9

// Geometry optimization scale clamp, in voxel widths; not a growth trigger.
#define ELLIPSOID_MAX_SCALE_VOXELS 4.0f

// Shared UI/upload bound for the outward-depth growth experiment.
#define GROWTH_MAX_FACE_PENETRATION_VOXELS 2.0f

#define MAX_CANDIDATES 16

// Keep the complete spatial map, but allocate attributes only for occupied cells.
// Each pool page stays below Falcor's 4 GiB buffer limit, including SH counts up to 16.
#ifndef SPARSE_POOL_PAGE_SIZE
#define SPARSE_POOL_PAGE_SIZE (1u << 18)
#endif
#ifndef SPARSE_POOL_MAX_PAGES
#define SPARSE_POOL_MAX_PAGES 256
#endif
#define SPARSE_INDEX_PAGE_EDGE 512
#define SPARSE_INDEX_MAX_PAGES 8

#if SH_COUNT < 1 || SH_COUNT > 16 || SH_OPACITY_COUNT < 1 || SH_OPACITY_COUNT > 16
#error "SH counts must be between 1 and 16."
#endif

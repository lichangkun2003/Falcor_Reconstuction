#pragma once

// Shared by host code and shaders. Reconstruction uses point-cloud initialization.

#ifndef GRID_RESOLUTION
#define GRID_RESOLUTION 512
#endif

// 0: fixed-resolution reconstruction;
// 1: conservative coarse-to-fine experiment.
#ifndef RECONSTRUCTION_MODE
#define RECONSTRUCTION_MODE 0
#endif
#define COARSE_TO_FINE_START_RESOLUTION 16
#define COARSEST_GROWTH_FACE_PENETRATION 1.3f
#define FINER_GROWTH_THRESHOLD_MULTIPLIER 2.0f
#define COARSE_TO_FINE_TOTAL_ITERATIONS 1000
// Four active levels use these relative weights (coarse -> fine).
// Other level counts interpolate the same curve across the active schedule.
#define COARSE_TO_FINE_LEVEL_WEIGHTS {19u, 12u, 14u, 25u}

#define REFERENCE_IMAGES_COUNT 100

// Exact per-window camera sets (foreground/background), not hashes or last-view stamps.
// Raise this and rebuild for datasets with more cameras. 128 covers the 100-view dataset.
#ifndef TOPOLOGY_EVIDENCE_MAX_VIEWS
#define TOPOLOGY_EVIDENCE_MAX_VIEWS 128
#endif
#define TOPOLOGY_EVIDENCE_VIEW_WORDS ((TOPOLOGY_EVIDENCE_MAX_VIEWS + 31) / 32)
#if TOPOLOGY_EVIDENCE_MAX_VIEWS < 1 || TOPOLOGY_EVIDENCE_MAX_VIEWS > 32767
#error "Topology camera capacity must fit the 15-bit evidence count."
#endif



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
#define GROWTH_MAX_FACE_PENETRATION_VOXELS 4.0f

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

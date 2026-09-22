#pragma once

// Shared by the host code and shaders. Select one reconstruction experiment here.
#define RECON_MODE_POINT_CLOUD 1
#define RECON_MODE_COARSE_TO_FINE 2
#define RECON_MODE_ORIGINAL 3

#ifndef RECON_MODE
#define RECON_MODE RECON_MODE_POINT_CLOUD
#endif

#if RECON_MODE != RECON_MODE_POINT_CLOUD && RECON_MODE != RECON_MODE_ORIGINAL && RECON_MODE != RECON_MODE_COARSE_TO_FINE
#error "Invalid RECON_MODE. Use one of the RECON_MODE_* values."
#endif

#ifndef GRID_RESOLUTION
#define GRID_RESOLUTION 128
#endif

// All modes share GRID_RESOLUTION as their final spatial resolution.
#define CTF_START_RESOLUTION 32
#define CTF_TOTAL_ITERATIONS 200
// Coarse stages use up to this many iterations; the final stage receives the remaining budget.
#define CTF_REFINE_INTERVAL 40
#define CTF_PRUNE_INTERVAL 10
#define CTF_EVALUATION_SPP 8

#if RECON_MODE == RECON_MODE_COARSE_TO_FINE
#if CTF_START_RESOLUTION < 4 || (CTF_START_RESOLUTION & (CTF_START_RESOLUTION - 1)) != 0
#error "CTF_START_RESOLUTION must be a power of two, at least 4."
#endif
#if GRID_RESOLUTION < CTF_START_RESOLUTION || (GRID_RESOLUTION & (GRID_RESOLUTION - 1)) != 0
#error "Coarse-to-fine requires GRID_RESOLUTION to be a power of two, at least CTF_START_RESOLUTION."
#endif
#if CTF_REFINE_INTERVAL < 1 || CTF_TOTAL_ITERATIONS < 1 || CTF_PRUNE_INTERVAL < 1 || CTF_EVALUATION_SPP < 1
#error "Coarse-to-fine iteration budgets, pruning interval and evaluation SPP must be positive."
#endif
#endif

#define REFERENCE_IMAGES_COUNT 100



// All modes use the same path-record layout and at most 8 contributing voxels per ray.
#define MAX_CONTRIBUTING_VOXELS_PER_RAY 8


#define LOBE_COUNT 4

// Radiance
#define SH_COUNT 9


// Opacity
#define SH_OPACITY_COUNT 9

#define MAX_CANDIDATES 8

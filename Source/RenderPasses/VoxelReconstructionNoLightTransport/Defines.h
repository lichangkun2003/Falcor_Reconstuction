#pragma once

// Shared by the host code and shaders. Select one reconstruction experiment here.
#define RECON_MODE_POINT_CLOUD 1
// 2 was RECON_MODE_COARSE_TO_FINE, removed. Do not reuse the value 2.
#define RECON_MODE_ORIGINAL 3

#ifndef RECON_MODE
#define RECON_MODE RECON_MODE_POINT_CLOUD
#endif

#if RECON_MODE != RECON_MODE_POINT_CLOUD && RECON_MODE != RECON_MODE_ORIGINAL
#error "Invalid RECON_MODE. Use one of the RECON_MODE_* values."
#endif

#ifndef GRID_RESOLUTION
#define GRID_RESOLUTION 128
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

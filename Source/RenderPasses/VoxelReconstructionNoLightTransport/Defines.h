#pragma once

// Shared by the host code and shaders. Select one reconstruction experiment here.
#define RECON_MODE_POINT_CLOUD 1
#define RECON_MODE_COARSE_TO_FINE 2
#define RECON_MODE_ORIGINAL 3

#ifndef RECON_MODE
#define RECON_MODE RECON_MODE_ORIGINAL
#endif

// Reserve the other modes without silently falling back to the original experiment.
#if RECON_MODE == RECON_MODE_POINT_CLOUD
#error "Point-cloud initialization is not implemented yet. Select RECON_MODE_ORIGINAL."
#elif RECON_MODE == RECON_MODE_COARSE_TO_FINE
#error "Coarse-to-fine reconstruction is not implemented yet. Select RECON_MODE_ORIGINAL."
#elif RECON_MODE != RECON_MODE_ORIGINAL
#error "Invalid RECON_MODE. Use one of the RECON_MODE_* values."
#endif

#define GRID_RESOLUTION 128
#define BLOCK_TO_VOXEL 8

#define REFERENCE_IMAGES_COUNT 100



#define MAX_CONTRIBUTING_VOXELS_PER_RAY 8


#define LOBE_COUNT 4

// Radiance
#define SH_COUNT 9


// Opacity
#define SH_OPACITY_COUNT 9

#define MAX_CANDIDATES 8

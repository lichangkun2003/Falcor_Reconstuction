# VoxelReconstructionNoLightTransport

This pass initializes a voxel reconstruction from a point cloud and optimizes its occupied voxels. The old full-grid `RECON_MODE_ORIGINAL` implementation and its initialization shader have been removed. There is no mode-selection macro; `GRID_RESOLUTION` in `Defines.h` sets the grid resolution. Rebuild the pass after changing it.

## Inputs and initialization

Set `ReferenceImageDir` and `ReferenceCameraFile` in `VoxelReconstructionNoLightTransport.h` to the same scene. Relative paths resolve from the Falcor project root. The point cloud is read from `ReferenceImageDir/point_cloud.ply`; training images and camera poses come from the reference data directory and JSON. `Reconstruction_Input` and `Reconstruction_Output` are ignored by Git.

The PLY loader accepts ASCII and binary little-endian files. A 3DGS PLY uses its Gaussian opacity, scale, and rotation to mark every intersected grid cell occupied, subject to the **Gaussian Opacity Threshold**. A plain XYZ point cloud marks the cell containing each point. Point-cloud color and SH coefficients do not initialize voxel radiance. Each occupied voxel starts with black radiance, `opacity.init()` (alpha 0.5), and a centered ellipsoid with world-space semi-axes `0.6 * voxelSize`. The coordinate transform for the NeRF data is `(x, z, -y)`.

Click **Init / Reset from PLY** to initialize without training, or enable reconstruction to initialize and train. Reinitialization clears the previous optimizer and loss state. Training renders at 800×800; when training is stopped, **Viewing Resolution** selects 800×800 or 1920×1080.

## Storage and limits

The full spatial lookup is stored in paged `R32Int` 3D textures. Only occupied cells receive a compact voxel ID and entries in segmented buffers for voxel data, gradients, Adam state, and cell indices. Radiance Adam state is allocated separately as needed. `512^3` uses one spatial-index page; `1024^3` uses eight `512^3` pages. Attribute pages contain `SPARSE_POOL_PAGE_SIZE` entries, and the pool is limited by `SPARSE_POOL_MAX_PAGES`. The UI reports active voxels, allocated capacity, and estimated pool/index/radiance-Adam memory. The estimate excludes render targets, path records, reference images, and scene resources.

The initialization error reports when the Gaussian coverage exceeds the pool limit or available GPU memory. Reducing the opacity threshold increases coverage and memory use. Pool pages are allocated according to the occupied count; future topology growth may request additional pages. The `Topology` UI and `TopologyDebug` draw mode are preview scaffolding: growth, split, and deletion do not yet modify the reconstruction.

`MAX_CONTRIBUTING_VOXELS_PER_RAY` and `MAX_CANDIDATES` in `Defines.h` limit the stored hard-hit and near-miss records. They do not change the grid resolution. A compile-time assertion checks the D3D12 structured-buffer stride limit.

## Optimization

Forward rendering uses hard ellipsoid intersections within each voxel. Geometry backward uses a continuous world-distance proxy for the hit/miss boundary, with **Geometry Tau (voxels)** converted to world units using the current voxel width. RGB and alpha losses can contribute to geometry gradients; **Alpha Geometry Weight** controls the extra alpha-to-geometry term, while **Alpha Loss Weight** controls the alpha image loss and its derivative. Center, log-scale, rotation, radiance, and opacity use separate Adam state and UI learning rates. Opacity has a warm-up and ramp. Training updates after each view's SPP batch, then advances to the next view. **Max Iteration** counts complete passes over the reference views.

## Save and load

Results are saved under `Reconstruction_Output/mode1` to preserve the existing output location. The bin filename begins with the last directory component of `ReferenceImageDir`; **Name Tag** can distinguish experiments. Automatic saving occurs at the iteration limit, and the UI can save the current reconstruction. Loss CSV files are saved separately under the output directory.

The current bin format is sparse **v3**: grid metadata followed by `(cellIndex, VoxelData)` for each active voxel. The **Reconstruction IO** panel lists files in `mode1`; **Load Selected Reconstruction** loads a v3 file for viewing and stops training. A loaded reconstruction does not require the PLY until it is explicitly reinitialized. Adam moments are not stored in the bin, so subsequent training starts with fresh optimizer state. Legacy dense v2 and earlier bins are no longer loaded by this pass.

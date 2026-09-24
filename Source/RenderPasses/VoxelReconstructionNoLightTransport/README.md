# Reconstruction experiments

在 `Defines.h` 中选择实验，修改后重新编译此 Pass。以下为配置示例，实际运行值以源码和 UI 为准：

```cpp
#define RECON_MODE RECON_MODE_POINT_CLOUD   // 1；原始模式为 RECON_MODE_ORIGINAL（3）。2 已废弃，不要复用
#define GRID_RESOLUTION 128                // mode1/mode3 共用的分辨率
```

实际选择以 `Defines.h` 为准。切换宏后需要重新编译 Pass；两个模式的新实验仍共用 `GRID_RESOLUTION`。

mode1/3 的迭代上限由头文件中的 `OptimizerParams::maxIteration` 初始化（当前默认 200），运行时可在 **Max Iteration** 中修改；达到上限后停止训练并自动保存。一次 iteration 指全部训练视角各完成一次参数更新，视角数量取自参考相机 JSON。

此 Pass 的 `CMakeLists.txt` 为 C++ 文件显式声明了本地头文件和 CPU/GPU 共享文件的依赖。修改 `ReferenceImageDir`、`ReferenceCameraFile`、`maxIteration` 或实验宏后，保存文件并正常 Build（F7），编译成功后重新启动 Mogwai，即可使用新值，无需每次清理重建。首次应用这份 CMake 修改时需先完成一次 CMake Configure。

这些显式依赖用于补足当前环境中 Ninja 未正确解析 MSVC 头文件依赖的问题，范围仅限此 Pass。新增参与 C++ 编译的本地共享文件时，需同步维护 `voxel_reconstruction_shared_headers`；GPU 专用 shader 不在此列表中。

## 数据目录与换机

头文件中的三个路径默认相对于 Falcor 项目根目录：

```cpp
inline std::string ReconstructionDataDir = "Reconstruction_Output";
inline std::string ReferenceImageDir = "Reconstruction_Input/hotdog";
inline std::string ReferenceCameraFile = "Reconstruction_Input/hotdog/transforms_train.json";
```

图片、相机 JSON、PLY 和输出目录均通过 `resolveReconstructionPath()` 解析。根目录来自 Falcor 的 `getProjectDirectory()`，由 CMake 在构建时确定，不依赖启动 Mogwai 时的工作目录。也可配置绝对路径；切换场景时同步修改图片目录和相机文件。

将 `Reconstruction_Input` 和 `Reconstruction_Output` 放在与根 `CMakeLists.txt` 同级的位置。两个目录均已加入 `.gitignore`，不会随 Git 提交；换机时自行复制所需数据和结果，并在新机器的项目目录重新 CMake Configure、编译后运行。

## Mode1：点云初始化

读取 `ReferenceImageDir/init_points.ply`。例如 `ReferenceImageDir` 为 `Reconstruction_Input/hotdog` 时，读取项目根目录下该场景的 `init_points.ply`。更换数据集时，头文件中的 `ReferenceImageDir` 和 `ReferenceCameraFile` 应指向同一场景。

- 支持 ASCII 和 binary little-endian PLY，按属性名读取 XYZ，忽略 RGB、法线等额外属性。点云仅决定占据状态，不使用点颜色。点云应使用原始 NeRF 世界坐标，加载时与相机一致转换为 `(x, z, -y)`，不重新缩放或拟合点云 AABB。
- 沿用当前固定世界 AABB 和 `GRID_RESOLUTION` 的密集网格。通过 `floor((position-gridMin)/voxelSize)` 确定体素；越界和非有限坐标分别统计并跳过，同格点合并，只有包含点的格子被占据。缺文件、无有效点或格式错误时明确报错并停止训练，保留已有体素。
- 占据体素的局部椭球中心为 `(0.5, 0.5, 0.5)`，半径为 `0.6`，`B=I/0.36`。这些初值不依赖学习率；学习率为零表示不优化该参数。
- 占据体素调用 `radiance.init()`，所有 radiance SH 系数（含 DC）为零，初始颜色全黑。opacity 直接调用与 mode3 相同的 `opacity.init()`，所有 opacity SH 系数为零，对应初始 alpha=`sigmoid(0)=0.5`。mode1 不额外写入 mode3 在 opacity 学习率为零时保留的 `16.29` DC 系数。

等待参考图片加载完，点击 **Init / Reset from PLY** 可先查看初始化结果，再勾选 **Enable Reconstruction** 开始训练。也可以直接勾选 **Enable Reconstruction**，首次会自动初始化。重置按钮会重新读取 PLY、清除迭代和 loss 状态并停止训练。训练建议保持默认 Spp=8：loss 使用前缀平均，N ≥ 2 才有意义。

结果及 loss 保存到 `Reconstruction_Output/mode1`。当前使用 v1 体素文件格式，包含全部体素的占据信息及参数；**Load Selected Reconstruction** 成功后停止训练，重新开始时直接使用加载结果，不要求 PLY 存在。加载采用文件中的实际分辨率，并按当前版本的固定重建范围 `[-1.326, 1.326]³` 还原网格。更早使用其他 AABB 或非立方网格的旧文件不在此次兼容范围内。

当前严格采用“有点才占据”：空体素不会在训练中自动生长，点云的孔洞可能保留。此模式提供点云初始化先验，没有增加邻域膨胀或空间平滑正则项。pruning 和梯度更新沿用固定分辨率流程。

## 路径记录

路径命中容量统一为 8（`MAX_CONTRIBUTING_VOXELS_PER_RAY` 与 `MAX_CANDIDATES`），保留单个路径记录 buffer，CPU 与 shader 使用同一组宏。`PathRecord.slang` 中有编译期检查，保证单个记录不超过 D3D12 的 2048 字节结构化 buffer 元素上限。mode1 和 mode3 都使用固定分辨率，初始化、更新和命中容量路径相同。

## 保存结果

输出根目录由头文件中的 `ReconstructionDataDir` 指定，默认是项目根目录下的 `Reconstruction_Output`，按 `mode1`、`mode3` 分开使用。

mode1 和 mode3 的新 bin 文件名都以 `ReferenceImageDir` 的末级目录名作为场景前缀，路径末尾带分隔符也可识别。例如目录 `Reconstruction_Input/hotdog` 对应 `hotdog_recon9_22_128_radiance_opacity_center_B.bin`。自定义 **Name Tag** 仍保留。已有文件无需重命名，仍可正常 Load。

```text
Reconstruction_Output/
  mode1/
    hotdog_recon*.bin
    Loss/
  mode3/
    hotdog_recon*.bin
    Loss/
```

mode1 和 mode3 同名 bin 会被覆盖；需要保留多次实验时使用不同的 **Name Tag**。

mode1 和 mode3 都使用 v1 体素文件格式，包含全部体素的占据信息及参数。场景名前缀只改变文件名，不改变 bin 格式。loss CSV 使用日期和 Name Tag 命名，同名会覆盖，不自动添加场景名。

## 加载当前版本的结果

在 **Reconstruction IO** 中，点击 **Refresh Files**，选择文件并点击 **Load Selected Reconstruction**。列表递归扫描当前 `RECON_MODE` 对应的目录，显示相对路径；不会混入其他 mode 的文件。v1 本身没有 mode 字段，所以 mode1、mode3 的归属由文件夹区分。

两个 mode 的 Load 均按文件实际网格尺寸重建资源，暂停优化、取消待执行的初始化，并清除旧采样累积。查看结果只需要已加载的场景和场景相机，不要求 PLY、训练图片或参考相机文件存在。`color` 和 `AccuColor` 都显示加载结果，暂停时不会因训练 Spp 大于 1 而将画面亮度除以 Spp。UI 会显示加载成功或具体失败原因；格式、长度或资源分配失败时保留原结果。

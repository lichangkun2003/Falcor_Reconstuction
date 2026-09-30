# Reconstruction experiments

## Mode1 compact storage (current implementation)

`RECON_MODE_POINT_CLOUD` now separates the complete spatial lookup from voxel attributes. The spatial lookup stores a compact voxel ID for every grid cell in paged `R32Int` 3D textures; only occupied voxels allocate `VoxelData`, `GradRecord`, geometry Adam state, and cell-index entries in segmented structured-buffer pools. Ray marching resolves `cell -> voxelID`, and path records, gradient accumulation, and updates use that compact ID directly.

- `512^3` uses one spatial-index page. `1024^3` uses eight `512^3` pages, so no individual resource exceeds Falcor's 4 GiB buffer limit. This only removes the per-resource size limit: a scene whose Gaussian coverage creates more than `SPARSE_POOL_PAGE_SIZE * SPARSE_POOL_MAX_PAGES` occupied cells still cannot initialize, and the combined index, voxel, and gradient allocations must fit GPU memory. The initialization error reports the occupied count and the minimum pool memory needed.
- Attribute pools grow in pages of `SPARSE_POOL_PAGE_SIZE`; initialization and loading reserve 25% extra capacity (at least 1024 entries) for later growth/dilation work. The current hard pool limit is `SPARSE_POOL_PAGE_SIZE * SPARSE_POOL_MAX_PAGES` active voxels.
- Mode1 saves sparse reconstruction format v3: grid metadata followed by `(cellIndex, VoxelData)` for active voxels only. Mode1 can still load the previous dense v2 files and converts their occupied entries into the compact pool. Mode3 keeps its existing dense v2 format.
- Growth, dilation, deletion compaction, and free-list allocation are not enabled yet. The paged capacity and cell-index reverse map are the storage foundation for those operations.

The UI reports active voxel count, reserved capacity, pool page count, and spatial-index page count. Change `GRID_RESOLUTION`, rebuild the pass, and reinitialize from PLY to start a new resolution.

在 `Defines.h` 中选择实验，修改后重新编译此 Pass。以下为配置示例，实际运行值以源码和 UI 为准：

```cpp
#define RECON_MODE RECON_MODE_POINT_CLOUD   // 1；原始模式为 RECON_MODE_ORIGINAL（3）。2 已废弃，不要复用
#define GRID_RESOLUTION 512                // mode1/mode3 共用的分辨率
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

读取 `ReferenceImageDir/point_cloud.ply`，即该场景 3DGS 优化输出的稠密点云。例如 `ReferenceImageDir` 为 `Reconstruction_Input/ship` 时，读取项目根目录下该场景的 `point_cloud.ply`。更换数据集时，头文件中的 `ReferenceImageDir` 和 `ReferenceCameraFile` 应指向同一场景，并把该场景的 3DGS 结果放到同目录下。

- 支持 ASCII 和 binary little-endian PLY，按属性名读取 XYZ 和 3DGS 的每高斯属性（`opacity`、`scale_0..2`、`rot_0..3`），忽略 RGB、法线、`f_dc_*`、`f_rest_*` 等额外属性。点云仅决定占据状态，不使用点颜色和 SH。点云应使用原始 NeRF 世界坐标，加载时与相机一致转换为 `(x, z, -y)`，不重新缩放或拟合点云 AABB。
- 沿用当前固定世界 AABB 和 `GRID_RESOLUTION` 的密集网格。取 `α = sigmoid(opacity)`、`σ = exp(scale_*)`、`Σ = R diag(σ²) Rᵀ`，将 `qᵀΣ⁻¹q ≤ 2·ln(α/ε)` 定义的阈值椭球与整个体素 AABB 做相交测试：只要体素中任意一点达到阈值就占据，包括仅碰到边界的格子，不再要求覆盖格心。这避免薄盆壁、细线高斯穿过体素却漏格。`α < ε` 的高斯整个丢弃。完整高斯 AABB 只裁剪到场景网格，不再截断每轴覆盖半宽；以 `8³` 体素块遍历，跳过已经完全占据的块，用主轴空间区间界递归排除空区域，边界格通过面、边和角点的最小二次型判定精确相交。越界、非有限、被丢弃、测试块和跳过块的数量写入统计；`cells added` 计新占据格数，重复覆盖不累加。`ε` 就是 UI 上的 **Gaussian Opacity Threshold**（`PointCloudState::opacityThreshold`，默认 `0.05`），改动在下次 **Init / Reset from PLY** 时生效；`ε` 必须落在 `(0, 1)` 内，否则初始化直接报错。降低阈值会增加占据数量和初始化耗时。
- PLY 里没有每高斯属性时（普通点云，例如 COLMAP 输出）退化为旧行为：只占据包含该点的格子。缺文件、无有效点或格式错误时明确报错并停止训练，保留已有体素。
- mode1 占据体素的局部椭球中心为 `(0.5, 0.5, 0.5)`，各轴 `σ = 0.6 × voxelSize`（`logScale = log(0.6 × voxelSize)`），rotation 为单位四元数。该半轴不会覆盖体素方格的所有角落；PLY 占据覆盖与椭球硬 hit 覆盖是两件事。这些初值不依赖学习率；学习率为零表示不优化该参数。mode3 保留自己的原始初始化。
- 占据体素调用 `radiance.init()`，所有 radiance SH 系数（含 DC）为零，初始颜色全黑。opacity 直接调用与 mode3 相同的 `opacity.init()`，所有 opacity SH 系数为零，对应初始 alpha=`sigmoid(0)=0.5`。mode1 不额外写入 mode3 在 opacity 学习率为零时保留的 `16.29` DC 系数。

等待参考图片加载完，点击 **Init / Reset from PLY** 可先查看初始化结果，再勾选 **Enable Reconstruction** 开始训练。也可以直接勾选 **Enable Reconstruction**，首次会自动初始化。重置按钮会重新读取 PLY、清除迭代和 loss 状态并停止训练。训练建议保持默认 Spp=8。整批 Spp 帧里，每一帧都用“不含自己”的前缀平均残差算出自己那一份无偏梯度，原子累加到梯度缓冲；整批结束后统一更新一次，按各体素的累积计数求平均，也就是整批的平均梯度。第 0 帧没有前缀可用，只作为后续采样的基线，所以 N ≥ 2 才有意义，Spp=1 退化成单样本的旧行为。

结果及 loss 保存到 `Reconstruction_Output/mode1`。新保存结果使用稀疏 v3 格式，形状参数仍为中心、世界空间对数半轴和单位四元数；**Load Selected Reconstruction** 成功后停止训练，重新开始时直接使用加载结果，不要求 PLY 存在。v3 直接保存网格 AABB、体素尺寸和非空体素条目；旧的 dense v2 mode1 文件仍可加载并转换到紧密池。旧的 v1 矩阵椭球 bin 不会按新布局误读，需要重新初始化或重新烘焙。

占据状态只在初始化时确定，空体素不会在训练中自动生长。改用 3DGS 的覆盖式初始化后，只要高斯覆盖到位，原始点云的大孔洞就会被填上；代价是不再是严格的“有点才占据”，多占的体素目前只能靠背景射线剔除压掉（ellipsoid pruning 默认关闭）。pruning 和梯度更新沿用固定分辨率流程。

## 几何代理的距离尺度

前向仍使用硬椭球相交。首先将 ray 裁到当前体素内部的 `[localFrom, localTo]`，再在线段参数 `t ∈ [0,1]` 上求 `gmin` 和 `xStar`，而不是在整条无限 ray 上求最小值；线段端点也可能是最小值位置。该线段与原椭球的相交等价于与体素截断椭球的相交。几何反向的 soft-hit 权重为 `w = sigmoid(-dWorld / tauWorld)`，其中 `dWorld = gmin / surfaceGradNorm` 是这个相交判定附近的一阶世界空间距离代理，并不是 ray 到截断椭球的精确欧氏距离。`surfaceGradNorm` 在沿主轴归一化方向投影到原椭球表面的点计算，避免在椭球中心直接除以零梯度。中心附近方向不确定的射线跳过几何梯度，外观梯度仍正常计算。

反向将 `1 / surfaceGradNorm` 视为常数（stop-gradient），乘到原有 center、logScale 和 rotation 的隐式函数梯度上。near-miss 候选排序使用同一个距离近似。`Geometry Grad Clamp` 仍限制转换后的 `dL/dg`；按计数平均和体素局部中心坐标保持原有定义。mode1 的 center、logScale 和局部切空间 rotation 均使用各自独立的 Adam 状态；mode3 保留原有 SGD。alpha loss 继续更新 opacity，同时通过下述几何代理传给 center、logScale 和 rotation。

**Geometry Tau (voxels)** 是唯一可编辑的 tau 参数，表示当前体素最大边长 `h` 的倍数 `k`，当前默认 `k=0.12`。每次计算梯度时按当前网格重新计算 `tauWorld = k·h`，hit 与 miss 使用相同数值。**Geometry Tau (world)** 为派生值的只读显示。固定 AABB 边长为 `2.652` 时，128 分辨率默认约 `0.00248625`，256 分辨率默认约 `0.001243125`。加载不同分辨率的 bin 后保留 `k`，世界 tau 自动跟随新体素大小，避免继续使用旧分辨率的 band。这里不使用像素大小、屏幕分辨率、相机距离或单个椭球半轴来设置 tau；它只控制当前体素内 ray 片段的几何代理过渡。旧的无量纲 `gmin` tau（如 `0.15`、`0.32`）不等于这里的体素倍数。

## Alpha loss 的几何代理

Loss Pass 计算 `L = L_RGB + AlphaLossWeight·(A_render - A_ref)²`，UI 的 **Alpha Loss Weight** 默认 `0.3`，范围 `0～10`，同时控制 alpha loss 和 `dL/dA`。几何分支不再乘一次该 loss 权重。设为 `0` 会关闭 alpha loss 及其传给 opacity 和几何的梯度，RGB 与背景 carve 保持原有逻辑。调整该值会重新开始当前视角的 SPP 批次，避免累积不同权重的梯度；重建 GPU 资源时保留所选权重。前向保持硬 hit，alpha 不乘 soft-hit 权重；新增项只用于反向的局部代理。

将一个实际 hit 看成由 gate 控制的 opacity，或在 near-miss 的位置插入一个虚拟 gate，记录路径的 alpha 为 `A = 1 - Π(1 - opacity_i·gate_i)`，所以 `dA/dgate_i = T_before·opacity_i·T_after`。新增距离梯度为：

```text
dL_alpha_geom/ddWorld = AlphaGeometryWeight · (dL/dA)
                     · T_before · opacity · T_after
                     · [-w·(1-w) / tauWorld]
```

与现有 RGB/背景 carve 的距离梯度相加后，一起乘 `distanceScale`、应用同一个 `Geometry Grad Clamp`，再传到 center、logScale 和旋转切向量。hard-hit 在反向遍历中使用不含自己的后方透射率；near-miss 根据 `frontHitCount`，只乘候选后方已记录 hard-hit 的 `(1-opacity)`。前方或后方完全不透明时，该 alpha 项为零，避免已被遮挡的椭球仍被 alpha 推动膨胀。near-miss 仍只处理参考前景，背景 hard-hit 保留原 carve 项并增加 alpha 修正。

UI 的 **Alpha Geometry Weight** 是额外的几何倍率，默认 `0.1`，不是 Loss Pass 的 `alphaLossWeight`，也不会改变 opacity 学习率或外观梯度。tau 与该倍率的默认值直接存储在 `GradientPass` 的成员声明中，`init()` 不再重复赋值覆盖配置。设为 `0` 可关闭新增的 alpha 几何项，在相同 tau、学习率、初始化等配置下得到接入前的几何更新。alpha 与 RGB 几何共用现有体素距离和 tau，没有独立的 Alpha Geometry Band。GradRecord、PathRecord 和 bin 格式保持原样，新增项与原梯度合并后每个记录只累计一次几何计数。

这是硬命中前向的代理梯度，不能宣称为硬 hit 的精确导数或真实像素覆盖面积的导数。数值检查应冻结距离归一化，验证局部 gate 代理的导数；完整场景效果需要实验。建议从 `0.1` 开始，与 `0` 对照，必要时测试 `0.05 / 0.2`，并为每组重新初始化、使用不同 Name Tag 保存。

## 几何学习率对照实验

学习率直接存储在 `UpdatePass::init()`，UI 编辑并传递实际数值，不再通过 `scale × 0.001` 派生，也不依赖 Update Pass 面板是否展开。默认如下：

| 参数 | 默认值 | 更新变量 |
| --- | ---: | --- |
| Adam LR center (voxel local) | `0.001` | mode1 的体素局部中心；mode3 仍为 SGD，默认 `0.005` |
| Adam LR log scale | `0.001` | mode1 的世界半轴自然对数；mode3 形状与旋转共用 SGD LR `0.1` |
| Adam LR rotation | `0.0005` | mode1 的局部旋转切向量；单步仍限制为 `0.02 rad` |
| LR opacity | `10` | opacity 的 logit SH 系数，保留原默认值 |
| Adam LR radiance | `0.001` | mode1 的每个 RGB SH 系数；mode3 仍为 SGD，默认 `0.1` |
| Geometry Tau (voxels) | `0.12` | 世界距离代理的 sigmoid 过渡宽度除以当前体素边长 |
| Alpha Geometry Weight | `0.1` | 已有 alpha loss 传给几何的额外倍率，`0` 关闭新增项 |
| Alpha Loss Weight | `0.3` | alpha 图像损失及其梯度的权重，范围 `0～10` |

这些是 mode1 Adam 的实验起点，不能沿用旧 SGD 的 shape LR=`0.1` 或 radiance LR=`0.1`：Adam 会按每个参数的二阶矩归一化梯度。每个视角的 SPP 梯度累积结束后更新一次。中心、logScale、三维局部切空间旋转和 radiance 各自维护一阶矩、二阶矩及实际收到对应梯度的步数；radiance 的所有 RGB SH 系数共用一个时间步，但每个系数的 moments 独立。没有对应梯度时不更新。旋转 Adam 的输出仍限制为单步最多 `0.02 rad`，再转换成增量四元数并归一化；radiance 单系数、单通道的更新仍限制为 `0.05`。`beta1=0.9`、`beta2=0.999`、`epsilon=1e-8`，opacity 仍用原 SGD。`SH_COUNT=9` 时 mode1 每个池槽占 304 字节 Adam 状态；重建初始化与加载 bin 会清零状态，当前 v3 bin 只保存体素参数，因此加载后训练是从新 Adam 状态开始，不是精确续训。

tau 同时影响作用范围和幅度。对中心位于体素中央、初始半径 `r=0.6h` 的球，体素内任意点满足 `|q| ≤ √3·h/2`，当前代理 `dWorld = (|q|²/r² - 1)·r/2` 的范围约为 `[-0.3h, 0.325h]`（中心点除外）。`tauWorld=0.15h` 时 sigmoid 的 10%–90% 过渡总宽度约 `4.394·tau = 0.659h`，该初始范围的 `w·(1-w)` 至少约为边界峰值的 37%，减少体素内有梯度的样本因为 band 过窄而饱和。这个范围估计只用于当前初始球，不适用于优化后的任意椭球。距离代理远离边界可能高估真实距离，不能将它宣称为始终受体素直径约束的精确距离。

先固定 **Geometry Tau (voxels)**=`0.12`，分别比较 mode1 的 Adam center 与 logScale LR；再固定合适的学习率比较 tau 倍数 `0.1 / 0.12 / 0.15`。每组重新初始化，保持相同 Alpha Geometry Weight、opacity LR=`10`、视角、Spp 和迭代次数，使用不同 Name Tag 保存。既比较 RGB/alpha loss，也比较椭球半轴和轮廓，不能只凭 loss 降低断定几何变好。tau 越小，边界附近梯度峰值越大、远离边界的 RGB/alpha 代理梯度衰减越快；tau 不会随单个椭球收缩而缩窄。保留原单次更新限幅：中心每轴 `0.02` 体素、logScale 每轴 `0.02`（半轴相对变化约 2%）、旋转总角度 `0.02` 弧度。频繁触发限幅时继续增大学习率不会按比例增加步幅。

三轴完全相等的初始球对旋转不敏感，此时旋转梯度为零；先出现轴长差异后才能学习朝向。梯度为零、射线未被记录、没有颜色误差信号等情况，不能单靠提高学习率解决。

## 路径记录

路径记录当前最多保留 16 个实际 hard hit（`MAX_CONTRIBUTING_VOXELS_PER_RAY=16`）和 16 个 near-miss candidate（`MAX_CANDIDATES=16`）。编译期布局检查会根据两个宏计算记录大小，并检查 D3D12 的 2048 字节 stride 上限。当前 mode1 的 `PathRecord` 为 1264 字节，mode3 为 1520 字节；800×800 输出下，mode1 路径 buffer 约为 0.75 GiB。前向仍会在透射率低于阈值时提前停止，两个数只是记录上限。

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

mode1 新文件使用稀疏 v3 格式，只保存活动体素及其空间格子编号；mode3 继续使用包含全部体素的 dense v2 格式。场景名前缀只改变文件名，不改变对应 mode 的 bin 格式。loss CSV 使用日期和 Name Tag 命名，同名会覆盖，不自动添加场景名。

## 加载当前版本的结果

在 **Reconstruction IO** 中，点击 **Refresh Files**，选择文件并点击 **Load Selected Reconstruction**。列表递归扫描当前 `RECON_MODE` 对应的目录，显示相对路径；不会混入其他 mode 的文件。v2/v3 头均不承担跨 mode 选择，mode1、mode3 的归属仍由文件夹区分。

两个 mode 的 Load 均按文件实际网格尺寸重建资源，暂停优化、取消待执行的初始化，并清除旧采样累积。查看结果只需要已加载的场景和场景相机，不要求 PLY、训练图片或参考相机文件存在。`color` 和 `AccuColor` 都显示加载结果，暂停时不会因训练 Spp 大于 1 而将画面亮度除以 Spp。UI 会显示加载成功或具体失败原因；格式、长度或资源分配失败时保留原结果。

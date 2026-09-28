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

读取 `ReferenceImageDir/point_cloud.ply`，即该场景 3DGS 优化输出的稠密点云。例如 `ReferenceImageDir` 为 `Reconstruction_Input/ship` 时，读取项目根目录下该场景的 `point_cloud.ply`。更换数据集时，头文件中的 `ReferenceImageDir` 和 `ReferenceCameraFile` 应指向同一场景，并把该场景的 3DGS 结果放到同目录下。

- 支持 ASCII 和 binary little-endian PLY，按属性名读取 XYZ 和 3DGS 的每高斯属性（`opacity`、`scale_0..2`、`rot_0..3`），忽略 RGB、法线、`f_dc_*`、`f_rest_*` 等额外属性。点云仅决定占据状态，不使用点颜色和 SH。点云应使用原始 NeRF 世界坐标，加载时与相机一致转换为 `(x, z, -y)`，不重新缩放或拟合点云 AABB。
- 沿用当前固定世界 AABB 和 `GRID_RESOLUTION` 的密集网格。取 `α = sigmoid(opacity)`、`σ = exp(scale_*)`、`Σ = R diag(σ²) Rᵀ`，将 `qᵀΣ⁻¹q ≤ 2·ln(α/ε)` 定义的阈值椭球与整个体素 AABB 做相交测试：只要体素中任意一点达到阈值就占据，包括仅碰到边界的格子，不再要求覆盖格心。这避免薄盆壁、细线高斯穿过体素却漏格。`α < ε` 的高斯整个丢弃。完整高斯 AABB 只裁剪到场景网格，不再截断每轴覆盖半宽；以 `8³` 体素块遍历，跳过已经完全占据的块，用主轴空间区间界递归排除空区域，边界格通过面、边和角点的最小二次型判定精确相交。越界、非有限、被丢弃、测试块和跳过块的数量写入统计；`cells added` 计新占据格数，重复覆盖不累加。`ε` 就是 UI 上的 **Gaussian Opacity Threshold**（`PointCloudState::opacityThreshold`，默认 `0.05`），改动在下次 **Init / Reset from PLY** 时生效；`ε` 必须落在 `(0, 1)` 内，否则初始化直接报错。降低阈值会增加占据数量和初始化耗时。
- PLY 里没有每高斯属性时（普通点云，例如 COLMAP 输出）退化为旧行为：只占据包含该点的格子。缺文件、无有效点或格式错误时明确报错并停止训练，保留已有体素。
- mode1 占据体素的局部椭球中心为 `(0.5, 0.5, 0.5)`，各轴 `σ = 0.6 × voxelSize`（`logScale = log(0.6 × voxelSize)`），rotation 为单位四元数。该半轴不会覆盖体素方格的所有角落；PLY 占据覆盖与椭球硬 hit 覆盖是两件事。这些初值不依赖学习率；学习率为零表示不优化该参数。mode3 保留自己的原始初始化。
- 占据体素调用 `radiance.init()`，所有 radiance SH 系数（含 DC）为零，初始颜色全黑。opacity 直接调用与 mode3 相同的 `opacity.init()`，所有 opacity SH 系数为零，对应初始 alpha=`sigmoid(0)=0.5`。mode1 不额外写入 mode3 在 opacity 学习率为零时保留的 `16.29` DC 系数。

等待参考图片加载完，点击 **Init / Reset from PLY** 可先查看初始化结果，再勾选 **Enable Reconstruction** 开始训练。也可以直接勾选 **Enable Reconstruction**，首次会自动初始化。重置按钮会重新读取 PLY、清除迭代和 loss 状态并停止训练。训练建议保持默认 Spp=8。整批 Spp 帧里，每一帧都用“不含自己”的前缀平均残差算出自己那一份无偏梯度，原子累加到梯度缓冲；整批结束后统一更新一次，按各体素的累积计数求平均，也就是整批的平均梯度。第 0 帧没有前缀可用，只作为后续采样的基线，所以 N ≥ 2 才有意义，Spp=1 退化成单样本的旧行为。

结果及 loss 保存到 `Reconstruction_Output/mode1`。当前使用 v2 体素文件格式，形状参数为中心、世界空间对数半轴和单位四元数；**Load Selected Reconstruction** 成功后停止训练，重新开始时直接使用加载结果，不要求 PLY 存在。加载采用文件中的实际分辨率，并按当前版本的固定重建范围 `[-1.326, 1.326]³` 还原网格。旧的 v1 矩阵椭球 bin 不会按新布局误读，需要重新初始化或重新烘焙。

占据状态只在初始化时确定，空体素不会在训练中自动生长。改用 3DGS 的覆盖式初始化后，只要高斯覆盖到位，原始点云的大孔洞就会被填上；代价是不再是严格的“有点才占据”，多占的体素目前只能靠背景射线剔除压掉（ellipsoid pruning 默认关闭）。pruning 和梯度更新沿用固定分辨率流程。

## 几何代理的距离尺度

前向仍使用硬椭球相交。首先将 ray 裁到当前体素内部的 `[localFrom, localTo]`，再在线段参数 `t ∈ [0,1]` 上求 `gmin` 和 `xStar`，而不是在整条无限 ray 上求最小值；线段端点也可能是最小值位置。该线段与原椭球的相交等价于与体素截断椭球的相交。几何反向的 soft-hit 权重为 `w = sigmoid(-dWorld / tauWorld)`，其中 `dWorld = gmin / surfaceGradNorm` 是这个相交判定附近的一阶世界空间距离代理，并不是 ray 到截断椭球的精确欧氏距离。`surfaceGradNorm` 在沿主轴归一化方向投影到原椭球表面的点计算，避免在椭球中心直接除以零梯度。中心附近方向不确定的射线跳过几何梯度，外观梯度仍正常计算。

反向将 `1 / surfaceGradNorm` 视为常数（stop-gradient），乘到原有 center、logScale 和 rotation 的隐式函数梯度上。near-miss 候选排序使用同一个距离近似。`Geometry Grad Clamp` 仍限制转换后的 `dL/dg`；SGD、按计数平均和体素局部中心坐标保持原有定义。alpha loss 仍只更新 opacity，没有传给几何。

**Geometry Tau (voxels)** 是唯一可编辑的 tau 参数，表示当前体素最大边长 `h` 的倍数 `k`，默认 `k=0.15`。每次计算梯度时按当前网格重新计算 `tauWorld = k·h`，hit 与 miss 使用相同数值。**Geometry Tau (world)** 为派生值的只读显示。固定 AABB 边长为 `2.652` 时，128 分辨率默认约 `0.003108`，256 分辨率默认约 `0.001554`。加载不同分辨率的 bin 后保留 `k`，世界 tau 自动跟随新体素大小，避免继续使用旧分辨率的 band。这里不使用像素大小、屏幕分辨率、相机距离或单个椭球半轴来设置 tau；它只控制当前体素内 ray 片段的几何代理过渡。旧的无量纲 `gmin` tau（如 `0.15`、`0.32`）不等于这里的体素倍数。

## 几何学习率对照实验

学习率直接存储在 `UpdatePass::init()`，UI 编辑并传递实际数值，不再通过 `scale × 0.001` 派生，也不依赖 Update Pass 面板是否展开。默认如下：

| 参数 | 默认值 | 更新变量 |
| --- | ---: | --- |
| LR center (voxel local) | `0.005` | 体素局部中心坐标，原几何默认值的 10 倍 |
| LR shape (log scale / rotation) | `0.1` | 世界半轴的自然对数及局部旋转切向量 |
| LR opacity | `10` | opacity 的 logit SH 系数，保留原默认值 |
| LR radiance | `0.1` | radiance SH 系数，保留原默认值 |
| Geometry Tau (voxels) | `0.15` | 世界距离代理的 sigmoid 过渡宽度除以当前体素边长 |

这是基于参数尺度的实验起点，尚未通过完整场景训练选出最优值。切换到新椭球表示时，shape 实际学习率从旧 Cholesky 更新的 `0.1` 改成了 `0.001`。对轴对齐球、相同 `dL/dg`、未触发限幅的一次更新，旧对数 Cholesky 对角线与新对数半轴的梯度大小相同、符号相反；旧形状步幅因此是迁移后默认值的 100 倍。现在恢复 shape 学习率量级并提高 center 步幅；opacity 保留 `10`，外观梯度和更新保持原样，只校准几何相关参数。这不表示两种形状参数化的一般更新轨迹完全相同。

tau 同时影响作用范围和幅度。对中心位于体素中央、初始半径 `r=0.6h` 的球，体素内任意点满足 `|q| ≤ √3·h/2`，当前代理 `dWorld = (|q|²/r² - 1)·r/2` 的范围约为 `[-0.3h, 0.325h]`（中心点除外）。`tauWorld=0.15h` 时 sigmoid 的 10%–90% 过渡总宽度约 `4.394·tau = 0.659h`，该初始范围的 `w·(1-w)` 至少约为边界峰值的 37%，减少体素内有梯度的样本因为 band 过窄而饱和。这个范围估计只用于当前初始球，不适用于优化后的任意椭球。距离代理远离边界可能高估真实距离，不能将它宣称为始终受体素直径约束的精确距离。

先固定 **Geometry Tau (voxels)**=`0.15`，比较 shape LR=`0.03 / 0.1 / 0.3`；再固定合适的 shape LR 比较 tau 倍数 `0.1 / 0.15 / 0.25`。每组重新初始化，保持 opacity LR=`10`、相同视角、Spp 和迭代次数，使用不同 Name Tag 保存。既比较 RGB/alpha loss，也比较椭球半轴和轮廓，不能只凭 loss 降低断定几何变好。tau 越小，边界附近梯度峰值越大、远离边界的 RGB 代理梯度衰减越快；tau 不会随单个椭球收缩而缩窄。形状仍使用按计数平均的 SGD，保留原单次更新限幅：中心每轴 `0.02` 体素、logScale 每轴 `0.02`（半轴相对变化约 2%）、旋转总角度 `0.02` 弧度。频繁触发限幅时继续增大学习率不会按比例增加步幅。

三轴完全相等的初始球对旋转不敏感，此时旋转梯度为零；先出现轴长差异后才能学习朝向。梯度为零、射线未被记录、没有颜色误差信号等情况，不能单靠提高学习率解决。

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

mode1 和 mode3 都使用 v2 体素文件格式，包含全部体素的占据信息及参数。场景名前缀只改变文件名，不改变 bin 格式。loss CSV 使用日期和 Name Tag 命名，同名会覆盖，不自动添加场景名。

## 加载当前版本的结果

在 **Reconstruction IO** 中，点击 **Refresh Files**，选择文件并点击 **Load Selected Reconstruction**。列表递归扫描当前 `RECON_MODE` 对应的目录，显示相对路径；不会混入其他 mode 的文件。v2 本身没有 mode 字段，所以 mode1、mode3 的归属由文件夹区分。

两个 mode 的 Load 均按文件实际网格尺寸重建资源，暂停优化、取消待执行的初始化，并清除旧采样累积。查看结果只需要已加载的场景和场景相机，不要求 PLY、训练图片或参考相机文件存在。`color` 和 `AccuColor` 都显示加载结果，暂停时不会因训练 Spp 大于 1 而将画面亮度除以 Spp。UI 会显示加载成功或具体失败原因；格式、长度或资源分配失败时保留原结果。

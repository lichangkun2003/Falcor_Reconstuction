# Reconstruction experiments

在 `Defines.h` 中选择实验，修改后重新编译此 Pass。以下为配置示例，实际运行值以源码和 UI 为准：

```cpp
#define RECON_MODE RECON_MODE_POINT_CLOUD   // 1；粗到细为 RECON_MODE_COARSE_TO_FINE（2），原始模式为 RECON_MODE_ORIGINAL（3）
#define GRID_RESOLUTION 128                // 所有模式共用的最终目标分辨率
```

实际选择以 `Defines.h` 为准。切换宏后需要重新编译 Pass；三个模式的新实验仍共用 `GRID_RESOLUTION`。

mode1/3 的迭代上限由头文件中的 `OptimizerParams::maxIteration` 初始化（当前默认 100），运行时可在 **Max Iteration** 中修改；达到上限后停止训练并自动保存。mode2 使用 `CTF_TOTAL_ITERATIONS` 分配各层预算（当前默认总计 200），不使用这个 `maxIteration` 上限。一次 iteration 指全部训练视角各完成一次参数更新，视角数量取自参考相机 JSON。

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

mode2 新检查点保存相对参考路径，方便跨机器恢复。已有检查点中包含完整 `Reconstruction_Input/...` 后缀的旧绝对路径也可匹配搬迁后的数据，但图像目录和相机文件必须同时对应；其他训练兼容性检查仍然生效。

## Mode1：点云初始化

读取 `ReferenceImageDir/init_points.ply`。例如 `ReferenceImageDir` 为 `Reconstruction_Input/hotdog` 时，读取项目根目录下该场景的 `init_points.ply`。更换数据集时，头文件中的 `ReferenceImageDir` 和 `ReferenceCameraFile` 应指向同一场景。

- 支持 ASCII 和 binary little-endian PLY，按属性名读取 XYZ，忽略 RGB、法线等额外属性。点云仅决定占据状态，不使用点颜色。点云应使用原始 NeRF 世界坐标，加载时与相机一致转换为 `(x, z, -y)`，不重新缩放或拟合点云 AABB。
- 沿用当前固定世界 AABB 和 `GRID_RESOLUTION` 的密集网格。通过 `floor((position-gridMin)/voxelSize)` 确定体素；越界和非有限坐标分别统计并跳过，同格点合并，只有包含点的格子被占据。缺文件、无有效点或格式错误时明确报错并停止训练，保留已有体素。
- 占据体素的局部椭球中心为 `(0.5, 0.5, 0.5)`，半径为 `0.6`，`B=I/0.36`。这些初值不依赖学习率；学习率为零表示不优化该参数。
- 占据体素调用 `radiance.init()`，所有 radiance SH 系数（含 DC）为零，初始颜色全黑。opacity 直接调用与 mode3 相同的 `opacity.init()`，所有 opacity SH 系数为零，对应初始 alpha=`sigmoid(0)=0.5`。mode1 不额外写入 mode3 在 opacity 学习率为零时保留的 `16.29` DC 系数。

等待参考图片加载完，点击 **Init / Reset from PLY** 可先查看初始化结果，再勾选 **Enable Reconstruction** 开始训练。也可以直接勾选 **Enable Reconstruction**，首次会自动初始化。重置按钮会重新读取 PLY、清除迭代和 loss 状态并停止训练。训练建议先保持默认 Spp=1。

结果及 loss 保存到 `Reconstruction_Output/mode1`。当前使用 v1 体素文件格式，包含全部体素的占据信息及参数；**Load Selected Reconstruction** 成功后停止训练，重新开始时直接使用加载结果，不要求 PLY 存在。加载采用文件中的实际分辨率，并按当前版本的固定重建范围 `[-1.326, 1.326]³` 还原网格。更早使用其他 AABB 或非立方网格的旧文件不在此次兼容范围内。

当前严格采用“有点才占据”：空体素不会在训练中自动生长，点云的孔洞可能保留。此模式提供点云初始化先验，没有增加邻域膨胀或空间平滑正则项。pruning 和梯度更新沿用固定分辨率流程。

## Mode2：分阶段优化

默认从 32 开始，每层分辨率翻倍，到 `GRID_RESOLUTION` 为止：128 对应 32 → 64 → 128，256 对应 32 → 64 → 128 → 256。起点由 `CTF_START_RESOLUTION` 指定，必须是至少 4 的 2 的幂；目标也必须是 2 的幂，且不小于起点。

`CTF_TOTAL_ITERATIONS` 默认 200，控制新实验的总迭代预算；`CTF_REFINE_INTERVAL` 默认 40，控制中间层的预算上限，最终层获得剩余预算。默认 32 → 64 → 128 分配为 **40 + 40 + 120 = 200**；目标改为 256 时分配为 **40 + 40 + 40 + 80 = 200**。如果层级较多，会自动缩短中间层预算，并在允许时对齐删除周期，保证各层至少一次迭代、默认总数仍为 200。手动追加迭代会增加总数。

视角数量读取自 `transforms_train.json`，不依赖 `REFERENCE_IMAGES_COUNT`。训练沿用现有梯度实现，建议保持默认 Spp=1。使用 **Restore Training Checkpoint** 恢复旧检查点时保留当时的本层预算，后续层按当前宏分配；要完整应用当前宏定义的总预算，使用 **Initialize / reset coarse experiment** 开始新实验。普通 **Load Selected Reconstruction** 只用于查看，不恢复训练预算。

1. 等待参考图片加载完，点击 **Start coarse-to-fine**，会自动初始化最低层。
2. **Pause after each stage** 默认不勾选。每层达到预算后自动保存、升层，连续优化到最终分辨率。
3. **Display level** 可选择 **Current optimization level (live)** 或已保存的层级，例如 32、64、128；条目同时显示保存时的本层迭代数。每层完成保存后自动加入列表，同层追加训练完成后更新为最新结果。训练中也可以查看之前的层级，最终训练完成后可依次切换比较。
4. 若需要逐层判断迭代是否足够，可勾选 **Pause after each stage**。暂停后设置 **Additional iterations**，点击 **Add iterations and continue this stage**；满意后点击 **Refine and start next stage**。
5. **Pause and save after this iteration** 和训练中的 **Save Reconstruction** 都等待当前整轮结束。前者暂停，后者保存后继续训练。
6. **Load Selected Reconstruction** 加载 mode2 文件用于查看，暂停训练，不恢复预算、学习率或 loss 历史。同一实验目录的层级结果会加入 **Display level**。需要接着优化时，选择文件并点击 **Restore Training Checkpoint**；配置兼容且恢复成功后仍保持暂停，可继续该层、追加轮次或进入下一层。

层级选择同时作用于 `VoxelReconstruction.color` 和 `VoxelReconstruction.AccuColor`。查看历史层时，可使用场景相机，或勾选 **Use ReferenceCamera** 并调整 **Camera Index**，以相同视角比较各层。切换显示不恢复训练状态，也不改变当前训练的层级、相机、迭代进度、学习率或 loss；**Save Reconstruction** 仍保存正在优化的层级。纯查看状态不生成训练检查点；若要回到早期层继续训练，使用 **Restore Training Checkpoint**。

每层重新分配体素、梯度和 vBuffer；旧层优化结果保存在磁盘，不永久保留全部 GPU buffer。可视化只按需加载所选历史层的一份独立网格，返回 live 时释放；不分配历史层梯度或路径记录。预览使用独立渲染和累积纹理，训练的跨帧 Spp 累积、loss 和保存图片始终使用当前训练层。新实验清空历史层选择列表。磁盘加载、自动保存和评估导出仍可能短暂耗时，连续模式表示不再等待手动确认升层。

升层时 `gridMin` 不变，`voxelCount` 加倍、`voxelSize` 减半；DDA、相机投影和偏移量读取当前层参数。路径记录与图像尺寸有关，在一次实验内固定大小。三个 mode 的路径命中容量统一为 8，保留单个路径记录 buffer，CPU 与 shader 使用同一组宏。`PathRecord` 大小为 944 字节，编译期检查其不超过 D3D12 的 2048 字节结构化 buffer 元素上限。

每次分裂按父椭球与子体素的几何相交情况决定占据：空父体素的子体素仍为空；占据父体素中，确定不与父椭球相交的子体素置空，其余保留。这个相交判断不使用椭球体积阈值，在中间层和升到最终层时都执行。

保留下来的子体素统一重新初始化为覆盖整格的椭球：局部中心 `(0.5, 0.5, 0.5)`、三个半轴均为 `0.87`、`B=I/(0.87²)`。半径略大于 `sqrt(3)/2`，覆盖局部 `[0,1]³` 的全部角点；实际光线求交仍限制在本体素线段内。父椭球只用于子体素占据筛选，子体素几何从整格覆盖开始继续优化。此修改只用于 mode2 升层后的子体素；最低层的初始椭球及 mode1、mode3 保持原有初始化。

迁移继续复制父体素的 radiance SH；opacity 在方向采样后，拟合 `alpha_child = 1 - sqrt(1 - alpha_parent)` 对应的 logit SH。这个补偿近似保持两次子体素命中相对一次父体素命中的透射率。子体素的几何覆盖会重新扩大，且实际命中数因方向而异，因此升层前后的渲染不保证完全一致，需要细层继续优化。粗层通过占据和外观参数提供初始化先验，没有加入显式空间平滑正则项。

所有层级均开启椭球体积剪枝。`CTF_PRUNE_INTERVAL` 默认 10：在每层第 10、20、30… 轮最后一个视角更新时执行一次，升层后按新层的迭代数重新计时，没有额外 warmup。判据沿用 mode3 的完整椭球体积与体素体积之比，低于 `Ellipsoid Prune Threshold`（默认 0.03）时取消占据，不按 opacity 或点云点数删除。分裂时的父椭球相交筛选与保留子体素的整格椭球初始化保持不变。几何参数在 mode2 新实验初始化时总会赋值；切层不重置用户的学习率。mode1、mode3 的初始化、更新、命中容量和固定分辨率路径保留。

## 保存结果

输出根目录由头文件中的 `ReconstructionDataDir` 指定，默认是项目根目录下的 `Reconstruction_Output`，按 `mode1`、`mode2`、`mode3` 分开使用。

三个 mode 的新 bin 文件名都以 `ReferenceImageDir` 的末级目录名作为场景前缀，路径末尾带分隔符也可识别。例如目录 `Reconstruction_Input/hotdog` 对应 `hotdog_recon9_22_128_radiance_opacity_center_B.bin` 或 `hotdog_stage0_res32_iter40.bin`；mode1/3 的自定义 Name Tag 仍保留。已有文件无需重命名，仍可正常 Load。

```text
Reconstruction_Output/
  mode1/
    hotdog_recon*.bin
    Loss/
  mode2/
    run_<时间>_target128/
      hotdog_stage0_res32_iter40.bin
      hotdog_stage1_res64_iter40.bin
      hotdog_stage2_res128_iter120.bin
      Loss/
      Images/
        hotdog_stage0_res32_iter40_view0_rgb.png
        hotdog_stage0_res32_iter40_view0_alpha.png
        hotdog_stage0_res32_iter40_evaluation.csv
  mode3/
    hotdog_recon*.bin
    Loss/
```

mode2 同一阶段追加训练会生成新的文件；同名保存追加 `_save1`、`_save2` 等编号，不覆盖已有检查点。stage 编号从 0 开始。mode1/3 保留原有保存行为，同名 bin 会被覆盖；需要保留多次实验时使用不同的 **Name Tag**。

mode2 文件包含网格 AABB、当前/目标分辨率、层级预算、训练设置和 loss 历史。mode1、mode3 使用当前 v1 体素文件格式，mode2 使用 v2；场景名前缀只改变文件名，不改变 bin 格式。mode2 的评估图片和 evaluation CSV 使用 bin 的文件名主体，因此也带场景名；mode1/3 的 loss CSV 仍使用日期和 Name Tag 命名，同名会覆盖，不自动添加场景名。

## 加载当前版本的结果

在 **Reconstruction IO** 中，点击 **Refresh Files**，选择文件并点击 **Load Selected Reconstruction**。列表递归扫描当前 `RECON_MODE` 对应的目录，显示相对路径；不会混入其他 mode 的文件。v1 本身没有 mode 字段，所以 mode1、mode3 的归属由文件夹区分。

三个 mode 的 Load 均按文件实际网格尺寸重建资源，暂停优化、取消待执行的初始化与自动升层，并清除旧采样累积。查看结果只需要已加载的场景和场景相机，不要求 PLY、训练图片或参考相机文件存在。`color` 和 `AccuColor` 都显示加载结果，暂停时不会因训练 Spp 大于 1 而将画面亮度除以 Spp。UI 会显示加载成功或具体失败原因；格式、长度或资源分配失败时保留原结果。

mode2 普通 Load 只检查体素布局、mode、文件完整性和网格空间信息；当前目标分辨率、参考数据路径及训练预算不同也可以查看。只有 **Restore Training Checkpoint** 才要求训练数据与当前配置兼容，并恢复迭代、预算、学习率等。加载某层后，**Display level** 的 **Loaded reconstruction** 表示所选文件；该层保留用户选中的版本，其他层优先选本层迭代数最多的文件，迭代数相同时选修改时间最新的文件。

`Images` 使用第 0、N/3、2N/3 个训练视角（去重），固定采样种子，默认 `CTF_EVALUATION_SPP=8`。在最后一次参数更新后重新渲染；RGB PNG 使用 sRGB 编码，alpha PNG 保存线性灰度覆盖率。evaluation CSV 为这些固定训练视角的最新 loss，训练 loss CSV 则记录每轮各视角更新前的平均 loss；两者含义不同，不是验证集指标。

保存或迁移失败会停留在当前层，不自动进入下一层。升层时新旧资源短暂共存，高分辨率仍需考虑显存容量。

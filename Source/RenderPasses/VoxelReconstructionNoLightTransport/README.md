# VoxelReconstructionNoLightTransport：理论与实现

本 Pass 使用多视角 RGBA 图像和相机位姿，把输入点云重建为稀疏体素网格中的可优化椭球。系统联合优化几何、方向相关 radiance 和 opacity，并通过邻域生长补充占据位置、通过多视角证据删除错误几何。

名称中的 NoLightTransport 表示当前训练直接拟合椭球的出射颜色，不求解光照、材质与多次反射。虽然使用了 GaussianEllipsoid 类型，当前前向是椭球硬相交后的 alpha 合成，不是 3DGS 的屏幕空间高斯投影，也不是沿椭球密度积分的体渲染。

本文描述当前实现；参数表是源码默认值，UI 和渲染图属性可以覆盖它们。只修改本文不会改变训练行为。

## 1. 数据表示与输入初始化

### 1.1 一个占据体素对应一个椭球

每个活跃体素保存：

- 占据标记和紧凑体素 ID。
- 体素局部中心 center、世界空间半轴的 logScale、旋转四元数。
- radiance RGB 球谐系数与 opacity logit 球谐系数。
- 梯度、Adam 状态、删除证据和生长生命周期信息。

椭球隐式函数可写为：

```text
q(x) = || diag(1/a) · Rᵀ · (x - c) ||² - 1
a = exp(logScale)
```

其中 x、c 转换到统一的世界空间后计算；q < 0 表示椭球内部。中心限制在所属体素局部 [0.01, 0.99]，但形状可以越过体素面。每个半轴的优化上限为对应 voxelSize 分量的 4 倍；这不是生长阈值。

方向外观为：

```text
radiance(v) = max(Σ RGB_SH[k] · Yk(v), 0)
opacity(v)  = sigmoid(Σ opacity_SH[k] · Yk(v))
```

当前 SH_COUNT 和 SH_OPACITY_COUNT 均为 9。体素的形状、占据位置、方向颜色和透明度是不同概念，不能把低 opacity 直接等同于空体素。

### 1.2 稀疏属性池与完整位置索引

空间查询使用分页 R32Int 三维纹理，保存 cell → compact ID；仅占据体素分配参数记录。反向 cellIndex 保存 compact ID → cell，供梯度、生长和压缩使用。

512³ 的位置索引使用一张页面，1024³ 使用八张 512³ 页面。属性页默认每页 2¹⁸ 条记录，最多 256 页。radiance Adam 独立按需分配，不预先为所有活跃体素铺满其状态。

位置索引并非完全稀疏；稀疏化的是体素参数、梯度和优化状态。UI 显存估计不包含图像、PathRecord、渲染目标和场景等全部资源。

### 1.3 输入与初值

路径定义在 VoxelReconstructionNoLightTransport.h，相对路径以 Falcor 项目目录为根：

- ReferenceImageDir：参考图像及点云所在目录。
- ReferenceCameraFile：训练相机 JSON。
- InitializationPointCloudFile：默认 init_points.ply，即输入 3DGS 的原始稀疏点云；point_cloud.ply 可用于对比优化后的 3DGS 输出。
- ReconstructionDataDir：默认 Reconstruction_Output。

PLY 支持 ASCII 和 binary little-endian。XYZ 点云仅占据每个点落入的 cell；带 opacity、scale、rotation 的 Gaussian PLY 则按高斯覆盖的相交 cell 初始化，并受 Gaussian Opacity Threshold 筛选。该阈值不影响纯 XYZ 输入。NeRF 数据坐标变换为 (x, z, -y)。

普通 XYZ/RGB 点云按 cell 聚合：中心为点位置均值（限制在优化器允许的局部 [0.01,0.99]），PCA 在世界尺度下估计主方向。半轴从协方差与该方向的 0.25 体素宽度下限取较大值，再等比放大到包含该 cell 内全部有效点，最后增加 5% 余量。单点没有方向证据，使用单位旋转及 0.6 × voxelSize 半轴，中心仍位于点附近。重合、共线、共面数据不会生成零半轴。不会自动剔除有限的离群点，因此应使用已清理的输入点云。

支持 red/green/blue 或 diffuse_red/diffuse_green/diffuse_blue。整数 RGB 按类型正最大值归一化，浮点 RGB 约定为 [0,1]；按 sRGB 转为线性 RGB 后逐点求平均，与参考图的 sRGB 纹理读取一致。radiance DC = 平均线性 RGB / Y00，高阶 SH 清零；缺失或无效颜色不影响几何，均无有效颜色的 cell 回退黑色。opacity logit SH 全零，即各方向 alpha = 0.5。点记录仅在 CPU 初始化时暂存，GPU seed 为每个占据 cell 56 字节。

带完整 opacity/scale/rotation 的 Gaussian PLY 保留原覆盖路径：覆盖 cell 中心为 (0.5,0.5,0.5)、半轴 0.6 × voxelSize、radiance 黑色；本次不将 Gaussian SH 或覆盖范围当作普通 RGB 点分布拟合。

## 2. 前向渲染与损失

### 2.1 DDA 与硬相交

RayMarchingPass.ps.slang 用 DDA 遍历射线经过的 cell；在占据 cell 内测试椭球，只把真正相交的椭球作为 hard-hit 参与前向合成。相交范围裁剪到所属 cell 的射线区间，不能仅因为椭球参数越界，就让它在未占据邻居里贡献颜色。

按前到后的命中顺序：

```text
T₁ = 1
C  = Σ Tᵢ · αᵢ · Lᵢ + T_end · C_background
Tᵢ₊₁ = Tᵢ · (1 - αᵢ)
A  = 1 - T_end
```

PathRecord 保存反传所需的命中外观、透射率及几何信息；near-miss 记录用于已有椭球的几何代理梯度，不是空体素生长候选。hard-hit 与 near-miss 记录默认各限制为 16 条；过长路径或提前终止会限制可获得的梯度与证据。

### 2.2 RGB 与 alpha 监督

LossPass.cs.slang 把参考 RGBA 与白底合成，再计算平方误差：

```text
R = reference.rgb · reference.alpha + white · (1 - reference.alpha)
L = ||C - R||² + λ_alpha · (A - reference.alpha)²
```

Alpha Loss Weight 默认 0.5。alpha 监督约束前景覆盖和背景透明，但不能单独决定哪个体素应保留。

背景 carve 是额外的几何优化信号；它不是独立计入当前显示 loss 的项。gBinaryOpacityWeight 虽仍上传，但对应二值 opacity 梯度已注释，当前未实际参与优化。

### 2.3 spp、视角和迭代的区别

一次完整迭代表示遍历全部训练相机一次，不是一个 GPU 帧，也不是一次 Adam 更新：

```text
一轮：随机打乱所有相机
  每个相机：完成 N 个 spp → 累计梯度 → 更新参数一次
所有相机完成：结算该轮 loss → 拓扑操作或升层
```

每轮每个相机恰好访问一次，同一视角的 spp 连续完成。实际相机、参考图像和证据 ID 使用相同的原始数据编号；currentView 只表示本轮遍历位置。查看用 Camera Index 仍按原始编号排列。

多 spp 时，第 0 个样本提供基线，不反传；之后每个样本用不包含自身的前缀平均计算上游残差，再反传当前 PathRecord，以减轻残差与同一个随机样本耦合产生的偏差。1 spp 时直接使用本帧残差。这不意味着硬几何的代理梯度是精确无偏梯度。

显示的轮次 loss 是各视角最后一次 loss 计算的平均，不是额外完整重渲染的评估指标；多 spp 下也不应直接把它视为最终 N-spp 合成图的精确误差。跨实验比较要固定视角、spp、参考数据和评估方式。

## 3. 梯度与 Adam

### 3.1 外观梯度与几何代理

梯度在 GradientPass.cs.slang 中手工计算，不依赖 Slang 自动微分或 Falcor SceneGradients。

外观梯度沿 alpha 合成路径反向累积。几何前向仍是硬相交，反向用世界距离代理：

```text
w = sigmoid(-distanceWorld / tau)
dw / dd = -w · (1 - w) / tau
```

Geometry Tau 默认 0.12 个体素宽度，由 CPU 转换为世界长度。代理作用于 hard-hit 和已有占据体素中的 near-miss，推动 center、logScale 和 rotation。Alpha Geometry Weight 默认 0.1，仅额外缩放 alpha 对几何的代理贡献，不替代图像损失中的 Alpha Loss Weight。

明确背景像素的处理不是只调透明度：

- 禁止 radiance 用改变颜色来拟合背景，背景 hard-hit 的 radiance 梯度置零。
- opacity 仍接受 RGB 和 alpha 梯度，可变得透明。
- 已命中的几何接受 carve 与 alpha 几何信号，可移动、收缩。
- GradientPass.cpp 当前上传 gBackgroundCarveWeight = 0.01。

radiance 在非正颜色区域保留能把颜色推回正区间的梯度，避免黑色初始化后无法恢复。

### 3.2 优化状态与默认值

每个参数组独立保存 Adam 一阶矩、二阶矩和步数；没有更新的组不递增其步数。默认 β₁ = 0.9、β₂ = 0.999、epsilon = 1e-8。梯度在稀疏缓冲中原子累计，默认按有效计数归一化。

| 参数 | 默认学习率 |
|---|---:|
| radiance | 0.001 |
| opacity | 0.0005 |
| center | 0.001 |
| logScale | 0.002 |
| rotation | 0.0005 |

单次几何步长限制为 center 每轴 0.02 个体素、logScale 每轴 0.02、rotation 0.02 rad。Background Carve Adam Multiplier 默认 5，在 Adam 归一化之后，按背景几何样本比例放大几何更新；它不是 gBackgroundCarveWeight。

mode0 的 opacity 前 20 轮冻结，随后 30 轮线性恢复到目标学习率；mode1 每个层级从第一轮就使用目标学习率，没有这段 warm-up/ramp。

旧的“按椭球体积比例直接清空 occupied”逻辑已移除，包括体积计算、CPU 参数、shader 分支和 UI。正常参数更新不再按体积阈值删除体素。

## 4. 两种训练模式与层级调度

| 模式 | 分辨率流程 | 生长 | 多视角删除 |
|---|---|---|---|
| mode0 | 固定分辨率，默认 512 | 默认每 10 轮，δ = 1 | opacity 恢复结束后开启 |
| mode1 | 起始分辨率逐次翻倍到目标 | 所有层级默认每 15 轮 | 仅目标层级开启 |

RECONSTRUCTION_MODE 默认 1。GRID_RESOLUTION 默认 512；COARSE_TO_FINE_START_RESOLUTION 默认 16。实际实验从 64 开始时，应在 UI 设置 Coarse Start Resolution = 64；不能把实验习惯与源码默认值混淆。

mode1 的总轮数由 COARSE_TO_FINE_TOTAL_ITERATIONS 定义，当前为 1000。权重 COARSE_TO_FINE_LEVEL_WEIGHTS 当前为 {19, 12, 14, 25}。

四个活跃层级直接用四个权重；其他层级数在四个控制点之间插值。先为每层保留一轮，再累计加权取整分配剩余轮数，保证总和严格等于总预算。从 64 开始、目标 512、总轮数 1000 时：

| 层级 | 优化轮数 |
|---|---:|
| 64 | 271 |
| 128 | 172 |
| 256 | 200 |
| 512 | 357 |

默认从 16 开始的六层预算为 184、143、120、132、179、242。UI 或宏修改后会重新计算，以上不是硬编码表。

分辨率必须为 2 的幂，起始不大于目标，目标不超过 1024，总轮数不小于活跃层数。mode1 使用 Total Iterations；mode0 使用 Max Iteration。改变模式、起始或目标分辨率会要求下一次初始化重新读取 PLY。

每层的升层/完成边界跳过生长，避免把完全未优化的新一层立刻细分或最终保存。其他完整轮次按层级局部时钟执行生长；目标层级先删除，再生长。

## 5. 升层分裂：粗网格到细网格

这里的“分裂”是替换网格，不同于保留父体素的邻域生长。

### 5.1 只考虑父 cell 对应的八个子 cell

每个父体素最多产生八个子体素；不会因为父椭球很大，就在父 cell 范围外额外生成子 cell。

子 cell 的闭包盒必须与原始父椭球相交。判定使用旋转椭球二次型在盒内的约束最小值，包含面、边、角情况；不是只测试子中心，也不是只测试 AABB。内部子 cell 和边界相交子 cell 都可以保留，完全不相交者为空。纯相切或无法容纳最小可表示子椭球的薄碎片会被拒绝。

Refinement Parent Opacity Threshold 默认 0，表示不按父 opacity 过滤。若提高阈值，使用方向 opacity 的保守 SH 上界筛选父体素；这一筛选与多视角证据删除不同。若全部父体素被过滤，停止升层并保留旧网格。

### 5.2 子椭球初始化

所有保留的子椭球继承父旋转与三轴比例，但尺寸不只是固定等比缩小：

- 八个角都在父椭球内部的子 cell：中心放在子 cell 中心，根据旋转后实际范围缩放。
- 边界子 cell：用可行尺寸搜索和约束中心搜索向父椭球内部靠近；当小幅向内移动能保留足够尺寸时，优先小幅移动。
- 默认允许越过子 cell 面 0.1 个子体素宽度，UI 范围 [0, 0.2]；中心球的半径上限因此约为 0.6 × 子体素宽度。
- 整个子椭球仍须在原始父椭球内部。在父单位球坐标中，同旋转、同轴比的子椭球满足 ||centerOffset|| + scale <= 1。

这些是初始化约束，不是后续优化的永久限制。它们也不能保证子椭球精确铺满父体积或升层前后图像完全一致。

### 5.3 外观与状态继承

radiance 仅继承父体素第 0 项（DC），高阶 SH 系数全部清零。opacity 基于父体素 DC 使用默认 0.65 的光学厚度衰减：

```text
alpha_DC = sigmoid(parentOpacityDC · Y0)
tau_DC   = -log(1 - alpha_DC)
childAlpha_DC = 1 - exp(-0.65 · tau_DC)
```

再调整 opacity SH 的 DC 系数，高阶方向系数全部清零；倍率设为 1 可禁用 DC 衰减，但仍清零高阶项。子体素初始化时不具有方向性，后续优化仍可学习全部 SH 系数。这只是减轻多个子命中叠加的初始化近似，不是逐射线透射率守恒的精确分裂。

新网格的 Adam、梯度和拓扑记录初始化，训练的全局轮数和 loss 历史继续。粗层级的生长绿色标记不会继承到细层级。

先统计实际选中的子体素，再分配新网格并初始化，成功后替换旧网格。升层峰值显存包含新旧网格和临时掩码/偏移缓冲。超出容量或分配失败时停止训练、保留原网格，不静默丢掉部分子体素。

## 6. 邻域生长：可撤销的几何探索

### 6.1 外伸、多父支持与多视角证据

`Use Multi-view Growth Evidence` 默认开启，`26-neighbor Growth` 默认开启：候选包括 6 个面邻居、12 个边邻居和 8 个角邻居，多父支持也在同一邻域统计。关闭后恢复 6 邻域。不同父体素必须超过新生等待期、未标记删除，并且椭球真正接触共享面、有限边或角点，且达到向目标方向的外伸深度；不使用 AABB 重叠代替。斜向深度按每个跨越轴分别计算，例如 (+1,+1,0) 的 0.05 表示 x、y 都越过各自边界 0.05 个体素宽度。两个父体不强制相对；单侧或邻域跨度之外的断口仍可能不满足支持条件。

| UI 参数 | 默认值 | 含义 |
|---|---:|---|
| 26-neighbor Growth | 开启 | 候选与多父支持使用 26 邻域，关闭为 6 邻域 |
| Growth Outward Depth (voxels) | 0.05 | 父椭球越过共享面的深度，以该轴体素宽度为单位 |
| Growth Min Supporting Parents | 2 | 满足成熟期和外伸条件的不同邻居数量 |
| Growth Min Foreground Views | 3 | 至少需要的前景缺失支持视角 |
| Growth Background Veto Views | 2 | 达到该数量的可靠背景视角则拒绝 |
| Growth Min Alpha Deficit | 0.05 | 参考 alpha 减渲染 alpha 的最小缺失量 |

每次计划生长之前收集一整轮训练视角。每个视角使用全部 SPP 的平均 alpha；候选中心及主轴方向的六个内部点投影到参考图和渲染图。一个视角最多贡献一票：任一探针落到可靠背景则投背景票，否则可靠前景中存在足够 alpha 缺失才投支持票。已有充分覆盖的前景、混合轮廓和屏幕外位置不给支持票。这是七点近似，不是精确投影面积或深度验证。

候选几何在证据轮次开始时冻结，使用父 cell 地址适应删除压缩后的 ID 变化；实际提交时重新检查父体及多父支持，支持不足则取消。多个合格父体对同一目标 cell 的提案仍只提交一次，由成功 claim 的父体提供外观与形状，不对多个父体颜色求平均。临时候选记录为每条 56 字节。

关闭 `Use Multi-view Growth Evidence` 可使用旧的 6 邻域外伸阈值模式；`COARSEST_GROWTH_FACE_PENETRATION` 与 `FINER_GROWTH_THRESHOLD_MULTIPLIER` 只影响旧模式。新参数也支持脚本属性 `growthUse26Neighbors`、`growthEvidencePenetration`、`growthMinSupportingParents`、`growthInitialOpacity`。

### 6.2 新生参数、等待与保护

子椭球仅继承父 radiance 的第 0 项（DC），高阶 SH 系数全部清零，后续优化可重新学习方向性。复制父旋转和轴比，中心放在新 cell 内、靠近实际接触位置。Child Contact Offset 默认 0.2 个体素，并应用到每个跨越轴。先以 0.7 倍缩小，再按旋转后的实际范围收缩：只允许向父体所在方向越界，其余面限制在 cell 内；面、边、角生长分别有 1、2、3 个朝父体的面允许越界。

子 opacity 使用 Child Initial Opacity（默认 0.5）：DC 设置为 logit(alpha) / Y0，高阶系数清零，使各方向初始 alpha 一致；默认 0.5 对应全部 opacity SH 系数为零。Adam、梯度与删除证据初始化；父参数及父 Adam 不受生长影响。

| 生命周期设置 | 默认完整轮数 | 作用 |
|---|---:|---|
| Newborn Growth Wait | 5 | 新生体素暂时不能当生长父体素 |
| Newborn Protection | 5 | 新生体素暂时不收集删除票、不具备删除资格 |
| Deleted Cell Cooldown | 5 | 删除位置暂时不能再长回去 |

等待期间仍正常优化外观和几何；等待到期不表示已经稳定。年龄条件和全局生长周期必须同时满足。例如 mode1 子体素在局部轮次 15 出生，默认下一次可参与的周期通常是局部轮次 30。

生长先统计唯一新 cell，再扩充属性页、初始化整层，最后发布有效空间索引。失败则回滚临时 claim 并禁用生长；回滚成功时已有参数仍可继续优化。界面显示 Last growth，是最近一次操作数量，不是累计数量。

### 6.3 ?????

?????????????????????????????????? alpha ??????????????????????????????????????????????????????????????? RGB ???????????????????? SPP ?????????????? opacity???????????????????????

??????????????? opacity loss??????????????

## 7. 多视角证据删除与压缩

### 7.1 局部移除影响

对真实 hard-hit，GradientPass 估计移除当前命中、保持记录路径中其他命中固定时的 RGB/alpha 变化：

```text
ΔC = T_before · opacity · (behindColor - radiance)
ΔA = -T_before · opacity · T_after
ΔL = dot(dL/dC, ΔC) + ||ΔC||²
   + (dL/dA) · ΔA + λ_alpha · ΔA²
```

这是结合当前残差的局部反事实估计，不是删除后重新渲染整个场景的全局 loss，也没有显式检查其他体素能否在重新优化后替代它。

- 可靠前景且 ΔL > 阈值：移除有害，投前景支持票。
- 可靠背景且 ΔL < -阈值：移除有益，投背景冲突票。
- 模糊 alpha、遮挡过强、影响过小或新生保护中的命中不投票。

默认前景 alpha >= 0.95，背景 alpha <= 1e-4，移除影响阈值为 1e-4，T_before 至少 0.05。背景可见性、透明度和随机命中机会都影响证据；没有 hit 不等于证明应该保留。

### 7.2 精确独立相机去重

每个体素分别存前景/背景相机位集合，原子 OR 首次置位才计票。同一窗口内，同一实际相机最多贡献一张前景票和一张背景票，不受像素数量、随机顺序或跨轮重复访问影响。

每个视角每轮只用最终 spp 样本收集证据；多 spp 不额外增加投票权重，1 spp 可以通过多个训练轮次获得不同 jitter 的观测机会。

TOPOLOGY_EVIDENCE_MAX_VIEWS 默认 128，覆盖当前 100 视角数据集。证据记录为 48 字节，包含两个相机集合、计数/标志和生命周期信息。超过容量的数据集会拒绝训练，需提高宏后重编译，不能用哈希碰撞或截断代替独立视角。

### 7.3 窗口资格与删除时机

当前 TopologySettings 默认值：

| 参数 | 默认值 |
|---|---:|
| minDeletionConflictViews | 5 |
| deletionConflictSupportRatio | 2 |
| evidenceInterval | 5 轮 |
| deletionInterval | 10 轮 |

设 B 为独立背景冲突相机数，F 为独立前景支持相机数。一个窗口满足 B >= 5 且 B >= 2F 时具备删除资格；等号成立也算满足。

两个满足条件的窗口确认删除候选，不要求严格连续。空窗口或弱证据窗口保留历史。取消历史/候选需要 F >= max(2, ceil(minConflictViews / ratio))，且背景没有占优势；按当前默认值需要至少 3 个前景支持相机，而不是一张前景票否决。

每个窗口结算后清空相机集合和本窗口计数，保留必要的资格/候选历史。证据收集本身不修改占据状态。

mode0 默认从轮次 50 开始收集，最早轮次 60 检查删除，之后每 10 轮检查。mode1 粗层级不删除；到目标层级重新从该层级局部时钟收集，最早在该层级训练 10 轮后检查。最终完成边界也处理已经确认的候选。

### 7.4 删除后的紧凑池

TopologyDeletion.cpp 用 GPU 生成空洞 ID 与尾部存活 donor ID，将 donor 移到紧凑前缀，修正位置索引，释放完整尾页，并保留适量生长余量。删除 cell 写入短暂空间冷却，避免立即在原位置循环生长。

移动保留外观、几何、几何/opacity Adam 及新生生命周期；独立 radiance Adam 映射和矩状态被重置，以回收孤立槽位。压缩后删除证据重置，下一次确认需要新窗口，新生保护期仍保留。

## 8. 暂停、追加训练与状态继承

停止训练不改变当前几何，暂停的现实时间不计入新生年龄。

在同一个网格上重新 Enable：

- 保留当前参数及现有 Adam 状态；重启轮数/视角时钟与 loss 历史。
- 清空删除证据、位置冷却、新生删除保护和暂态生长等待时钟。
- 保留本次会话中的生长来源标记。
- mode1 从当前分辨率重新分配整份总预算；已在目标层级时，整份预算都在该层级运行。
- mode0 的 opacity warm-up/ramp 随重启的训练时钟再次执行。

Init / Reset from PLY 会重建网格并清空相应状态。升层和文件加载会创建新资源，其 Adam 为初值，不能与同网格再次 Enable 的行为混为一谈。

## 9. 保存、加载与可视化

### 9.1 各层级文件与命名

mode0 保存到 Reconstruction_Output/mode0，mode1 保存到 Reconstruction_Output/mode1。Save Each Completed Level 默认开启，在每次升层前保存当前层级，最终层级完成后自动保存。每个层级是独立 v3 文件，不是一个同时包含所有层级的容器。中间层级保存失败会报告，但不阻止升层。

默认命名包含场景、日期、实验编号、可选标签、分辨率和优化参数，例如：

```text
ship_recon10_6_E0_64_radiance_opacity_center_shape_rotation_adam.bin
ship_recon10_6_E0_512_radiance_opacity_center_shape_rotation_adam.bin
```

扫描同场景/日期下已有 bin 和 Loss CSV 的最大 E 编号，下一次使用最大值加一，无文件从 E0 开始。同一训练的不同层级复用编号；mode0/mode1 独立计数。加载、重新 Enable 或重复保存已经存在的层级，会开启新的命名会话/编号，以免覆盖。文件列表按数字排序，E2 在 E10 前。

CSV 位于对应 mode 的 Loss 子目录，文件名与 bin 的 stem 一致。训练历史可能跨层级累计，层级 checkpoint CSV 不一定只含本层级的数据。

### 9.2 v3 存储边界

v3 保存网格元数据和每个活跃体素的 (cellIndex, VoxelData)。不保存 Adam、删除证据、新生时钟、空间冷却、生长来源标记或完整金字塔调度进度。

Load Selected Reconstruction 加载参数、停止训练并清空当前 loss 历史。加载后可直接查看，不要求原始 PLY；再次 Enable 从保存分辨率优化，使用新 Adam。旧 dense v2 及更早格式不再支持。

### 9.3 切换同一实验层级

加载一个带 E 编号的结果，或当前重建完成并保存后，在 Reconstruction IO → Experiment Level 选择对应已保存分辨率即可直接切换，再选择目标层级可返回最终保存结果。

归组使用同目录下的场景/日期/E 编号/Name Tag，不是只比较 E 编号。训练期间禁用切换。切换会加载文件并替换当前网格、重置 Adam 和 loss，不是保留完整训练状态的临时预览；再次训练从正在查看的层级开始。

只列出实际保存的层级。未保存的粗层级不能从最终网格逆推出。无 E 编号的旧文件仍能手动加载，但不自动归组。Viewing Resolution 控制输出图像尺寸，与体素层级无关：训练固定 800×800，停止时可选择 800×800 或 1920×1080。

### 9.4 生长与删除调试显示

Topology 中 Show Grown Voxels / Stop and Show Grown Voxels 显示绿色生长 cell，Show Occupied Context 可添加灰色上下文；Show Default 恢复正常渲染。删除候选层用红色显示候选。

调试层显示 cell，不依赖椭球 opacity，因此可以看见低透明度的新生位置。生长标记在同网格优化、删除压缩、停止和重新 Enable 后保留，删除后不再显示；升层、重新初始化或加载文件不保留来源标记。不能用绿色 cell 可见性判断正常渲染是否已有有效几何。

## 10. 代码导航与验证

| 代码 | 职责 |
|---|---|
| Defines.h / VoxelReconstructionNoLightTransport.h | 编译默认值、运行时设置、资源与生命周期状态 |
| VoxelReconstructionNoLightTransport.cpp | 按视角/spp 调度、随机顺序、UI |
| PointCloudLoader.cpp / PointCloudInitialization.cpp | PLY、稀疏网格分配与初始化、状态重启 |
| RayMarchingPass.cpp / Shader/RayMarchingPass.ps.slang | 前向 DDA、硬相交、记录路径和调试显示 |
| LossPass.cpp / Shader/LossPass.cs.slang | 图像残差、alpha 分类与上游导数 |
| GradientPass.cpp / Shader/GradientPass.cs.slang | 手工外观/几何梯度、独立相机证据票 |
| UpdatePass.cpp / Shader/UpdatePass.cs.slang | Adam 与参数范围、步长限制 |
| ReducePass.cpp | loss 归约、完整轮次历史 |
| CoarseToFine.cpp / Shader/RefineGrid.cs.slang | 层级预算、父子转移、升层事务 |
| TopologyGrowth.cpp / Math/GrowthEllipsoid.slang / Shader/GrowthPass.cs.slang | 有限面外伸判定、原子 claim、初始化与回滚 |
| TopologyPass.cpp / Shader/TopologyPass.cs.slang | 证据窗口、资格与候选状态 |
| TopologyDeletion.cpp / Shader/DeleteCompactPass.cs.slang | 删除、紧凑池搬移、冷却与资源回收 |
| DataProcess.cpp | v3 IO、E 编号、实验层级归组 |

GPU 回归测试入口为 Tests/NeighborGrowth.cpp 和 Tests/CoarseToFine.cpp，使用小页与合成数据，不依赖实际训练数据集：

```text
配置：-DVOXEL_RECONSTRUCTION_BUILD_GROWTH_TESTS=ON
构建：VoxelNeighborGrowthTests
运行：在对应 Debug/Release 运行目录运行程序；设置 FALCOR_DEVMODE=1 从源码解析 shader
```

覆盖基值/倍数与旧属性迁移、随机视角、独立相机去重及并发像素、生命周期保护、所有层级生长与最终边界、有限面判定、共享邻居去重、属性页扩充/溢出回滚、升层尺寸与父体积约束、Adam 转移、实际优化器绑定、目标层级删除、v3 往返、E 编号和层级查看归组。

回归通过只验证实现约束，不保证真实场景收敛质量、细绳连续性或没有白雾。

## 11. 当前限制与实验排查

- 证据生长结合外伸、多父支持与多视角 alpha 缺失，但七点投影不验证表面深度，仍可能加厚表面或制造漂浮几何；旧模式仅使用几何外伸。
- 单次只加一层不是总数量预算；大量父体素同时触发仍会新增很多体素。
- 新生等待是轮数门槛，不是稳定性检测；接触面越界和后续优化变大仍被允许。
- 升层改变覆盖、命中数量及透明度叠加，不能保证 loss 连续或不突升；较粗初始化也可能丢失点云的细轮廓。
- 当前粗层级不删除，错误粗几何可被传到后续层级；调高后续阈值不会自动清理这些已有后代。
- 前景支持可能遮挡背景反证，反之过强删除也可能损伤细结构。当前局部移除估计不等价于重新优化后的可替代性。
- 低初始 opacity、radiance 尚未稳定、多层透明遮挡均可能产生雾状外观；仅凭白色不能断言 radiance 没收到梯度。
- hard-hit / near-miss 记录容量、透射率提前终止和低 opacity 导数都会影响观测与优化。

排查建议固定起始层级、预算比例、spp、数据与学习率，每次只改一个因素。比较升层前后及每个生长周期前后的图像、活跃体素数、Last growth 和实际阈值，区分“升层初始化产生的缝隙/雾”和“本层级生长新增的雾”。全程 δ = 2 与最粗 1.3、后续 2.6 是不同的几何历史，不能只比较最终层级的阈值。

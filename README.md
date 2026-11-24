# 夜间检测

检验百图夜间标注能否恢复到白天源模型的大部分水平，并比较随机与一致性/多样性选样，回测白天性能保留。

## 状态

**白天源模型 A 已按五类受限冻结。** 夜间零样本基线和后续百图对照共用 [`align_001/17/last.pt`](runs/align_001/17/last.pt)。在完整 500 张白天开发图和锁定多视图协议下，行人、骑行者、汽车、卡车、有轨车辆在三个种子上通过全部 15 项类别验收。行人召回（P≥75%）分别为 63.22%、60.74%、61.57%。[冻结决定与权重身份](specs/s4/plan.md) · [逐类结果和复算命令](reports/a.md#整景尺度适配三种子五类通过) · [夜间基线与百图对照准备](specs/s5/plan.md)

**S6 本轮收束，百图恢复目标未达成。** [随机臂 D](reports/night.md#百图随机臂)及 [E1/E2](reports/night.md#百图策略训练)均有三种子开发证据；追加的[机制对照](specs/s6/plan.md#百图恢复决策轨迹)仍未找到同时通过两域要求的配方。有轨车辆和卡车是主要缺口，见[阶段结论](reports/night.md#百图阶段结论)。最终留出未启动。[整体进度](plan.md#阶段)。S2 诊断已执行，阶段结论仍待单独复核。

**S7 已完成 seed 43 的 300 张开发对照。** 先随机、再按同预算比较 E1/E2；三臂白天保护均通过，但夜间均只过 3/5。E2 卡车改善，整体选图优势是否扩大尚未确认；[结果](reports/night.md#300-张策略对照)。不作为百图达标，最终留出未启动。

<details><summary>白天历史记录（以下为当时状态）</summary>

`alias_002` 的 seed 29 完成正式训练并关机，行人 AP50 70.10%，但召回 56.40%、汽车严格 AP 67.976%，未满足五类要求；没有继续该训练配方的 17/43。跨初始化平均和外观校准也未入围，见[别名正式对照](reports/a.md#行人别名正式对照)。
2026-11-22 的端点平均、末轮、整图外观与行人分类行适配尚未补上 seed 29 的缺口。平均权重达到行人 AP50 70.22%，但在 Precision≥75% 时只有 285/484 人；没有据此替换 `seed_002`。见[端点对照](reports/a.md#端点平均与整图外观)与[分类先验](reports/a.md#行人先验迁移)。

本轮固定权重复核未找到可替代 `seed_002` 的方案。等召回诊断和训练侧原图盲审见[正负配对](reports/a.md#同召回下的误报与训练侧配对)；整图 960 复核对 seed 29 的工作点有局部增益，但更高召回下出现大量重复框，三种子行人 AP50 均未过线，见[整图对照](reports/a.md#整图位置复核)。本轮未启动训练。

行人跨种子差距已定位到工作点排序：seed 29 相对 17 丢失的 36 人全都有贴合但低分的框。裁图、骑行者竞争降分、逐候选复看均未替代锁定方案。本机 `teacher_002` 配对训练已完成：在规定精度下教师臂比普通臂少检出 22 人，不扩大该配方。见[教师试验](reports/a.md#跨尺度教师试验)与[行人跨种子缺口](reports/a.md#行人跨种子缺口)。

`seed_002` 原配方三种子复验已完成：骑行者、汽车、卡车、有轨车辆三臂均达标；行人只有 seed 29 未达标，三轮车 seed 29／43 均未达标。只有 seed 17 六类过线，A 未冻结。[完整结果](reports/a.md#原配方三种子复验) · [封存](runs/seed_002/manifest.json)

最新跨种子推理 `seed_001`：在两条已有 640 短程微调权重上，锁定多视图未稳定复现六类过线。骑行者、卡车和有轨车辆两臂均达标，行人与三轮车两臂均未达标；汽车只有 seed 29 过线。这不是 `a_001` 原配方三 seed 正式复验，未冻结 A。[两臂结果](reports/a.md#多视图跨种子验证)

三轮车候选 `tricycle_001` 已完成原图复现：在锁定 `scale_004` 的五类输出上，只更换三轮车的多尺度分支，白天开发集六类数值门槛全部通过。三轮车仅 22 个真值，区间证据仍不足；A 尚未冻结。详见[三轮车报告](reports/a.md#三轮车多尺度)和[封存清单](runs/tricycle_001/manifest.json)。

白天多视图开发候选已锁定：保留 `a_001 best` 的原 640 非行人输出，用 960/1280 原图和镜像融合行人。行人 AP50 **71.61%**、AP50:95 **44.22%**、Precision **75.75%**、Recall **62.60%**；行人、骑行者、汽车、卡车、有轨车辆达标，**仅三轮车未达标**。这不是 A 冻结；夜间和终评未开始。见[锁定记录](specs/s3/scale.md#候选锁定)、[报告](reports/a.md#行人多视图融合)和[产物清单](runs/scale_004/manifest.json)。

首次正式训练`a_001`已完成100轮，best第80轮；原source_dev行人/汽车AP50:95宏平均0.500039822187496。首训工程验收通过、best/last本地SHA通过，但A尚未完成六类质量验收或冻结。S2自动诊断与Agent主审已完成，本轮用户复验已反馈，阶段结论待单独复核，详见[阶段计划](plan.md)与[验收规则](RULES.md)。

固定best的统一诊断覆盖4000张train和500张dev，规则工作点下行人Recall39.88%、三轮车31.82%，仍未过线；938项gpt-6-luna视觉主审已完成，用户本轮反馈74项判断并认可初审可采用，不确定项按需复验，不能称A冻结。S3资格选模、固定100轮及dev-only/本地最终评价已实现并通过本地回归。用户已重开云端并授权预检通过后直接运行seed17的R/S；两臂完整预检已通过，R/S均完成100轮且完整取回本地，本地完整评价与首对比较已完成，当前采样未带来dev小/细行人净收益；实际状态以[计划](plan.md)和runs记录为准。最终测试和夜间实验未启动，A仍未冻结；费用估计不等于账单核验。

</details>

## 入口

| 内容 | 位置 |
|---|---|
| 结构与规则 | [STRUCTURE.md](STRUCTURE.md)、[AGENTS.md](AGENTS.md) |
| BitaHub GPU | [开机、状态与自动收尾](GPU.md#bitahub-开发机) |
| 实验验收与冻结 | [RULES.md](RULES.md) |
| 计划 | [plan.md](plan.md) |
| 阶段文档 | [当前 S6 计划](specs/s6/plan.md) · [S5 夜间验收](specs/s5/plan.md#恢复目标) · [阶段目录](specs/) |
| 固定best诊断 | [运行身份](runs/a_001_diag/run.json)、[数值对照](runs/a_001_diag/comparison.csv)、[初审与复审队列](runs/a_001_diag/cases.jsonl)、[状态](runs/a_001_diag/review.json)；结论见[a.md诊断](reports/a.md#诊断) |
| 数据报告 | [data.md](reports/data.md) |
| 正式训练报告 | [a.md](reports/a.md) |
| 4000张训练图/标签 | [images](data/soda10m/splits/source_v1/images/source_train)、[labels](data/soda10m/splits/source_v1/labels/source_train) |
| 500张开发图/标签 | [images](data/soda10m/splits/source_v1/images/source_dev)、[labels](data/soda10m/splits/source_v1/labels/source_dev) |
| 数据身份 | [READY.json](data/soda10m/splits/source_v1/READY.json) |
| 冻结源模型 A | [`align_001/17/last.pt`](runs/align_001/17/last.pt) · [冻结记录](runs/align_001/17/freeze.json) · [三 seed 验收](runs/align_001/comparison.json) · [复算入口](scripts/verify_scene.py) |
| 历史原 A 权重 | [best.pt](runs/a_001/weights/best.pt)、[last.pt](runs/a_001/weights/last.pt) |
| 固定best的train评价 | [metrics.json](runs/a_001_train/metrics.json)、[train/dev对照](runs/a_001_train/comparison.csv)、[图](plots/a_001_train/ap.png) |
| 逐轮指标 | [results.csv](runs/a_001/train/results.csv)、[source_metrics.jsonl](runs/a_001/train/source_metrics.jsonl) |
| 参数与日志 | [args.yaml](runs/a_001/train/args.yaml)、[training.log](runs/a_001/job/guarded/training.log) |
| 完成与验收 | [审计](runs/a_001/job/COMPLETION_AUDIT.json)、[验收](runs/a_001/review.md) |
| 留存清单 | [manifest.json](runs/a_001/manifest.json) |
| 曲线 | [loss.png](plots/a_001/loss.png)、[ap.png](plots/a_001/ap.png) |
| 配置与环境说明 | [配置](configs/README.md)、[环境](environment/README.md) |

## 检查

```powershell
.venv-data/Scripts/python.exe scripts/make_splits.py verify --stage source
.venv-data/Scripts/python.exe scripts/make_splits.py verify --stage night
.venv-data/Scripts/python.exe scripts/verify_run.py --run runs/a_001
.venv-data/Scripts/python.exe -m unittest discover -s tests/data -v
.venv-train/Scripts/python.exe -m pytest tests/train -q -p no:cacheprovider
```

这些检查不运行模型最终测试。Git同步代码、配置/环境锁、规划、报告及可复建图表；`.gitignore`排除数据、原始实验记录、权重、虚拟环境、缓存、日志和凭据。这些本地内容不会随clone恢复，报告中指向data/runs/weights的链接用于本地追溯；数据和权重另按许可及实验清单管理，不通过Git上传。

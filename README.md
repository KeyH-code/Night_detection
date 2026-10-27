# 夜间检测

有限夜间标注预算下比较随机与一致性/多样性选样，检查漏检、误报及白天性能保留。

## 状态

首次正式训练`a_001`已完成100轮，best第80轮；原source_dev行人/汽车AP50:95宏平均0.500039822187496。首训工程验收通过、best/last本地SHA通过，但A尚未完成六类质量验收或冻结。S2自动诊断与Agent初审已完成，待用户复审，详见[阶段计划](plan.md)与[验收规则](RULES.md)。

固定best的统一诊断覆盖4000张train和500张dev，规则工作点下行人Recall39.88%、三轮车31.82%，仍未过线；938项gpt-6-luna视觉初审待用户复审，不能称A冻结。活动训练脚本尚未落实新资格选模和停止规则。用户确认旧云任务已关机/已停止，未独立核验API或实际账单。最终测试、夜间实验和新训练未启动。

## 入口

| 内容 | 位置 |
|---|---|
| 结构与规则 | [STRUCTURE.md](STRUCTURE.md)、[AGENTS.md](AGENTS.md) |
| 实验验收与冻结 | [RULES.md](RULES.md) |
| 计划 | [plan.md](plan.md) |
| 阶段文档 | [specs/](specs/)，每阶段独立目录；当前[S2计划](specs/s2/plan.md)待用户复审关闭 |
| 固定best诊断 | [运行身份](runs/a_001_diag/run.json)、[数值对照](runs/a_001_diag/comparison.csv)、[初审与复审队列](runs/a_001_diag/cases.jsonl)、[状态](runs/a_001_diag/review.json)；结论见[a.md诊断](reports/a.md#诊断) |
| 数据报告 | [data.md](reports/data.md) |
| 正式训练报告 | [a.md](reports/a.md) |
| 4000张训练图/标签 | [images](data/soda10m/splits/source_v1/images/source_train)、[labels](data/soda10m/splits/source_v1/labels/source_train) |
| 500张开发图/标签 | [images](data/soda10m/splits/source_v1/images/source_dev)、[labels](data/soda10m/splits/source_v1/labels/source_dev) |
| 数据身份 | [READY.json](data/soda10m/splits/source_v1/READY.json) |
| 权重 | [best.pt](runs/a_001/weights/best.pt)、[last.pt](runs/a_001/weights/last.pt) |
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

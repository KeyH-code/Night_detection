# 夜间检测

有限夜间标注预算下比较随机与一致性/多样性选样，检查漏检、误报及白天性能保留。

## 状态

正式A已完成100轮，best第80轮；source_dev行人/汽车AP50:95宏平均0.500039822187496。总控独立验收通过，best/last均本地验SHA通过。用户确认平台已关机/已停止，未独立核验API或实际账单。最终测试与夜间实验未运行，后续等待用户安排。

## 入口

| 内容 | 位置 |
|---|---|
| 结构与规则 | [STRUCTURE.md](STRUCTURE.md)、[AGENTS.md](AGENTS.md) |
| 计划 | [plan.md](plan.md) |
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

这些检查不运行模型最终测试。数据、运行记录和权重不入Git；公开发布前另核验许可。

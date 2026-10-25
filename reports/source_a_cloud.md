# 白天源模型 A：云端启动与取回

本包仅供本项目私有迁移。包含冻结 source_train 4000 张与 source_dev 500 张、必要代码、环境锁及独立下载的 COCO 预训练；不含 source_test、night_pool/dev/test、白天跨场景最终评价的清单、图片或标签。READY 保留测试用途计数及哈希元数据，不读取对应数据。SODA10M 的数据许可仍适用，不公开分发此包。

当前终点是 READY_FOR_CLOUD；本机 smoke 与云端预检均不是正式 A。云端实例、平台、连接方式、持久盘挂载与单价尚未提供；本阶段未租赁、支付或创建资源。Linux 锁由 Windows 上 uv 针对 Linux x86_64/Python3.12 解析并带包哈希，尚未在 Linux 实例安装运行。

## 1. 传输与初始检查

需要单张 RTX4090 24GB、至少8 vCPU、32GB RAM、50GB持久存储；把工作目录放在持久盘。需要 Python3.12、venv、可运行 CUDA13.0 的 NVIDIA 驱动；先核对宿主 nvidia-smi。不要通过安装命令升级宿主驱动。

将交付归档与其 `.json` 摘要传入实例，先比较归档 SHA-256（见交付报告），然后解压。以下在自定持久盘目录执行，路径可包含中文与空格：

```bash
sha256sum source_a_kit_v2.tar.gz
tar -xzf source_a_kit_v2.tar.gz
cd source_a_kit
python3.12 scripts/verify_source_bundle.py
bash environment/setup-linux.sh
.venv-train/bin/python -m pytest training_tests -q --basetemp .cache/training/pytest
```

`verify_source_bundle.py` 无 torch 依赖，会验证包内全部文件哈希、数据精确白名单、READY、映射、协议、数据指纹和9000个训练／开发数据文件。`setup-linux.sh` 的安装临时文件、pip和训练库缓存均位于包目录 `.cache/training`；不会使用已有学习环境。环境检查会保存实际版本、CUDA核验和 pip freeze；不得把 Linux 解析成功当成实际安装成功。

包内另有官方 `yolo26n.pt`，仅供锁定版本 Ultralytics 的内部 AMP 数值兼容性检查，避免检查时再次联网下载；它不参与源模型 A 初始化。正式起点始终是 `yolo11m.pt`。Linux 基础镜像还需具备 OpenCV 的系统共享库；若环境检查报告缺少 libGL/libglib 等库，按平台镜像说明补齐后重新检查，不能绕过检查开始正式训练。

固定白天 READY SHA-256：

```text
72e75ebd99014c830e0227061e390e6aed768ae724b532a5c2c90fd8dec93e85
```

## 2. 云端预检与配方冻结

在包根执行（将 PRICE 改为平台实际每小时单价，货币单位由运行记录注明；未提供时命令可省略价格参数，费用保持未知）：

```bash
PRICE=0.0  # 使用真实单价替换后执行；0.0不产生费用估算
CURRENCY=CNY  # 改为平台实际计费币种
.venv-train/bin/python scripts/train_source.py preflight \
  --data-root data/source_v1 --config configs/source_a.yaml \
  --ready-sha256 72e75ebd99014c830e0227061e390e6aed768ae724b532a5c2c90fd8dec93e85 \
  --output runs/source_a/cloud_preflight_v1 --hourly-price "$PRICE" --currency "$CURRENCY" \
  > cloud_preflight.log 2>&1
```

预检使用全部4000／500张、2轮，从原始 COCO 预训练开始。先 batch16；OOM 时丢弃该次运行作为正式初始化来源，从原始预训练重新以 batch8 开始，仍失败就停止并保留诊断。库内部自动降 batch 会被拦截。记录显存峰值、包含开发评价的吞吐、单轮耗时、真实优化器参数，成功才生成 `runs/source_a/cloud_preflight_v1/FROZEN_RECIPE.json`。预估100轮时间按预检平均单轮时间外推，有启动／缓存与早停偏差；实际账单另记。

## 3. 正式 A

预检输出 `FROZEN_RECIPE.sha256.json`；先记录其中SHA-256到云端运行日志。将下方 `RECIPE_SHA` 替换为该记录值，正式入口核验配方字节身份：

```bash
RECIPE_SHA=替换为已记录的64位配方SHA256
.venv-train/bin/python scripts/train_source.py formal \
  --data-root data/source_v1 --config configs/source_a.yaml \
  --ready-sha256 72e75ebd99014c830e0227061e390e6aed768ae724b532a5c2c90fd8dec93e85 \
  --recipe runs/source_a/cloud_preflight_v1/FROZEN_RECIPE.json \
  --recipe-sha256 "$RECIPE_SHA" \
  --output runs/source_a/formal_a_v1 > formal_a.log 2>&1
```

formal 入口检查数据、配置、适配器代码和原始权重身份，使用预检冻结的 batch，再次从包内 `weights/yolo11m.pt` 初始化。不会用本机或云端预检的权重启动 A。100轮上限、SGD与其它指定参数来自 `configs/source_a.yaml`；其余设置来自锁定 Ultralytics 默认值。

以 source_dev 中 pedestrian 和 car 的真实类别 AP50:95 宏平均选择 best，并用于 patience20；分数相同保留早期权重且不重置耐心值。六类 AP 同时保留，辅助类不存在时记录 null。开发集只用于评价／选模，无开发集反向传播。最终测试始终不运行。

输出同时保存 `config_resolved.json`、`defaults_locked.json`、`trainer_passed.json`、`trainer_effective.json` 和 `optimizer_effective.json`；实际调整或缩放的权重衰减、累积步数、AMP状态以最后两项为准。`source_metrics.jsonl` 是逐轮主指标与六类指标；Ultralytics `results.csv` 仍保留通常的全类指标。

## 4. 断点续训

```bash
.venv-train/bin/python scripts/train_source.py resume \
  --data-root data/source_v1 --config configs/source_a.yaml \
  --ready-sha256 72e75ebd99014c830e0227061e390e6aed768ae724b532a5c2c90fd8dec93e85 \
  --checkpoint runs/source_a/formal_a_v1/train/weights/last.pt \
  --output runs/source_a/formal_a_v1 > formal_a_resume.log 2>&1
```

保留 `.pt` 及相邻 `.pt.sha256.json`；续训检查文件哈希和内嵌身份，校验数据／配置／代码／原始权重／batch／scope，然后恢复epoch、优化器、AMP scaler、EMA、调度进度与早停状态。每5轮保存 `epoch005.pt` 等，last、best也保留优化器状态。已完成100轮的checkpoint会被拒绝继续训练；已早停但未到100轮的恢复应核对项目是否仍需运行，不改配置隐式增加轮数。换路径后入口从当前已核验数据重新构建沙箱，不使用checkpoint里的旧数据绝对路径。

## 5. 取回与验收

在已授权连接的实例上打包结果（不重新打包原始数据），再用平台下载或用户提供的 scp 连接信息取回：

```bash
tar -czf source_a_results.tar.gz \
  runs/source_a/formal_a_v1/train \
  runs/source_a/formal_a_v1/environment \
  runs/source_a/cloud_preflight_v1/FROZEN_RECIPE.json \
  runs/source_a/cloud_preflight_v1/FROZEN_RECIPE.sha256.json \
  artifacts/cloud_environment formal_a*.log cloud_preflight.log
sha256sum source_a_results.tar.gz
```

运行目录的 train 子目录只含训练日志／权重，外部 `data` 或 `data_resume_*` 沙箱无需取回。取回后核验权重及sidecar哈希、best选择来源、finite loss、真实生效配置、训练与开发数量、正式起点、运行时长和实际费用，再更新 `reports/source_a.md`。正式完成前不得填写正式开发性能或最终测试性能。

## 6. 本机 smoke 重现

在原工作区使用独立 Windows 环境：

```powershell
& ./.venv-train/Scripts/python.exe scripts/train_source.py smoke --data-root data/soda10m/splits/source_v1 --config configs/source_a.yaml --ready-sha256 72e75ebd99014c830e0227061e390e6aed768ae724b532a5c2c90fd8dec93e85 --output runs/source_a/local_smoke_new
```

64／32张确定性抽样兼顾六类并保留可用空标注，YOLO11m/640/AMP、batch2、workers0，总共2轮。第一轮完成后保留真实checkpoint，在相同2轮训练计划内续训第二轮；OOM才从COCO重试batch1。重复时使用新的输出目录，保留已有证据。成功文件 `SMOKE_PASSED.json` 明确标示 smoke，不是正式 A 或性能结果。

# 环境

数据与训练环境分开，不修改已有学习环境。

- 数据：`python -m venv .venv-data`，随后`.venv-data/Scripts/python.exe -m pip install -r environment/data.txt`。数据测试不需要torch。
- Windows训练：使用Python3.12运行`powershell -File environment/setup-windows.ps1`。
- Linux训练：使用Python3.12运行`bash environment/setup-linux.sh`；迁移包会先做白名单校验。两平台使用带哈希依赖锁，不能静默升级版本。
- `training.in`为依赖需求，`requirements-windows.lock`、`requirements-linux.lock`为安装锁；正式A的Linux实际环境及当时锁另存`runs/a_001/environment/`。

框架版本：torch2.14.0+cu130、torchvision0.29.0+cu130、Ultralytics8.4.171。云端资源使用cgroup/affinity实际配额检查；workers4须通过共享内存门禁，不能用宿主资源冒充实例配额。安装入口将缓存和YOLO配置隔离到任务`.cache`，不写用户全局配置。运行后不需要的缓存及时删除。

本地只读检查：`.venv-train/Scripts/python.exe scripts/train_source.py verify --ready-sha256 72e75ebd99014c830e0227061e390e6aed768ae724b532a5c2c90fd8dec93e85`。

训练入口：`scripts/train_source.py`支持smoke/preflight/formal/resume；云端使用`run_source_cloud_job.py`强制资源/身份门禁及预检收据，不绕过此入口。`--help`列出所需的外部身份、配方哈希及输出参数。新任务先保存完整runtime身份，再预检，审核后从原始COCO权重开始正式训练；S3按阶段计划和用户当前执行安排运行。

打包：`scripts/bundle_source.py --ready-sha256 <READY哈希> --output <项目外临时包.tar.gz>`。仅包含白天train/dev白名单、代码、配置和锁；`--smoke`可附本次验证收据，不能依赖旧smoke目录。解压后先运行`verify_source_bundle.py --root <解压目录>`。传输包及解压副本任务结束删除，正式证据留在runs。

固定权重的单用途评价用`evaluate.py --weights <权重> --weights-sha256 <外部SHA> --ready-sha256 <READY外部SHA> --purpose source_train --output <新结果目录>`；可用`--reference <既有best_dev_metrics.json>`直接生成对照，不重跑dev。冻结数据只读，标签缓存和复制数据在仓库外临时目录，结果不复制权重/图片。

以后云端每轮完整train观察结果在train/train_metrics.csv与train/train_monitor.json，取回正式证据时必须包含这两份文件；启动器核对train/dev轮次覆盖，缺失不能晋升完成/预检通过。先在同一实例预检实测开销，不使用本机batch1耗时外推100轮账单。

## 诊断

固定best的S2入口为scripts/diagnose.py，只允许source_train/dev。使用现有Windows锁，不新增依赖；完整集采集含沙箱副本，临时目录必须在仓库外。下面的scratch示例按实际任务设置，不复用来源不明的临时目录；collect拒绝覆盖结果目录。

```powershell
.venv-train/Scripts/python.exe -B scripts/diagnose.py collect --scratch D:/Temp/night_s2_a001
.venv-train/Scripts/python.exe -B scripts/diagnose.py analyze --scratch D:/Temp/night_s2_a001
.venv-train/Scripts/python.exe -B scripts/diagnose.py trace --scratch D:/Temp/night_s2_a001
.venv-train/Scripts/python.exe -B scripts/plot.py --diagnostic runs/a_001_diag
.venv-train/Scripts/python.exe -B scripts/diagnose.py audit --scratch D:/Temp/night_s2_a001
.venv-train/Scripts/python.exe -B scripts/diagnose.py review --scratch D:/Temp/night_s2_a001 --port 8766
```

默认权重、READY及外部SHA绑定a_001/source_v1，规则门槛直接读取RULES.md并保存其SHA。collect做工程等价核验和完整4500图采集，analyze只读导出数值计算AP、实际工作点/错误/切片/区间，trace按登记子集补采中间候选；不修改现有evaluate、monitor或云门禁。

review在127.0.0.1提供简单临时审核页，仅按已核验key读取图片，不提供任意文件接口。用户授权的Agent初审包使用`diagnose.py import-initial --scratch <任务临时目录> --review-file <汇总JSON>`，包内phase=agent_initial且绑定队列身份；仅写cases的initial_review，不写人工通过状态。页面可按初审疑点筛选。

用户在浏览器保存并导出phase=human_review的JSON后，用`diagnose.py import-review --scratch <任务临时目录> --review-file <实际导出路径>`核验权重、READY、导出对象及队列身份再写入cases；Agent包不能误用该入口。`diagnose.py seal --scratch <任务临时目录>`按真实审核状态生成留存清单。用户复审未完成时只标awaiting_review，不称S2验收完成。网页使用结束关闭服务并清理外部任务目录；待复审材料继续保留并告知位置。

长期回归在tests/train/diagnose/，包含于本机`pytest tests/train`递归检查；不进入只收集顶层训练回归的云训练包。不要为了本机诊断更新旧云预检/门禁身份。必要的后续云训练仍须另建身份、重新预检。

2026-10-27起使用Agent主审/人工抽查。audit记录固定100项名单且不写人工结论；review默认每页12项，无技术分组导航。选择结论及翻页自动保存到外部review_draft.json，重开恢复，并保留浏览器备份；未选择不默认同意。导出JSON仍由import-review校验并导入，空JSON不补造判断。正式抽查完成不等于全938项人工确认，seal仍保持awaiting_review等待阶段验收。

人工反馈留存：选择结论之前也可以只写备注，导出中的decision=null表示尚未定论，不会计为人工完成。输入即保存，同意/改判/翻页不清空备注；导出含全部当前文字及必要note_history。用户已认可本轮Agent初审可采用，原100项名单只作追溯，不要求继续补齐。不确定项按需复验，正式阶段验收另行复核。

## S3

训练活动入口以configs/source_s3.yaml区分新策略；云端连接、预检与正式运行按用户当前授权安排，实际阶段状态见根plan.md。本地回归和评价使用现有.venv-train，不静默安装或升级依赖。运行时缓存默认在系统临时目录的night_source_runtime，可用NIGHT_SOURCE_SCRATCH指定项目外任务目录；S3数据沙箱、评价缓存与一次性核验退出后清理。安装脚本的既有.cache约定不据此改写。

未来各臂/seed的云命令必须显式传入`--config <根>/configs/source_s3.yaml --arm R|S --seed 17|29|43`，并经run_source_cloud_job.py提供外部runtime/runner哈希。S3的train_source.py拒绝缺少资源门禁身份的直接preflight/formal/resume；Linux、锁版本、完整数据、batch16/workers4及worker收尾须在新实例实测。旧BASE_CODE门禁只保留历史兼容检查，新S3绑定当前代码、规则、已审计划、原配方与基线数值、库源码、配置/臂/seed；不能复用旧A收据。

完整预检保持100轮学习率/关闭Mosaic日程，在第2轮后受控停止；失败预检重新从COCO开始。每轮4000训练槽位/500 dev选模及权重重载的真实时间/空间单列；4000 train评价、最终逐图导出和配对分析在本地完成，预检两轮属于冷启动/warmup，不冒充稳态/关闭Mosaic阶段测量。formal恢复只接受相同新身份的完整轮last，重新从核验过的白名单建立外部沙箱；不读已失效的数据路径。

本地复用检查：`python -B -m pytest tests/data tests/train -q -p no:cacheprovider --basetemp <项目外本轮测试目录>`。包含真实workers4消费/关闭Mosaic/退出、合成模型连续与恢复数值对照、CUDA下EMA与重载一致性，以及门禁/无best分支；这些不是正式候选训练或新云预检。

本地收尾使用 evaluate.py --finalize-run runs/a_002/train --ready-sha256 <冻结SHA>（a_003同理）。需完整回传train目录，不只下载角色权重；默认source_v1路径为本项目冻结目录。入口核验代码/文件哈希及框架版本，生成本地train/dev证据并更新evaluation_status；跨设备差异记录，不静默更换阈值/权重。云端pending_local_evaluation阶段不能直接配对分析。

Windows启动本地评价前，先在同一PowerShell中将TEMP/TMP和NIGHT_SOURCE_SCRATCH指向仓库外D盘任务目录，避免沙箱落到系统盘。必须在启动Python之前设置，不能依赖进程启动后更改环境变量来覆盖tempfile已缓存的路径。以下命令会实际运行完整评价，准备或下载检查时不执行最后一行：

```powershell
$taskScratch = 'D:\Temp\night_s3_local'
New-Item -ItemType Directory -Path $taskScratch -Force | Out-Null
$env:TEMP = $taskScratch
$env:TMP = $taskScratch
$env:NIGHT_SOURCE_SCRATCH = $taskScratch
$env:PYTHONUTF8 = '1'
$env:PYTHONDONTWRITEBYTECODE = '1'
.venv-train\Scripts\python.exe -B scripts/evaluate.py --finalize-run runs/a_002/train --ready-sha256 72e75ebd99014c830e0227061e390e6aed768ae724b532a5c2c90fd8dec93e85 --device 0
```

两臂需使用同一套本地环境及设备设置。各自完成本地收尾后，再以source_policy.compare_runs读取runs/a_002/train与runs/a_003/train生成配对结果；未完成本地评价的run会被拒绝。最终工作点沿用云端选定角色的dev阈值，训练集不重选阈值。下载的transfer.json只记录取回校验，不能当作本地评价完成。

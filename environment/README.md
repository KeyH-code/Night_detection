# 环境

数据与训练环境分开，不修改已有学习环境。

- 数据：`python -m venv .venv-data`，随后`.venv-data/Scripts/python.exe -m pip install -r environment/data.txt`。数据测试不需要torch。
- Windows训练：使用Python3.12运行`powershell -File environment/setup-windows.ps1`。
- Linux训练：使用Python3.12运行`bash environment/setup-linux.sh`；迁移包会先做白名单校验。两平台使用带哈希依赖锁，不能静默升级版本。
- `training.in`为依赖需求，`requirements-windows.lock`、`requirements-linux.lock`为安装锁；正式A的Linux实际环境及当时锁另存`runs/a_001/environment/`。

框架版本：torch2.14.0+cu130、torchvision0.29.0+cu130、Ultralytics8.4.171。云端资源使用cgroup/affinity实际配额检查；workers4须通过共享内存门禁，不能用宿主资源冒充实例配额。安装入口将缓存和YOLO配置隔离到任务`.cache`，不写用户全局配置。运行后不需要的缓存及时删除。

本地只读检查：`.venv-train/Scripts/python.exe scripts/train_source.py verify --ready-sha256 72e75ebd99014c830e0227061e390e6aed768ae724b532a5c2c90fd8dec93e85`。

训练入口：`scripts/train_source.py`支持smoke/preflight/formal/resume；云端使用`run_source_cloud_job.py`强制资源/身份门禁及预检收据，不绕过此入口。`--help`列出所需的外部身份、配方哈希及输出参数。新任务先保存完整runtime身份，再预检，审核后从原始COCO权重开始正式训练；当前没有新的训练授权。

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
.venv-train/Scripts/python.exe -B scripts/diagnose.py review --scratch D:/Temp/night_s2_a001 --port 8766
```

默认权重、READY及外部SHA绑定a_001/source_v1，规则门槛直接读取RULES.md并保存其SHA。collect做工程等价核验和完整4500图采集，analyze只读导出数值计算AP、实际工作点/错误/切片/区间，trace按登记子集补采中间候选；不修改现有evaluate、monitor或云门禁。

review在127.0.0.1提供简单临时审核页，仅按已核验key读取图片，不提供任意文件接口。用户授权的Agent初审包使用`diagnose.py import-initial --scratch <任务临时目录> --review-file <汇总JSON>`，包内phase=agent_initial且绑定队列身份；仅写cases的initial_review，不写人工通过状态。页面可按初审疑点筛选。

用户在浏览器保存并导出phase=human_review的JSON后，用`diagnose.py import-review --scratch <任务临时目录> --review-file <实际导出路径>`核验权重、READY、导出对象及队列身份再写入cases；Agent包不能误用该入口。`diagnose.py seal --scratch <任务临时目录>`按真实审核状态生成留存清单。用户复审未完成时只标awaiting_review，不称S2验收完成。网页使用结束关闭服务并清理外部任务目录；待复审材料继续保留并告知位置。

长期回归在tests/train/diagnose/，包含于本机`pytest tests/train`递归检查；不进入只收集顶层训练回归的云训练包。不要为了本机诊断更新旧云预检/门禁身份。必要的后续云训练仍须另建身份、重新预检。

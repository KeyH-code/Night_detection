"""Render training-owned report from actual local evidence. No dataset mutation."""
import json
import xml.etree.ElementTree as ET
from pathlib import Path

from source_contract import check_equal, read_json, sha256, write_json
from train_source import code_identity

root = Path(__file__).resolve().parents[1]
smoke_root = root / 'runs/source_a/local_smoke_v3'
smoke = read_json(smoke_root / 'SMOKE_PASSED.json')
check_equal('final core code matches GPU smoke identity', smoke['identity']['code'], code_identity())
run = smoke_root / 'smoke_batch2'
initial = read_json(run / 'audit_initial/runtime.json')
resumed = read_json(run / 'audit_resume/runtime.json')
progress = read_json(run / 'resume_progress.json')
locks = read_json(root / 'artifacts/source_a/dependency_lock_audit.json')
settings = read_json(root / 'artifacts/source_a/global_settings_audit.json')
package = read_json(root / 'artifacts/source_a/cloud/source_a_kit_v2.tar.gz.json')
qa = read_json(root / 'artifacts/source_a/relocation_qa.json')
check_equal('relocation status', 'passed', qa['status'])
check_equal('relocation archive', package['sha256'], qa['archive_sha256'])
check_equal('READY still immutable', smoke['identity']['ready_sha256'],
            sha256(root/'data/soda10m/splits/source_v1/READY.json'))
xml = ET.parse(root/'reports/source_training_tests.xml').getroot()
cases = list(xml.iter('testcase'))
assert len(cases) == 15 and not list(xml.iter('failure')) and not list(xml.iter('error'))
release_xml = ET.parse(root/'reports/source_release_tests.xml').getroot()
release_cases = list(release_xml.iter('testcase'))
assert len(release_cases) == 3 and not list(release_xml.iter('failure')) and not list(release_xml.iter('error'))
test_total = len(cases) + len(release_cases)
assets = {name: sha256(root/name) for name in [
    'configs/source_a.yaml', 'scripts/train_source.py', 'scripts/source_trainer.py',
    'scripts/source_contract.py', 'scripts/bundle_source.py', 'scripts/verify_source_bundle.py',
    'environment/requirements-windows.lock', 'environment/requirements-linux.lock',
    'weights/yolo11m.pt', 'weights/yolo26n.pt']}
state = {'status': 'READY_FOR_CLOUD', 'preparation_complete': True, 'local_smoke_passed': True,
         'cloud_instance_connected': False, 'linux_executed': False, 'formal_A_completed': False,
         'final_tests_run': False, 'unit_tests_passed': test_total,
         'tests_breakdown': {'training_core': len(cases), 'release_state': len(release_cases)},
         'identity': smoke['identity'],
         'final_core_matches_smoke': True,
         'file_sha256': assets, 'package': package, 'relocation': qa, 'resume': progress,
         'smoke_initial_runtime': initial, 'smoke_resumed_runtime': resumed,
         'smoke_best_sha256': sha256(run/'weights/best.pt'),
         'smoke_last_sha256': sha256(run/'weights/last.pt'),
         'dependency_locks': locks, 'global_settings_side_effect': settings}
write_json(root/'artifacts/source_a/delivery_audit.json', state)
rows = '\n'.join(f'| `{k}` | `{v}` |' for k, v in assets.items())
text = f'''# 白天源模型 A 训练报告

更新于2026-10-25（Asia/Shanghai）。本阶段状态：**READY_FOR_CLOUD**。准备完成、本机 smoke 通过、可迁移训练包验证通过；云端实例尚未接入，Linux 尚未运行，正式 A 未开始，最终测试未运行。没有租赁、付费或创建云资源，没有执行 Git add/commit/push，也未修改 plan.md 或数据会话文件。

## 1. 本阶段产物与边界

- 入口 `scripts/train_source.py` 支持环境检查、权重获取、只读交接验证、本机 smoke、云端预检、正式 A 和续训；配置 `configs/source_a.yaml` 不硬编码运行盘符。
- `scripts/source_contract.py` 负责 READY、数据指纹、归一化、分组协议、六类映射、train/dev清单及全部9000白名单文件哈希验证。消费者不打开 source_test/night_test/白天跨场景最终评价清单、图片或标签。协议和READY里的保留测试计数、哈希属于冻结身份元数据。
- 白名单复制到 runs 中独立沙箱，开发集仅验证/选模，不用于梯度训练；Ultralytics标签缓存只写沙箱。冻结 source_v1 未覆盖、未修改，交付时READY字节再次核验一致。
- `scripts/source_trainer.py` 以真实 class_id=[0,2] 的 AP50:95 宏平均作为 checkpoint/early stopping fitness；不使用默认六类平均值，不使用缺失类的均值填充。六类指标独立保留；主类缺失时报错，辅助类缺失记null。并列保留早期best且不重置patience=20，保留last/best与每5轮可续训checkpoint。
- 保存完整默认值、解析配置、传入参数、trainer最终参数、真实SGD参数组/累积步数/AMP、逐轮主指标和六类指标、性能记录、checkpoint身份和SHA。续训检查数据/配置/代码/原始权重/batch/scope及checkpoint字节，恢复epoch、optimizer、scaler、EMA、scheduler进度和早停状态。

## 2. 数据和权重身份

白天 READY：`data/soda10m/splits/source_v1/READY.json`；SHA256=`{smoke['identity']['ready_sha256']}`；source_train/dev/test=4000/500/500，训练消费只包含前两部分。data_fingerprint=`{smoke['identity']['data_fingerprint']}`；grouping_protocol SHA256=`{smoke['identity']['grouping_protocol_sha256']}`；归一化版本 `{smoke['identity']['normalization_version']}`。这是项目自定义相似场景分组，不宣称官方同车同日sequence身份已经验证。

官方COCO预训练[Ultralytics assets v8.4.0 yolo11m.pt](https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo11m.pt)独立下载，40684120字节；SHA256=`{assets['weights/yolo11m.pt']}`。来源和下载时间见 `weights/yolo11m.provenance.json`。另携带 `yolo26n.pt` 仅用于锁定版本内部AMP数值检查，其来源/哈希见独立provenance；不作为A初始化。正式A必须从原始yolo11m重新初始化，不能用smoke或preflight权重。

## 3. 环境和依赖锁

独立 `.venv-train`：Python3.12.14、torch2.14.0+cu130、torchvision0.29.0+cu130、Ultralytics8.4.171。`pyvenv.cfg` 为 include-system-site-packages=false；以已有Python3.12解释器引导venv，未向学习环境或.venv-data安装包。CUDA前反向与torchvision NMS、pip check通过。

Windows带wheel哈希锁47包与实际安装逐包匹配；Linux x86_64/Python3.12带哈希锁65包，其中46个共同包与Windows实装版本相同。Linux新增CUDA依赖仅解析；**尚未安装、导入或运行Linux训练**。实装快照 `environment/windows-installed.txt`，环境证据 `artifacts/source_a/environment/environment.json` 和 smoke目录environment，锁一致性 `artifacts/source_a/dependency_lock_audit.json`。

首次torch下载因C盘TEMP空间不足失败，改为D盘任务专属临时目录后成功；没有清理用户文件。所有后续训练进程和setup脚本显式设置TEMP/TMP、pip、Ultralytics、Torch和Matplotlib缓存到任务 `.cache/training`。

## 4. 本机 smoke 与测试

最终证据 `runs/source_a/local_smoke_v3/SMOKE_PASSED.json`，控制台 `artifacts/source_a/smoke-v3-console.log`。v1/v2为先前实现迭代的通过记录，最终包绑定v3的最终代码/配置身份；不会用任一smoke权重启动A。

RTX4060 Laptop 8GB，YOLO11m/640、AMP=True、batch2、workers0，冻结训练/开发分别确定性抽64/32张，两部分均覆盖六类且各保留1张空标注。总共2轮：第1轮保存后停止；以同一2轮计划恢复第2轮；恢复记录checkpoint epoch0 -> start_epoch1，optimizer/scaler已恢复，早停最佳epoch与分数状态保留。每轮32批，真实优化器步数分别{initial['optimizer_steps']}/{resumed['optimizer_steps']}；有限损失、前反向、开发评价、best保存/重载均通过。未触发OOM降batch。

本机smoke训练循环耗时分别{initial['elapsed_seconds']:.3f}/{resumed['elapsed_seconds']:.3f}秒（包含逐轮开发评价与保存，不含启动和最后额外best评价）；峰值allocated={max(initial['peak_allocated_bytes'], resumed['peak_allocated_bytes'])}字节，reserved={max(initial['peak_reserved_bytes'], resumed['peak_reserved_bytes'])}字节。这些只描述小子集本机验证，不能外推为正式A吞吐、费用或性能。

smoke best SHA256=`{state['smoke_best_sha256']}`；last SHA256=`{state['smoke_last_sha256']}`。原始AP日志仅用于验证链路，**不作为正式A性能结果**。

训练核心测试15项通过，XML=`reports/source_training_tests.xml`。覆盖中文路径、路径穿越、禁止最终测试用途、READY/文件损坏错误、真实AP类别索引与缺主类拒绝、六类覆盖/空标注/确定性、零分并列耐心、best并列保留、5轮checkpoint/身份保存、冻结配方哈希和非法batch拒绝。另有发布状态3项回归通过，XML=`reports/source_release_tests.xml`：重定位失败不得晋升READY、归档SHA错误不得晋升、成功后原子发布真实归档SHA；两组共18项。包内携带15项训练核心测试；本地发布回归验证本地包发布门禁。数据测试入口无需torch；训练测试位于独立training_tests。

## 5. 迁移包与重定位

归档 `artifacts/source_a/cloud/source_a_kit_v2.tar.gz`，{package['bytes']}字节，SHA256=`{package['sha256']}`。共{package['archive_members']}个文件成员，精确9000个train/dev图和标签，外加必要代码、配置、环境锁、原始预训练/AMP检查权重及本机证据。没有source_test、夜间或白天跨场景最终评价数据文件与用途清单；不打包整个data目录。

`scripts/bundle_source.py` 只按冻结files白名单加必要元数据选取；`PACKAGE_MANIFEST.json`记录包内全部SHA。已在本机解压到独立中文路径 `{qa['destination']}` 并从新目录执行纯stdlib验证器，包内全部{package['archive_members'] - 1}个受清单管理文件及9000数据文件哈希重新验证，READY与数据指纹通过，无额外或保留测试数据文件。证据 `artifacts/source_a/relocation_qa.json`、`artifacts/source_a/package-relocation-v2.log`。这是Windows本机重定位验证，**不等同于Linux或云端执行通过**。

包文件清单 `artifacts/source_a/cloud/source_a_package_manifest.json`；整体交付审计 `artifacts/source_a/delivery_audit.json`。迁移包仅供本项目私有使用，仍遵守数据许可，不上传公共仓库。

首次v1重定位QA的启动器错误地在切换cwd后仍传相对脚本路径，子进程在开始验证前退出2；失败日志保留 `artifacts/source_a/package-relocation.log` 和旧解压目录。修复为绝对脚本/root路径并在失败时保留stderr。v1的sidecar曾过早标READY，现已撤回为SUPERSEDED_NOT_RELEASED，最终交付只使用v2。v2构建时为PACKAGED_PENDING_VALIDATION，只有解压后全部验证成功，外部sidecar才原子升级READY_FOR_CLOUD；包内manifest保留构建时待验快照，最终发布依据外部sidecar与relocation_qa，均绑定同一归档SHA。白天source_v1与训练核心代码/配置没有因此改变。

## 6. 云端依赖与下一条命令

完整命令见 [source_a_cloud.md](source_a_cloud.md)。目前缺少平台/实例连接、持久盘路径、真实单价与币种。需单卡RTX4090 24GB、至少8vCPU/32GB/50GB持久盘。Linux依赖、OpenCV共享库、CUDA驱动和真实硬件仍须在实例核验。

接入后先执行包验证和 `bash environment/setup-linux.sh`；然后 `preflight` 用全量4000/500、2轮、batch16，仅OOM可从原始COCO重试batch8，仍失败即停止诊断。成功才输出FROZEN_RECIPE及其SHA；**当前正式配方尚未在云端冻结，其哈希不存在**。正式命令必须显式提供该外部配方SHA，再从原始yolo11m开始100轮上限、固定SGD配方；记录峰值显存、吞吐、耗时，并按真实单价/币种估算和记录账单。禁止把本机batch2配方当正式配方。

正式A最佳权重、正式source_dev指标、云端时长/吞吐/费用均未产生。未启动其余23次实验，最终测试始终留出。

## 7. 首次导入的用户配置副作用

最初参考已有学习环境时，未先隔离Ultralytics用户配置的import触发库自身迁移，**已有全局用户配置发生改写**：`{settings['path']}`。文件创建2026-10-04 17:28:57，修改2026-10-25 19:54:40（Asia/Shanghai），与首次import日志相符。初始日志原文为“Ultralytics settings updated to the latest schema. Existing values were preserved where possible.”；这是库的迁移声明，不能当作已验证所有旧字段保留。

只读现状SHA256=`{settings['current_sha256']}`；当前schema0.0.8。该目录可见文件仅Arial.ttf与settings.json，未找到同目录迁移前备份；没有迁移前字节或哈希，因此**无法验证字段级差异、被新增/丢弃字段或完全无影响**。未自动回滚或覆盖该文件，避免在原内容未知时造成再次改写。学习/数据环境已安装包未被此次导入修改；后续进程已强制D盘任务专属YOLO_CONFIG_DIR。审计 `artifacts/source_a/global_settings_audit.json`，原始会话日志 `C:/Users/胡成凯/.codex/sessions/2026/10/25/rollout-2026-10-25T19-53-49-01a172e8-e9f4-77d1-955b-5fc14b3a3c6d.jsonl`，迁移实现位于锁定版本 `ultralytics/utils/__init__.py` 的SettingsManager._validate_settings。已向总控报告这一已知副作用与未知边界。

## 8. 最终文件哈希

配置语义SHA256（规范JSON）=`{smoke['identity']['config_sha256']}`。下表为实际文件字节SHA256：

| 文件 | SHA256 |
|---|---|
{rows}
'''
(root/'reports/source_a.md').write_text(text, encoding='utf-8')
print(json.dumps({'status': state['status'], 'package': package, 'tests': test_total}, ensure_ascii=False))

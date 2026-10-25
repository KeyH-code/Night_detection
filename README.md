# 夜间道路人车检测适配

在有限目标域训练图片预算下，比较随机选样与一致性、多样性选样，研究夜间检测、误报控制和白天性能保留。完整方案见 [plan.md](plan.md)。

## 当前状态（2026-10-24）

已从官方入口获取 SODA10M 有标注 train/val，完成 10,000 张图片全量解码与文件哈希、78,239 个框的结构检查，并建立独立的数据准备环境。

**正式数据划分尚未冻结**：下载包没有可验证的采集组映射；视觉初审发现部分 Night 标签与画面不符，并存在相似画面。数据准备报告见 [reports/data_preparation.md](reports/data_preparation.md)。当前没有正式训练、测试指标或模型交付。

## 人工抽样复核网页（2026-10-25）

已重新从当前官方入口取得两份标注的归档字节，并抽查 8 张远端原图，与本地完全一致。没有发现当前 trainval 包缺件或本地取错分组的证据。官方 sequence 隔离声明针对原始 train/val/test；公开格式未承诺附带重新划分用的映射。详见 [来源对齐报告](reports/source_reconciliation.md)。

打开 [本机复核台](http://127.0.0.1:8731)：

1. 先查看“来源与划分”。
2. “夜间属性”可先看 24 张分层覆盖样本，再切换 15 张 Agent 标记或其他范围；逐项判断。
3. “相似候选组”默认随机抽 12 组，也可浏览全部 320 个组、展开配对距离。
4. 点击“导出审核记录”保存 JSON。判断只保存在当前浏览器；换浏览器不会自动同步。

服务已在本机启动。需要重新启动时，先确认端口未被现有复核服务占用，再运行：

```powershell
& 'D:\Night_detection\.venv-data\Scripts\python.exe' 'D:\Night_detection\scripts\screen_similar_scenes.py'
& 'D:\Night_detection\.venv-data\Scripts\python.exe' 'D:\Night_detection\scripts\serve_data_review.py'
```

当前相似场景复核入口为 `http://127.0.0.1:8731/?view=groups&mode=sample`：固定 50 个新组，上轮 12 组只读保留，官方时段只读。重建抽样依赖本项目已导入的私有历史审核解释，并核验指纹；见 `reports/scene_screening_2026-10-25.md`。查看后请导出 JSON；网页不删图、不改 period、不创建正式划分。

只监听 `127.0.0.1:8731`；不提供目录浏览或原始文件改写接口。没有发布到外部托管。网页读取原始 COCO 标签；审核结果不自动转为训练标签。

## 本地目录

| 目录 | 用途 | Git |
|---|---|---|
| `scripts/` | 获取、审计、归一化和验证入口 | 保留 |
| `configs/` | 类别对应关系 | 保留 |
| `tests/` | 数据工具必要检查 | 保留 |
| `reports/` | 聚合审计结果和限制 | 保留，发布前复核许可 |
| `data/soda10m/archives/` | 官方原始压缩包 | 忽略 |
| `data/soda10m/raw/` | 原始图片与原始 COCO 标注 | 忽略，保持不变 |
| `data/soda10m/normalized/` | 仅裁剪越界框的派生标注 | 忽略 |
| `data/soda10m/sources/` | 来源网页快照、下载和解压记录 | 忽略 |
| `data/soda10m/audit/` | 图片级清单、GT 统计、问题与复核记录 | 忽略，选样程序禁止使用 |
| `artifacts/data_review/` | 本地图片和标注复核预览 | 忽略 |
| `.venv-data/` | 数据准备环境，不含训练依赖 | 忽略 |

## 数据准备命令

在 Windows PowerShell 中先确认目录并设置 UTF-8：

```powershell
Set-Location -LiteralPath 'D:\Night_detection'
Get-Location
[Console]::InputEncoding = [System.Text.UTF8Encoding]::new($false)
[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new($false)
$OutputEncoding = [System.Text.UTF8Encoding]::new($false)
$env:PYTHONUTF8 = '1'
$env:PYTHONIOENCODING = 'utf-8:surrogateescape'
```

独立数据环境使用 Python 3.13.5，依赖锁定在 `requirements-data.txt`。已建立环境无需重复创建；新工作目录可按下列方式建立：

```powershell
python -m venv .venv-data
& '.\.venv-data\Scripts\python.exe' -m pip --cache-dir '.cache/pip' install -r requirements-data.txt
```

首次获取和解压：

```powershell
& '.\.venv-data\Scripts\python.exe' scripts/prepare_soda10m.py download
& '.\.venv-data\Scripts\python.exe' scripts/prepare_soda10m.py extract
```

`download` 会校验已有完整包及来源快照，部分下载仅在服务端支持 Range 时续传。`extract` 拒绝路径穿越、链接、越界范围及覆盖已有原始文件；本机已经解压，无需再次运行它。

重建审计、派生标注与复核候选，然后验证：

```powershell
& '.\.venv-data\Scripts\python.exe' scripts/audit_soda10m.py --workers 4
& '.\.venv-data\Scripts\python.exe' scripts/review_soda10m.py
& '.\.venv-data\Scripts\python.exe' scripts/verify_soda10m.py
& '.\.venv-data\Scripts\python.exe' -m unittest discover -s tests -v
& '.\.venv-data\Scripts\python.exe' -m pip check
```

所有入口接受 `--data-root`，默认 `data/soda10m`；相对路径以工作目录为基准。重跑生成工具后，原视觉复核记录需要检查是否仍适用。`audit` 输出包含 GT 派生信息，不是未来候选池或选样入口。

## 数据使用边界

来源：[SODA10M 官方下载](https://soda-2d.github.io/download.html)、[数据格式](https://soda-2d.github.io/documentation.html)、[使用条款](https://soda-2d.github.io/about.html)。本地用途为非商业研究。原始数据、修改版本和预览不进入 Git；统计、模型和展示材料在发布前分别核验条件。本仓库尚未提交或推送。

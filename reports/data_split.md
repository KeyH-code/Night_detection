# 自定义相似场景分组与数据冻结报告

日期：2026-10-25。已完成数据划分与冻结；没有执行任何模型训练或最终测试。训练状态独立见 `reports/source_a.md`。

本次采用用户批准的自定义相似场景协议，替代等待权威 sequence 映射的历史门禁。它控制已识别相关场景跨用途的风险，不证明同车同日、采集序列或全量语义独立。官方 period 直接沿用，不更改时段、不删图、不机械每组留一张。

## 数量、行人与空标签

| 用途 | 图片 | 风险组(含单图) | 含行人图片 | 空标签图片 | 框数 |
|---|---:|---:|---:|---:|---:|
| source_dev | 500 | 466 | 210 | 3 | 4171 |
| source_test | 500 | 466 | 210 | 2 | 4028 |
| source_train | 4000 | 3726 | 1677 | 28 | 32911 |
| daytime_cross_scene_test | 3344 | 3076 | 603 | 6 | 26894 |
| night_dev | 331 | 316 | 46 | 2 | 2033 |
| night_pool | 994 | 948 | 139 | 9 | 6140 |
| night_test | 331 | 316 | 46 | 1 | 2062 |

三用途白天正好 4000/500/500，夜间正好 994/331/331，均无目标偏差。白天跨场景 3344 张完整保留；总量 10000 张、78239 框、51 张空标注。夜间开发和测试各 46 张含行人图，满足各至少 40 张。

## 场景属性与类别分布

source_train/source_dev/source_test 全部 Shanghai / Citystreet / Clear，来源保持官方 train 的集中域。下列各计数顺序对应同列标题。

| 用途 | 城市 GZ/SH/SZ | 道路 Citystreet/Countryroad/Highway | 天气 Clear/Overcast/Rainy |
|---|---|---|---|
| daytime_cross_scene_test | 804 / 1792 / 748 | 1157 / 303 / 1884 | 1407 / 1354 / 583 |
| night_dev | 33 / 161 / 137 | 85 / 74 / 172 | 123 / 195 / 13 |
| night_pool | 99 / 482 / 413 | 255 / 223 / 516 | 370 / 584 / 40 |
| night_test | 33 / 161 / 137 | 85 / 74 / 172 | 123 / 195 / 13 |

| 用途 | pedestrian | cyclist | car | truck | tram | tricycle |
|---|---:|---:|---:|---:|---:|---:|
| source_dev | 484 | 710 | 2366 | 422 | 167 | 22 |
| source_test | 440 | 684 | 2279 | 429 | 175 | 21 |
| source_train | 3977 | 5154 | 18811 | 3446 | 1339 | 184 |
| daytime_cross_scene_test | 2128 | 2055 | 13772 | 7857 | 989 | 93 |
| night_dev | 144 | 103 | 1372 | 332 | 81 | 1 |
| night_pool | 425 | 360 | 4109 | 1004 | 239 | 3 |
| night_test | 140 | 106 | 1395 | 339 | 82 | 0 |

night_test 没有 tricycle 框；夜间总共仅 4 个此类框。协议不强求罕见辅助类在每部分出现，不能据此给出稳定的夜间 tricycle 结论。场景、天气与目标构成共同变化，后续昼夜差异不能全部归因于光照。

## 分组与确定性

- 原 320 个严格候选组，解除第一轮人工仅场景接近的 6 组；第二轮明确否定 probe-e9121c8945b0（宽松误报）不采纳。宽松补查仅加入人工确认的 5 组，最终保守约束为 314 严格组 + 5 宽松确认组。没有把未审核候选当作人工通过。
- 采用传递闭包形成风险组，组内不同画面全部保留；跨官方来源/时段的采纳关系会明确报错，不能静默拆分。本次采纳关系均在相应来源/时段池内。
- seed=20261025；大组优先，固定随机打破平局，贪心先优化数量，再优化含行人比例、city/location/weather 与其他类别图片覆盖。随后最多 8 轮、每轮 6000 个确定性同规模整组交换建议，按相同字典序只接受改进。数量容差上限 5，夜间行人覆盖为硬条件，未满足则失败。
- 完整输入哈希、人工原件绑定、解除和确认的组 ID、所采纳组成员、算法与版本在每版 protocol.json。原始图片、JSON、审核原件保持不变。

## 冻结身份与消费接口

| 版本 | READY SHA-256 |
|---|---|
| source_v1 | `72e75ebd99014c830e0227061e390e6aed768ae724b532a5c2c90fd8dec93e85` |
| night_v1 | `fc9e71fdfd7384e54ce33811844fef7f6e364a44c2d5c97230acfad199fabd5d` |

共同数据指纹：`7a275a3ec80a7628ffea31c05093c4960ba453dad94156f70f2d3d8871c36f54`。

共同 protocol.json SHA-256：`1c0d8cae665dc3023fbd48ec32879da8eaa36f55b54714a34d397f9ef85940e3`。

归一化版本：`clip_xyxy_to_image_bounds_v1`。数据指纹是新规范 JSON 摘要，完整算法与组成在协议中；历史审核 f210f1… 指纹保留为对应审核工具的来源身份，不与新算法混称。

- source_v1 唯一交接入口是 READY.json。全量派生与验证成功后原子发布，发布后禁止覆盖。构建目录独占 mkdir 防止同版本并发发布；失败不会发布 READY 或自动覆盖残留目录。
- `files` 恰好 9000 条图片/标签记录，仅覆盖 source_train/source_dev；source_test 的 500 张图与标签和清单独立留存，全部不在白名单。训练侧须核对外部 READY 哈希和每文件哈希，再复制到运行沙箱，避免框架缓存污染冻结目录。详见 `reports/source_ready_contract.md`。
- 全部路径相对对应版本根，支持 Windows/Linux 重定位。原图目录没有写 YOLO 标签；派生采用独立复制，避免对原图硬链接写入的耦合。
- night_pool GT 标签在 `hidden_ground_truth/night_pool/`；`selection/night_pool_images.json` 仅含图片、图片哈希、组 ID，没有类别、框或 GT 路径。完整数据管理清单含 GT 统计，未来选样入口不能读取它或 audit/inventory。当前完成文件接口隔离，未来程序还需实施运行时读路径约束。
- night_v1 独立发布，未更改 source_v1；开发仅验证/选模，最终测试未执行。

## 验证记录

运行环境为现有独立 `.venv-data`，划分脚本和新增测试仅用 Python 标准库，未安装训练框架。

```powershell
.venv-data/Scripts/python.exe -m unittest discover -s tests/data_split -v
.venv-data/Scripts/python.exe -m unittest discover -s tests -p test_data_preparation.py -v
.venv-data/Scripts/python.exe scripts/make_splits.py plan --stage source
.venv-data/Scripts/python.exe scripts/make_splits.py plan --stage night
.venv-data/Scripts/python.exe scripts/make_splits.py generate --stage source
.venv-data/Scripts/python.exe scripts/make_splits.py generate --stage night
.venv-data/Scripts/python.exe scripts/make_splits.py verify --stage source
.venv-data/Scripts/python.exe scripts/make_splits.py verify --stage night
.venv-data/Scripts/python.exe scripts/verify_soda10m.py
```

generate 两命令已经执行成功；冻结版本不能再次执行 generate。plan 和 verify 都只读，不写版本目录。

- 新增 10 项语义测试通过：守恒、组隔离、固定种子、否定关系解除、不可分组失败、跨池失败、夜间行人硬条件、框回算、空标注、中文路径迁移、篡改/测试注入拒绝、路径越界拒绝、独占发布和旧状态推导。既有数据准备 9 项测试通过。
- 发布前两版全量验证通过，验证回执在各版 validation.json。全量检验哈希及框转换是数据检查，不是运行最终模型测试。
- 发布后只读结果写在版本外 `reports/data_split_validation/`，不改冻结目录。最终状态见下方追加结果。
- `verify_soda10m.py` 不再硬编码 formal_split_created=false/final_test_frozen=false；根据两版 READY 的用途清单及其哈希推导当前状态。更新旧审计回执前以内容哈希命名保存 audit/verification_history 历史副本。旧审计/人工报告保留为历史证据。

## 当前边界

本职责的已批准数据范围已落盘。没有启动模型训练、使用最终测试选模、选择100/300张适配样本、租赁付费云资源或执行Git写操作。源模型 A、smoke、训练日志/权重和云端接入由训练会话独立报告；数据冻结不代表 A 训练成功。尚无 sequence 身份证明或全量标签语义准确性证明。

## 发布后最终复核结果

- source_v1 与 night_v1 的独立只读全量验证均 passed；报告分别是 `reports/data_split_validation/source_v1_readonly.json` 和 `night_v1_readonly.json`。
- 原始归档 10002 文件哈希、78239 个归一化框及 933 个裁剪框复核通过；原始标注/类别一致。审计副本在 `raw_archive_and_freeze_state.json`，过程日志在同名 `.log`。
- 当前 `formal_split_created=true`、`final_test_frozen=true`，两版冻结清单/哈希为真值；`final_test_executed=false`。历史 verification 回执保留在 ignored `data/soda10m/audit/verification_history/`。
- 夜间发布和所有复核结束后，两版 READY SHA-256 仍与上表一致；source_v1 没有被修改。当前状态汇总在 `reports/data_split_validation/current_state.json`。

候选池图片选样清单 SHA-256：`04e000c0fb125c2eb7518658a94bbfeaba24cb2885347029af17d637fd7bd4e0`。这是 `night_v1/selection/night_pool_images.json` 的外部固定身份；冻结 READY 中该项只记录相对路径，未来选样入口须同时绑定此哈希并拒绝读取 GT 清单。未回写 READY。

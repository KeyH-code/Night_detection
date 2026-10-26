# 数据

SODA10M官方有标注train+val，共10000张、78239框。官方period直接沿用；六类保留，cyclist不并入pedestrian。当前冻结用途如下。

| 用途 | 图片 | 含行人 | 空标注 |
|---|---:|---:|---:|
| source_dev | 500 | 210 | 3 |
| source_test | 500 | 210 | 2 |
| source_train | 4000 | 1677 | 28 |
| daytime_cross_scene_test | 3344 | 603 | 6 |
| night_dev | 331 | 46 | 2 |
| night_pool | 994 | 139 | 9 |
| night_test | 331 | 46 | 1 |

## 依据

- 来源入口、许可网页及获取记录在`../data/soda10m/sources/`；原始归档、图片和JSON在archives/raw。用于非商业本地研究，原图和派生数据不公开再分发。
- 原始JSON SHA：train `466795449573f3e5f290522c9b0c3de78cd523be1f6ea3331044e6f26eb16be2`；val `b6553208c7b742f0bd841df7e223224e6ee4f7801aa432e1dd1dbe3a580dd4c6`。
- 933个越界框采用clip_xyxy_to_image_bounds_v1裁剪，零丢框/改类/改period；变更明细在audit/box_normalization_changes.json，规范记录在normalization.json。
- 自定义风险组用于防止相关画面跨训练/开发/测试。解除人工仅场景接近的6组及宽松误报；保守纳入314个严格候选组和5个人工确认宽松组。保留全部图片，未机械每组只留一张。
- 组数包含单图组，不能当成独立采集场景数。没有官方同车同日sequence证明；感知哈希/结构检查也不能证明全量语义标注完整。
- seed20261025，整组贪心与局部交换优化数量、含行人比例、city/location/weather及其他类别覆盖；目标数量容差5，本次全部精确满足；夜间开发与测试各46张行人图，超过最低40张。
- train全部Shanghai/Citystreet/Clear。白天source_test检查来源相近场景；val/Daytime的3344张跨场景白天评价检查其他城市、道路或天气，二者均未用于本次A。
- night_test无tricycle框；夜间总共此类仅4框，不能据此给出稳定的罕见类结论。

## 判断

`audit/human_reviews/`只保留直接决定分组的判断原件与核验解释：第一轮的export.original.json/interpretation.json；第二轮的同类两文件及第6组聊天补充chat_supplement.json。这是数据决策证据，不是临时网页或操作说明。
`audit/scene_screening.json`保留机器候选及其来源，与冻结protocol的evidence_sha256相互绑定。新划分入口显式输入已审核协议，不依赖网页服务或代码中硬编码审核日期。

## 身份

- `source_v1` READY SHA-256：`72e75ebd99014c830e0227061e390e6aed768ae724b532a5c2c90fd8dec93e85`。
- `night_v1` READY SHA-256：`fc9e71fdfd7384e54ce33811844fef7f6e364a44c2d5c97230acfad199fabd5d`。
- 数据指纹：`7a275a3ec80a7628ffea31c05093c4960ba453dad94156f70f2d3d8871c36f54`。
- protocol SHA-256：`1c0d8cae665dc3023fbd48ec32879da8eaa36f55b54714a34d397f9ef85940e3`。
- 归一化版本、原始/归一化JSON哈希、候选与人工判断哈希均在各版protocol.json。原始哈希记录原样留存，不因整理目录重排JSON。

## 消费

白天入口`data/soda10m/splits/source_v1/READY.json`：schema_version=1、status=frozen；data_fingerprint、normalization_version、grouping_protocol、categories、category_mapping、counts、manifests、files和held_out共同定义身份。
清单是`{purpose,records:[...]}`，image/label均为相对版本根的POSIX路径；禁止绝对路径或..。files仅9000条train/dev图片和标签，test清单在held_out且不加载。训练侧先核验外部READY SHA与每文件哈希，再复制到运行沙箱，禁止框架缓存写回冻结目录。
YOLO标签为训练ID+归一化cx/cy/w/h，十二位小数；可回算误差<1e-6像素。空图保留，标签零字节。READY在全量派生与验证之后最后原子发布，现有版本不可覆盖。
夜间GT位于`night_v1/hidden_ground_truth/night_pool/`；未来选样只读`selection/night_pool_images.json`及预测，不能读完整管理清单、GT或audit。图片选样清单SHA为`04e000c0fb125c2eb7518658a94bbfeaba24cb2885347029af17d637fd7bd4e0`，READY仅记录其相对路径，消费者须绑定此外部哈希。

## 核验

首次全量解码/归档10002文件哈希/78239归一化框与933裁剪框通过；发布前和发布后源/夜两版校验通过。数据身份原件在audit，版本内validation.json是冻结时回执，保留不改。
只读检查：`python scripts/make_splits.py verify --stage source`及`--stage night`；完整原始归档/归一化检查：`python scripts/verify_soda10m.py`，其回执写audit且历史有内容哈希副本。
结构审计或归一化生成入口在已有冻结根下拒绝覆盖。新版本生成需显式`--protocol <已审核协议>`；不将历史人工抽样当作全量审核。

## 留存

2026-10-26按用户批准的瘦身原则删除临时网页、截图、测试导出和重复报告。上述判断原件、源数据、归一化、清单、协议和冻结目录保持原字节。整理没有重做划分，也没有训练或运行最终测试。

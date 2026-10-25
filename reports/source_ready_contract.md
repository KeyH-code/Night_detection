# 白天数据消费契约 v1

日期：2026-10-25。由数据与训练会话通过消息确认，入口为 `data/soda10m/splits/source_v1/READY.json`。

`schema_version=1`、`version=source_v1`、`status=frozen`。消费者必须先核对交接消息提供的 READY SHA-256；消息不能代替文件验证。所有 path 均为相对 source_v1 根的 POSIX 路径，禁止绝对路径、反斜杠、冒号、`..` 及越界符号链接。

| 字段 | 定义 |
|---|---|
| data_fingerprint | SHA-256 字符串，规范 JSON 的组成见 protocol.json/fingerprint_components |
| normalization_version | clip_xyxy_to_image_bounds_v1 |
| grouping_protocol | `{path,sha256}`，分组协议及证据哈希 |
| categories | `{path,sha256,names}`，六类快照 |
| category_mapping | 官方 ID/名称、训练 ID/名称快照，训练 ID 顺序 0..5 |
| counts | source_train/source_dev/source_test 实际数量 |
| manifests | 仅 source_train/source_dev，各 `{path,sha256,count}` |
| files | 仅 train/dev 的 `{path,sha256,purpose,kind}` 白名单，kind 为 image/label |
| held_out | source_test 独立清单的 `{path,sha256,count}`；训练不解析其记录或加载其图片/标签 |

清单采用 UTF-8 JSON 对象 `{purpose,records:[...]}`。每条记录包括 `key,image,label,classes,group_id,width,height,image_sha256,label_sha256,annotation_sha256,raw_annotation_sha256,normalized_annotation_sha256,purpose,protocol_version` 及属性和框统计。`classes` 是该图实际存在的训练类别 ID；空标注为 `[]`，对应 txt 是零字节。标签每行是 `training_id cx cy width height`，后四项相对原图宽高归一化，固定十二位小数，可回算到归一化 COCO 框，最大绝对误差小于 1e-6 像素。

数据指纹为 `sha256(json.dumps(components,ensure_ascii=False,sort_keys=True,separators=(',',':')).encode('utf-8'))`。components 包含 inventory_sha256、integrity_sha256、两份 raw_annotation_sha256 和两份 normalized_annotation_sha256；在协议内完整保存，无须训练访问原始文件。协议、类别及清单的文件哈希包含末尾换行。

训练先验证 READY、协议、类别、两清单及每个白名单文件。仅把已验证 train/dev 图片、标签和必要元数据复制到运行沙箱；Ultralytics 的 cache、配置及训练产物写在沙箱。不得把 source_test 或未列白名单文件纳入训练包。消费者在 Windows/Linux 重新生成绝对数据路径，不依赖本机盘符。开发只验证/选模，不参与梯度训练。

READY 必须在全量派生和验证通过之后最后原子重命名发布。版本目录的独占 mkdir 防止同时发布，残留 BUILDING/FAILED 不能自动恢复覆盖；冻结版本禁止重写。若冻结内容确需修正，应报告并建立新版本，不偷偷修补 source_v1。夜间由独立 night_v1 管理，不能改变此版本。

夜间候选池的 `selection/night_pool_images.json` 仅列图片、图片哈希与组 ID；真实标签在 `hidden_ground_truth/night_pool/`，完整数据管理清单含 GT 统计，未来选样程序不得消费完整清单或历史 audit/inventory。这是当前文件接口隔离；未来选样入口还需实施运行时读路径约束。

night_v1 候选池图片清单外部 SHA-256：`04e000c0fb125c2eb7518658a94bbfeaba24cb2885347029af17d637fd7bd4e0`。READY 的 selection_manifest 是路径字符串，未内嵌该哈希；后续选样须绑定此外部身份，不能仅信任路径。

# 结构

## 目录

| 位置 | 用途 | 允许内容 | 命名 |
|---|---|---|---|
| 根目录 | 项目入口 | README.md、AGENTS.md、STRUCTURE.md、RULES.md、GPU.md、plan.md、Git规则 | 固定短名 |
| specs/ | 阶段文档 | 每个计划阶段单独建子目录；根目录不散放阶段文件 | 小写阶段ID，如s0/、s2/、s10/；与根plan阶段表对应 |
| specs/s0/、specs/s1/ | 已完成阶段索引 | review.md链接已有数据/首训证据，注明事后整理 | 固定短名review.md；不补造同期plan或复制原始记录 |
| specs/s2/—specs/s10/ | 各阶段开发安排 | plan.md；有独立用途时增加roadmap.md、changemap.md等，职责见下表 | 固定短名；阶段进度看[根计划](plan.md#阶段)，不机械创建空文档 |
| scripts/ | 活动实现 | 数据、训练、校验、打包入口及共用模块；`gpu.py` 是本机 BitaHub 运维入口，`gpu_queue.py` 接续平台排队和已批准的 scale_003 训练，不随训练包部署 | 小写；必要时下划线；不加临时时间或实例端口 |
| tests/ | 本机运维回归 | `test_gpu.py`、`test_gpu_queue.py` 覆盖开关机、预算、资源身份、单次排队与产物校验；不随训练包部署 | test_<功能>.py |
| tests/data/ | 数据回归 | 归档安全、框转换、分组、冻结身份 | test_<功能>.py |
| tests/train/ | 训练回归 | 白名单、选模、续训、资源、运行和发布边界 | test_<功能>.py |
| tests/train/diagnose/ | 本机诊断和复核器回归 | 原生匹配/导出、工作点、尺寸、风险组区间、审核及固定 R 复核器的裁剪、分组和选轮 | test_<功能>.py；完整 pytest 递归覆盖，训练包顶层 glob 不带入本机回归 |
| configs/ | 配置 | 类别映射、实验参数、README.md参数说明；`gpu.local.json` 是本机实例、账号和 SSH 绑定，不入 Git 或训练包 | 按功能/实验命名；JSON/YAML；本机绑定用 `.local.json` |
| environment/ | 环境 | 数据依赖、平台锁、安装入口 | 按平台/用途；旧有效锁名保留 |
| data/soda10m/ | 数据 | 下表限定内容 | 数据集/版本/用途分层 |
| runs/a_001/ | 首训A基线 | 一份原始训练记录、代码/配置/环境快照、权重及清单；不等于最终A已冻结 | 实验名_序号；如 a_001 |
| runs/a_001_train/ | 固定best的训练集评价 | metrics.json、comparison.csv、run.json、必要日志及评价代码快照；不复制数据/权重 | 关联实验名_用途；原训练清单不改 |
| `runs/<实验>/`、`runs/<实验>_diag/` | 候选、诊断、对照及冻结证据 | 实际执行后保存配置/代码/环境身份、必要日志、指标、权重及冻结记录；固定权重诊断保存可复查预测/匹配/PR，不复制原权重 | 训练用a_002、d_001等短ID；诊断用`<实验>_diag`；seed写身份/配置；不预建空目录 |
| reports/ | 结论 | data.md数据报告、a.md源模型报告、night.md夜间基线及后续夜间结果；形成有效结果后维护对应主题报告 | 按主题/实验；不留过程流水 |
| plots/a_001/ | 图表 | 正式日志生成的分项损失、逐类AP等分析图 | loss.png、ap.png；不放页面截图 |
| plots/a_001_train/ | 对照图 | 固定best的train与已有dev逐类AP对照 | ap.png；由comparison.csv复建 |
| `plots/<实验或诊断ID>/` | 可复建结果图 | 实际产生的PR、错误、尺寸及配对比较图；引用run中的原始数值 | pr.png、errors.png等短名；不放审核网页截图 |
| weights/ | 初始化 | 官方预训练权重及来源收据 | 模型名.pt、模型名.provenance.json |
| .venv-data/、.venv-train/ | 本机环境 | 已安装依赖 | 不入Git，不当交付证据 |

## 入口

| 文件 | 唯一职责 |
|---|---|
| README.md | 当前状态摘要、数据/权重/报告等导航 |
| AGENTS.md | Agent行为、留存、编码及执行边界 |
| [GPU.md](GPU.md) | GPU 使用原则与 BitaHub 开发机操作入口；长期维护，具体参数和实验安排归当前任务 |
| STRUCTURE.md | 维护目录的职责、允许内容与命名模式 |
| RULES.md | 实验指标、保护、证据、选模、停止及冻结标准和依据；长期留存，不是临时审核材料 |
| plan.md | 全项目目标、范围、阶段状态、衔接、文档导航和实验矩阵 |
| specs/s<编号>/plan.md | 该阶段目标、输入、实施步骤、文件范围、验证、产物、失败分支和审核状态；引用RULES |
| specs/s<编号>/roadmap.md | 可选；复杂阶段内部的里程碑、顺序和依赖，不复制根plan路线或plan任务正文 |
| specs/s<编号>/changemap.md | 可选；拟改文件/接口、变更原因、引用影响和验证对应关系；简单改动合入阶段plan |
| specs/s3/rerank.md | 固定 R 的行人复核器实验方案；样本、模型、同池对照和继续条件 |
| specs/s3/scale.md | 行人分辨率专项路线、训练尺度对照与执行准备 |
| specs/s3/tricycle.md | 三轮车专项方案、预算、验证顺序和锁定五类的保护要求 |
| specs/s3/replica.md | 原 A 的 seed 29/43 配方、云自动执行计划、预算与验证边界 |
| specs/s5/select.md | E1/E2 仅图选样的算法、设计依据、隔离门及实施验证，方案状态见文档 |
| specs/s<编号>/review.md | 已完成阶段的证据索引、退出结论及必要限制；无同期计划时明确事后整理，不替代reports或runs |

后续实验/图目录行描述生成模式，不表示这些目录或结果已经存在。只有实际形成可复查证据时创建；本次规则与规划不预建实验目录或空报告。

`plan.md`维护全项目路线和阶段衔接；`specs/`按阶段组织具体安排；`RULES.md`定义验收标准；`reports/`记录执行后的有效结论。阶段文档不兼作启动prompt、操作流水或原始实验存储。每个职责只有一个主要位置，使用链接衔接，不通过复制正文凑齐文件。

S0—S10各有目录。S0/S1为事后证据索引；S3 原采样配方保留在[阶段计划](specs/s3/plan.md)，按优先级推进的行人错误排查归[路线图](specs/s3/roadmap.md)。现行状态以[根计划阶段表](plan.md#阶段)为准；从 [S4 冻结决定](specs/s4/plan.md)查共同 A，从 [S5 计划](specs/s5/plan.md)查当前夜间准备。后续阶段按实际用途细化，不创建空模板。

## 数据

| 子目录 | 内容 | 边界 |
|---|---|---|
| archives/ | 官方原始归档 | 原始输入，不是临时迁移包 |
| raw/ | 官方图片与JSON | 不改原图/原标注；原文件名保留 |
| normalized/ | 裁剪越界框后的JSON | 同一归一化版本训练/评价共用 |
| sources/ | 来源、许可快照与获取记录 | 不留测速、二次下载片段或网页目录探针 |
| audit/ | 原图哈希、标注检查、归一化与分组证据 | 只保留可复查数据；human_reviews是影响分组的判断原件，不是网页材料 |
| splits/source_v1/ | 4000训练/500开发/500留出 | READY、清单、协议、图/标签不可改 |
| splits/night_v1/ | 994候选/331开发/331测试及3344跨场景白天 | 候选GT在hidden_ground_truth；selection仅图片 |

## 实验

`runs/a_001/manifest.json` 是首次正式训练的留存清单，逐项记录相对路径、哈希、大小及旧来源路径。它只声明实际保留内容，不声称原60份文件全部还在；当前冻结 A 的记录见 [S4 冻结决定](specs/s4/plan.md)。

| 子目录 | 内容 | 命名 |
|---|---|---|
| train/ | 原始CSV、逐轮六类指标、最终开发指标、运行身份、实际参数 | 保留原始文件名与字节 |
| job/ | 完成审计、退出结果、资源门禁与完整训练日志 | 保留所需原件；无PID/进度轮询快照 |
| preflight/ | 正式启动所依赖的冻结配方与通过收据 | 保留原始名称与字节 |
| code/ | 正式A当时的训练核心与门禁代码 | 相对项目路径；只作不可改身份快照 |
| environment/ | 当次真实环境、依赖锁及安装一致性 | 原始记录；不留缓存wheel |
| weights/ | best.pt、last.pt及原始sidecar/下载收据 | 不重复放到根weights；不留中间调试权重 |
| review.md | 总控独立验收结论 | 仅与正式A相关的验收事实 |

历史文件中的旧盘符、云端路径或artifacts路径是当次原始记录，不据此恢复已删除目录；当前文件映射在manifest中。活动代码后续修改不改变本次代码快照。

## 命名

`scripts/diagnose.py`是固定权重source_train/dev诊断入口，复用evaluate.py及source_contract.py，不调用Trainer；`scripts/detection_metrics.py`仅做离线匹配、PR/工作点、错误/切片和风险组统计。RULES门槛运行时读取，不在新配置另设一套标准。

`scripts/rerank.py` 是固定 R 行人复核器的本机预检、训练和同池评估入口。`configs/rerank.json` 只保留纳入排除的训练侧疑点和审核来源。实际执行使用新 `runs/rerank_<编号>/`，如 `runs/rerank_001/`。`run.json` 记录身份、分组和逐轮验证，`best.pt`、`last.pt` 保存复核器权重，`source_dev/` 保存 `scores.jsonl.gz`、`pr.npz` 和 `metrics.json`，`manifest.json` 与 `code/` 保留产物核验和实际运行代码。原始图像与 R 预测只引用源路径，预检无持久输出。

固定诊断按runs/<实验>_diag/组织：根目录run.json/protocol.json/manifest.json绑定身份，train/和dev/保存objects.jsonl.gz、matches.jsonl.gz、pr.npz、metrics.json及原生指标native.json；根目录workpoints.json、comparison.csv、slices.csv、intervals.json、errors.json和quality.json记录离线数值与规则缺口，trace.jsonl.gz保留必要候选链与全扫描摘要。objects用稀疏正IoU边保存原生输入坐标下的数值，保护离线匹配边界。

cases.jsonl是逐项判断的主位置：Agent初审放initial_review，用户复审使用独立status/decision/reviewer；备注独立留存，未选结论保持pending，先前不同措辞以note_history保留；review.json汇总真实覆盖和状态，并保存固定人工抽查名单/种子/分层支持量及身份；Agent判断不自动晋升人工通过。scripts/diagnostic_review.py复用抽样与草稿校验、多图模板，网页实例与review_draft.json只放外部任务目录，待审核结束后清理。code/留实际使用的实现与必要锁/测试；采集之后的代码修正仅为变化文件生成code/versions/<SHA前12位>/<项目相对路径>，run的stage_code分别引用分析、追踪、交付实际身份，不回写采集快照。只保留仍被实际阶段引用的版本，不复制权重或图片。plots/<诊断ID>/只保存可复建的pr.png、errors.png、sizes.png；审核网页、contact sheet和密集raw不进入run。

`scripts/evaluate.py`负责单一批准用途的无增强评价；`scripts/train_monitor.py`负责后续训练每轮完整train的独立EMA评价。未来正式实验在原生train/val loss与dev记录之外保留`train_metrics.csv`（每轮六类AP/P/R及同轮dev AP、耗时）和`train_monitor.json`（身份与开销）。这些是可复查的正式观察记录，不用于选模；仅dev决定best/早停。原始A当时没有逐轮train AP，不能事后补造。

- 标题用“规则、结构、数据、训练、环境”等短词，文件名表达功能，不复述整句话。
- 版本仅用于不可覆盖的数据/实验版本；普通说明直接维护同一文件。
- 原图、冻结文件、外部来源收据和正式原始日志不为缩名改写；生成的衍生说明使用短名。
- reports写结论与决策；runs存原始证据；plots存图；不跨目录重复保存同一内容。
- 临时文件使用系统临时目录，不建项目内temp、archive或backup。工具缓存可在运行时短暂生成，任务结束清除，不入Git。

## 整理

2026-10-26按用户批准的边界完成：删除27个一次性脚本、临时审核网页/截图/美化材料、重复迁移包/解压副本、三轮smoke和中间权重、安装/测试缓存及重复报告。删除49923个文件，逻辑大小7153020880字节（约6.66GiB，不含稀疏/硬链接等物理占用差异）。活动脚本16个，报告2份，回归测试按data/train两组组织；不另外保存本次清理脚本或临时日志。

正式A当前保留44个受manifest管理文件及manifest自身，包含原始训练记录/配置/代码/环境快照和最终best/last，不再声称原60份下载证据逐项全部保留。两版冻结目录的20016个文件集合/大小/修改时间未变，READY及逐文件内容哈希与确定性回算校验通过；best/last原SHA保持一致。数据回归22项、训练回归41项及新增可复用打包白名单回归1项通过。训练代码核心3文件及配置保持原身份；runner、打包/数据工具的活动改动不回写历史快照。

整理只检查数据和已有记录；未连接云端、训练或运行最终模型测试，未Git提交/推送。

## 观察

固定best的train评价已留在runs/a_001_train，派生图在plots/a_001_train。新增evaluate.py/train_monitor.py及对应长期测试，分别负责单用途评价和后续每轮完整train观察；原始A的44文件清单不变。活动代码和配置的新观察身份已与原A快照分开，旧A当时没有逐轮train AP/P/R。此类输出按实验/用途分组，不另建过程报告。

## S3

本轮云预检的必要证据归runs/s3_preflight/：manifest.json管理实际留存范围及哈希，r17_1—r17_4记录四次失败的job状态/日志与运行配置/代码身份、已完成的实际采样消费，r17_5另存被用户方案变更中止的控制器快照/停止结果和按校验过的部署材料恢复的代码，云端完整run未取回；amp_probe.json保存AMP数值修正依据，linux_qa保存对应旧实现的云端回归结果；后续r17_<修订号>/、s17_<修订号>/按manifest声明保存失败或通过预检的配置/代码、收据、日志、指标及采样摘要；数值重放只留有效诊断数值，不充作通过收据。linux_qa_<修订号>/保存对应代码的Linux回归。未取回的权重及二进制PR/消费数组不得写成已本地留存。该目录不存图片、缓存或预检梯度权重，不把预检当正式质量结果。正式R/S结果仍归a_002/a_003。

scripts/source_sampling.py维护白名单train的目标层、稳定采样、有限轮workers4加载与实际消费/增强统计；source_policy.py维护RULES解析、资格/诊断角色、horizon状态及配对组重采样。configs/source_s3.yaml独立承载已审两臂配方，source_a.yaml继续保留旧路径。长期回归在tests/train/test_sampling.py、test_policy.py及相关既有测试。

S3实际执行后，run内sampling/index.json定义原图索引，sampling/epochNNN.npz保存实际消费序列、同名json保存来源/增强统计与worker退出；observations/epochNNN.json在当前profile保存完整dev指标、阈值/计数、切片和选模状态，train明确deferred_local，同名npz以分数及十个IoU的累计TP无损保存PR，P/R/FP按记录公式重建。source_metrics.jsonl、dev_metrics.csv为当前逐轮索引；旧profile的train_metrics.csv仅按原运行范围解释，train_monitor.json记录观察开销与累计梯度预算，timings.jsonl保存逐轮分段耗时。

checkpoint_roles.json区分best、diagnostic、last；零合格允许缺best，同一字节以引用去重。checkpoint保存独立训练模型、EMA、优化器、scaler、调度器、剩余累计梯度和RNG，完整轮可恢复。final_eval/云端先写pending_local_evaluation交接，只有权重核验；本地完成后保留原cloud.json交接、选定角色的完整train/dev对象/匹配/PR及last的dev重载，并更新manifest的evaluation_status；manifest.json管理可复查文件的SHA/大小。实际对照目录的comparison.json/csv由source_policy.compare_runs校验完整formal记录后生成，包含固定seed配对差/区间及身份，不预建。code/保留实际实现、配置、已审计划和规则身份。沙箱图片/标签缓存放仓库外，退出后清理；不预建run或复制原A权重。

正式run取回后，根transfer.json记录云端来源、封存manifest/传输包SHA、文件数及下载校验状态，job/保存最终运行/资源日志，environment/保存实际环境。逐个训练文件的身份以train/manifest.json为主，transfer.json不重复维护该清单，也不代表本地完整评价完成。临时压缩包、下载清单和传输脚本放仓库外，用途结束后清理。

a_008的preflight/留存通过的预检配方、收据和训练日志。deployment/留存影响正式启动的共享盘临时目录故障、修正和预算记录。两处均由transfer.json记录下载校验，预检权重和缓存留在正式run之外。

本地收尾前将原云端封存清单原样保存在job/cloud_manifest.json，保留下载身份；train/manifest.json随后管理新增本地评价证据。最后一条run的job/cloud_completion.json记录两臂完成与云端进程/GPU核实，不代替平台关机或账单确认。配对结果使用runs/s3_<seed>/（如s3_17/），保存comparison.json/csv、逐臂intervals.json、采样汇总sampling.json/csv及绑定输入/代码的run.json；它是对照结果，不是新训练run。对应plots/s3_<seed>/的metrics.png、effects.png、sampling.png可用scripts/plot.py --source-pair重建。

配对记录复核继续维护同一目录：transitions.json保存GT迁移及最终框的分数/IoU/匹配证据，同名csv是行人迁移GT的查阅简表；trajectory.csv/json保存原云端逐轮指标、描述性窗口及选模敏感性，不能伪装成逐轮逐GT预测或独立重复。trajectory.png用scripts/plot.py --trajectory重建。图表代码更替时，按实际被图引用的code/versions/<SHA前12位>/保存必要版本，不回写既有图所引用的代码快照。一次性统计程序仍放仓库外；机器证据与视觉/因果结论分开。

errors.json维护同一配对的探索性工作点增量FP、逐IoU匹配/AP、IoU75迁移GT框误差，以及有明确来源/不确定性/原图索引的Agent案例观察；run.json绑定其输入和方法身份。不存图片副本或临时审核载体，原主指标/工作点保留。score.json保存定向R/U候选追踪的样本、权重、复现状态、关键候选分数及NMS保留证据，由同目录run.json收录，不保存密集预测张量。context.json记录固定权重的背景纹理对照、输入相等与扰动范围校验、候选及配对差，不保留处理图或缓存。contour.json记录固定权重的轮廓反差干预、原图最佳候选跟踪、逐像素核验及保持的原工作点，由同目录run.json收录。grid.json记录同位置分数和框的交叉配对、检测头布局校验、IoU分档及目标邻域汇总，索引归run.json，不保留密集输出。

均匀有放回控制复用同一比较目录：compatibility.json留存实际旧/新身份、源码差异、200轮R/S逐项索引核验及本地回归身份，明确cloud_pending或后续状态；不是U训练或预检结果。control.json/csv仅在真实U正式封存及完整本地评价后生成，维护三臂表、S−U/U−R、条件区间及原云轨迹/本地last辅助；不覆盖原comparison。实现入口scripts/source_control.py，精确兼容配置configs/source_s3_control.json，长期回归tests/train/test_control.py；正式U仍按runs/<实验>/既有结构留存，不预建目录。

固定 R 的输入尺度对照归 `runs/s3_17/scale.json`，保存权重及数据身份、三臂输入几何、逐目标覆盖和工作点迁移、六类 AP 及风险组区间。原图、权重和临时缩放图只引用冻结源或放在临时目录。

`runs/s3_17/scale24_trace.json` 保存第 24 轮三模型行人对齐和候选重放证据。逐目标以图像 key 与冻结标签行号索引，记录 960 下的候选 IoU、行人分数、NMS 保留与工作点匹配，并引用 R80 原尺寸对照。仅保留分组结果、必要的候选及抑制证据和输入身份。原封存评价、权重和数据仍在原位置，临时重放脚本和密集张量不留存。

`runs/s3_17/scale24_agreement.json` 保存同一末轮权重的跨尺度候选对照和深低分行人/骑行者分数。保留近阈值分段的 36 个框、全开发集复评计数、110 个目标的代表解码位置及候选分数、判定口径和身份。全量 640 临时输出、重放数据沙箱和探针代码随本轮清理。

`runs/s3_17/scale24_riders.json` 保存冻结训练集的 12 组行人难正例、普通行人与标注骑行者对照。保留选样依据、配对 ID、代表框同位置分数、场景观察和身份校验。开发集只引用已定位的参考案例，原图和临时裁剪视图不复制留存。

`scripts/rider_supervision.py` 复用原生分配与分类损失执行固定分类头对照。`runs/riders_<编号>/` 保存 `data.json` 的训练图与目标审核、实际代码和配置身份、每臂末轮权重和逐轮记录、原生开发评价、配对结果及清单。`train_probe.json` 记录已见监督目标的固定权重同框复核。审查裁图、缓存和联调数据只在临时目录。

负例模式 `--negative` 在同一目录新增 `anchors.json` 保存经训练输入核验的教师位置及张量身份，`dev960/protected.json` 保存冻结解码通道的逐图校验值。`comparison.json` 另存共享阈值与规定精度工作点的骑行者双报统计。

`runs/s3_17/rider_fp.json` 保存分类监督两臂固定阈值的背景与错类框迁移，以原生保留索引报告同一候选的分数和 NMS 抑制链。保留输入身份、逐框判断与局部观察，复用原实验清单，不留复查裁图或密集输出。

`runs/s3_17/rider_negatives.json` 保存冻结训练集的骑行者双报难负例，记录筛查身份、候选级标签和框几何、待复核疑点、Agent 入选理由和入选框的原生保留索引。后续试验方案归 `specs/s3/roadmap.md`，原图、放大裁图和重放临时数据只在原处或临时目录。

`runs/s3_17/scale_preflight.json` 记录尺度微调的本机短预检，包含 R80 初始化身份、样本白名单、实测显存、累积更新、实际学习率和预算外推。正式配方与四格评价见 [scale.md](specs/s3/scale.md)，短预检权重留在临时目录并随退出清理。

`scripts/train_scale.py` 执行固定预算的双臂微调，`scripts/scale_eval.py` 执行四格评价及配对比较。`runs/scale_<编号>/` 例如 `scale_001/` 保留运行身份、实际代码和配方快照、两臂（`640/`、`960/`）的逐轮记录及末轮权重、`dev640/` 和 `dev960/` 的评价证据、`comparison.json` 和封存清单。分开训练与评价时，`training_manifest.json` 用于核验跨机转移的训练产物和训练环境；评价完成后仍保留。运行中进度在 `progress.json`，臂完成后由逐轮正式记录接续。白名单暂存、数据缓存和小样本联调留仓库外并随退出清理。

`scripts/person_fusion.py` 是行人多视图融合的本机白天开发复现入口。无梯度，单方案只用同一 checkpoint；`a1` 保持 `a_001 best` 的 640 非行人输出，`r80` 保持 R80 的 960 非行人输出。`tests/train/diagnose/test_person_fusion.py` 保护合并、类别隔离、上限、镜像坐标和身份语义。`runs/scale_<编号>/` 的固定权重子型只留筛查摘要、最终候选及原生参照的比较、可复查的融合对象及必要身份，不存重复原图、权重或验证器临时目录。

`scripts/tricycle_fusion.py` 用同一检测器复现三轮车视图对照，并在白天开发集只替换三轮车输出；`tests/train/diagnose/test_tricycle_fusion.py` 保护五类输出隔离、融合和输出上限。`runs/tricycle_<编号>/` 留基线诊断、完整筛查摘要、有效候选和锁定对照的比较、必要输出及清单；不存裁图、未入围重排器权重、重复原图或检测器权重。

`runs/seed_<编号>/` 保留多视图跨种子推理的实际权重引用、训练配方身份、各臂逐图融合对象/匹配/PR、条件配对差及产物清单。一轮只读验证不复制权重、原图或中间视图，现有缓存引用原 run，临时编排和沙箱随任务清理。

`scripts/seed_eval.py` 复用锁定行人、三轮车推理入口，对正式源 A 三种子生成上述 `runs/seed_<编号>/` 的原配方复验子型；`tests/train/diagnose/test_seed_eval.py` 保护角色绑定和过线判断。只读准备与现有其他 GPU 工作同时进行，设备繁忙时拒绝启动。

`runs/views_<编号>/` 保留六类视图筛查的配方、原生与锁定对照、各臂结果与已有权重迁移检查；留必要对象和身份，不复制原图、权重或所有中间视图。

`scripts/person_local.py` 在原配方三种子上比较行人局部视图和骑行者竞争。此类 `views_<编号>/` 的 `diagnosis.json` 记录对照漏检分解、误报及阈值上限，`screen.json` 记录各实验臂、切片和逐目标迁移，`comparison.json` 记录统一选臂和风险组配对差。仅入选方案按 seed 保存完整对象、匹配与 PR；原图和中间裁图预测不留存。

`scripts/person_verify.py` 复用同一 `views_` 系列做训练侧几何校正和同检测器逐框复看；`geometry.json` 保存校正参数、内部组拆分和停止判断。`scripts/person_teacher.py` 测量冻结训练白名单上可配对的跨尺度软目标；信号子型 `runs/teacher_<编号>/targets.jsonl.gz` 按训练图和 GT 保存候选，`signal.json` 汇总支持。`scripts/train_teacher.py` 用这些信号训练普通／教师两臂；同系列的训练子型只保留每臂固定末轮权重、实际训练历史、`single640/` 与 `locked/` 的评价、配对比较、清单及代码身份。原图、输入沙箱和未用缓存不留。

`scripts/person_pairs.py` 只读回放等召回对照与训练侧分数匹配对。`teacher_` 的配对子型以 `sample.json` 保存不看后续模型响应的候选与组划分，`review.json` 保存逐对 Agent 原图判断和理由，`comparison.json` 保存核准配对的固定权重响应、开发误报索引及区间。裁图只在仓库外供当轮复核。

`scripts/person_phase.py` 是固定融合框、整图 960 密集查询的两臂推理入口。`views_` 该子型每粒种子的 `supports.jsonl.gz` 按封存的 `seed_002` 行人顺序保留四个查询分，足以重算排序；`screen.json` 与 `comparison.json` 记录各臂资格与配对区间，仅入围方案额外保存完整对象／匹配／PR。

`scripts/source_endpoints.py` 在 `seed_` 系列比较同一轨迹的末轮和等权端点；`scripts/person_views.py` 在 `views_` 系列对照整图横向采样或暗部提亮的独立视图。`<配方>/<seed>/` 保存实际跑过的完整开发对象、匹配、PR、指标与输入身份；五类资格跨种子核验后才标记入选。端点平均失败权重由同种子原件可复建，成功的衍生 checkpoint 归对应 run。

`scripts/person_prior.py` 在 `runs/prior_<编号>/` 保存训练侧行人分类行适配：`train.json` 给出互斥拟合／内部验证组的图像键；`adapter/<seed>/fit.json` 给出内部候选及停止理由，通过才生成 `model.pt` 和完整开发评价。`comparison.json` 记录各阶段实测范围，总清单绑定初始模型、官方预训练权重、数据与代码。

`scripts/source_alias.py --check` 可在本机验证 COCO 别名只改变目标网络的行人行；Linux 单种子正式入口复用原 A 训练器和原配方检查，在实际运行目录封存 `initialization.json`、代码、日志、原始权重及清单。本机 `--check` 不训练、不生成候选权重。

`scripts/alias_queue.py` 只提交一个显式单种子别名对照，读取实时卡价和余额，管理队列、平台截止时间、远端任务、取回和停机。`runs/alias_<编号>.queue.json` 绑定本单的预算、job、包、远端持久路径和实际收尾。取回后的 `alias_<编号>/run/` 存正式训练及预检，`job/` 存命令及原日志；失败仅下载必要故障凭据。`scripts/alias_eval.py` 在停机和下载校验后用 `dev/` 保存同权重九路及 640 对照、匹配和逐类门槛。

`scripts/alias_series.py` 串接已测量时限的单 seed 云任务与关机后的本地复验；发现既有收据不会重复开机。按 29、17、43 顺序，任一粒五类未通过即停止。`alias_001/failure/preflight/` 的实测原件是后续云预算外推来源，不把它标成正式训练结果。

`scripts/source_moments.py` 使用 `prior_` 的 `moments/<seed>/` 子型；仅用固定 256 个训练组的图像校准 BN 缓冲。`train.json` 保存采样身份，`calibration.json` 保存各层观测偏移、实际缓冲变化和参数隔离校验，评价复用分类行适配的同一完整开发协议。

`scripts/person_neighbors.py` 以 `views_` 结构执行原框的同图邻人支持对照。训练侧固定规则筛查失败时，只存 `comparison.json` 与清单，不建立未执行的开发种子结果。

`scripts/person_consistency.py` 的 `views_` 子型用 `fit.json` 保存训练／内部组索引、成员分数和几何特征的校准系数；`scripts/consensus_endpoint.py` 复用其系数与同粒权重原件，验证两个已登记因素的合用结果。两类都按已执行 `<配方>/<seed>/` 保存一次完整开发评价，不保留临时单视图输出和可由原件重建的失败中点权重。

`scripts/alias_mean.py` 在 `seed_` 系列验证同一 seed 原训练与显式别名训练之间的单检查点平均，绑定两个亲本清单，并保存相对两条训练的配对差异。没有对应独立训练原件的 seed 不计入复验。

`scripts/person_appearance.py` 在 `views_` 系列用各 seed 自身骨干训练候选框外观残差，`fit.json` 绑定校准参数、共享的训练组拆分与内部筛查，`metrics.json` 另记附加推理时间。`scripts/train_scene.py` 在 `runs/align_<编号>/` 用全量训练白名单做整景尺度适配，每种子 `train.json` 记录实际消费、轮次和更新数，`last.pt` 为固定末轮 EMA，`dev/` 为重载后完整评价；运行中 `resume.pt` 为完整轮恢复点，正式末轮重载校验后清理。

`scripts/verify_scene.py` 是 `align_001` 的只读复算入口。默认由三粒种子的封存对象、白名单和规则重建验收，`--replay <seed> --scratch <外部目录>` 另外从原图和对应单份权重重放；不修改封存、不保留重复预测。

`runs/align_001/17/freeze.json` 是 S4 确认后单独留存的冻结记录，绑定决定所在的 Git 提交和原实验清单，不回写训练时的 `manifest.json`。仓库内从 `specs/s4/plan.md` 查权重 SHA、适用类别和 S5 交接。

## S5

五类统一评价入口为 `scripts/night_accept.py`，活动协议 `configs/accept.json`，长期回归 `tests/train/diagnose/test_night_accept.py`。`runs/accept_<编号>/` 只保存实际开发复算：`reference.json` 固定 A 的白天分母及阈值，`metrics.json` 留 A 夜间与 D 两域逐类工作点、保护、配对区间和复核项，`protocol.json`、`run.json`、`code/`、`manifest.json` 绑定身份及读取范围。不复制封存预测或原数据；具体读数见[夜间报告](reports/night.md#五类统一口径复评)。

夜间评价身份归 `configs/night.json`；只读消费 `scripts/night_contract.py` 区分 `night_dev` 与候选池仅图沙箱，测试在 `tests/data/test_night.py`。`scripts/night_eval.py` 提供封存白天预测复算、`night_dev` 指定用途核验和正式评价入口，并维护共享阈值、风险组区间、五类输出融合和逐类恢复判定。`scripts/night_run.py` 负责仅开发集的五路采集、复算与封存；回归在 `tests/train/diagnose/test_night_eval.py`。

`runs/night_<编号>/` 留存实际夜间基线：`reference.json` 是固定 A 的白天同口径参照；`objects.jsonl.gz`、`matches.jsonl.gz`、`pr.npz`、`metrics.json` 保存本轮逐图及汇总证据；`run.json`、`protocol.json`、`manifest.json` 绑定用途、实际环境、预检、运行参数及文件清单。`code/` 为运行身份快照，不作活动入口。只引用已存在的权重及白天预测，不复制原图、标签或缓存；结果与边界见 [S5 计划](specs/s5/plan.md)。

`scripts/night_select.py` 是仅图随机名单的活动入口，`configs/select.json` 锁随机协议，`tests/train/diagnose/test_night_select.py` 保护无放回、用途和运行时读取。`runs/pool_001/` 仅为随机对照：`random.json` 保存各 seed 的有序名单及身份，`run.json` 保存读路径、覆盖、复算与环境，`protocol.json`、`code/`、`manifest.json` 绑定本次身份。策略引用原随机名单，不回写此 run。

`configs/strategy.json` 单独锁定 [E1/E2 策略](specs/s5/select.md)。`scripts/night_strategy.py` 管两臂调度、排序及子进程读取边界；`scripts/night_capture.py` 是 E1 原图及变体的 CUDA 子进程；长期回归归 `tests/train/diagnose/test_night_strategy.py`。[`runs/pool_002/`](runs/pool_002/manifest.json) 封存 E1：`e1.json` 留名单和逐步选择，`features.npz` 留共享原图特征，`scores.npz` 独存 E1 分数，`capture.json`、`run.json`、`strategy.json`、`code/`、`manifest.json` 留实际采集与身份。预检不建 run，不保留原图、标签、权重副本或整池预测。[`runs/pool_003/`](runs/pool_003/manifest.json) 独立封存 E2：`e2.json` 留三份名单、风险组与逐步覆盖/多样值，`run.json` 绑定父清单和共享特征 SHA、读取范围、复算和跨种子重合，`strategy.json`、`code/`、`manifest.json` 固定身份；不复制特征或分数。

## S6

[`specs/s6/hard.md`](specs/s6/hard.md) 维护困难尾类百图的仅图排名、入选标签支持与训练接续。`configs/hard.json` 绑定封存来源，`scripts/night_hard.py` 分离仅图排行与后置入选标签核验，长期边界回归归 `tests/train/diagnose/test_night_hard.py`。`runs/pool_007/` 留仅图名单和实际读取回执，`runs/hard_001/` 留支持和身份；不复制图、标签或旧分数。

`configs/joint.json` 固定已核验百图及联合损失，`scripts/night_joint.py` 执行、复算同名单控制和处理，`tests/train/test_night_joint.py` 保护损失隔离及几何语义。`runs/hard_002/` 留实际两臂同序输入收据、末轮权重、指标及 SHA；失败只声明已完成部分。

`configs/split.json` 锁 [同名单单因素拆解](specs/s6/hard.md#单因素拆解)。既有 `night_joint.py` 以独立模式保存 `runs/hard_003/` 的配置、两臂权重与开发证据，只引用不复制 `hard_002/control`。

`configs/quality.json` 锁 [仅质量分类](specs/s6/hard.md#仅质量分类)，复用同一入口与同名单控制，`runs/hard_004/` 留单臂原始记录、权重及开发证据。

`configs/distill.json` 锁 [白天回放蒸馏](specs/s6/hard.md#白天回放蒸馏)。`scripts/night_distill.py` 负责同白图教师信号和附加损失，`tests/train/test_night_distill.py` 保护梯度和锚点隔离。复用 `night_joint.py` 的对照与复算，`runs/hard_005/` 留训练侧预检和正式证据，不复制白天教师预测或 A 权重。

S7 的 [`configs/expand.json`](configs/expand.json) 和 `scripts/night_expand.py` 锁定 [300 张对照](specs/s7/plan.md#300-张探索)。`runs/expand_<编号>/` 保存相应选样、运行、末轮、开发评价与配对证据，不复制图或标签；长期回归归 `tests/train/test_night_expand.py`。

[`specs/s7/e2.md`](specs/s7/e2.md) 维护 300 张 E2 追加配比及 E1 接续分支；[`configs/e2_300.json`](configs/e2_300.json) 锁 D300、平衡臂、历史种子与配额。现有 `night_expand.py` 和同一 `expand_` 运行结构负责两新档；`mix.json` 保存处理配置，`selection.json` 另记覆盖、共图槽位和候选读取，`comparison.json` 另记同策略平衡配对。直接两极对照由 `mix-pair` 从封存证据复算。实际结果仍归夜间报告和对应 run。

`scripts/night_consensus.py` 是 [S7 尾类固定框复核](specs/s7/plan.md#尾类跨视图复核)入口，回归归 `tests/train/diagnose/test_night_consensus.py`。`runs/consensus_<编号>/` 保存同口径两域指标、配对、运行身份和封存；新视图保存必要的车辆分数与原图坐标，亲本离线对照只引用已有对象。不重复复制原始对象、权重、图或标签；`verify` 只读开发封存，不重新推理。

`runs/coverage_<编号>/` 留 [训练侧尾类支持](specs/s7/plan.md#训练侧尾类支持)的已选标签核验身份、固定风险组折的量化读数和清单，不保存图、标签正文或重复名单。

`runs/crop_<编号>/` 留 [夜间局部放大支持](specs/s7/plan.md#夜间局部放大支持)的聚合几何、风险组支持和父封存，不保留裁切图、标签或探针脚本。

`configs/crop.json` 与 `scripts/night_crop.py` 固定局部放大训练，复用 `night_expand.py` 的 `expand_` 正式运行证据；`tests/train/test_night_crop.py` 覆盖槽位、标签及几何。裁切图仅在批次内生成。

`runs/tail_diag/` 留 [尾类检出与误报迁移](specs/s7/plan.md#尾类检出与误报迁移)的三模型聚合数值、误报索引及输入身份；不复制原预测或样本图。

`runs/tail_gap/` 留 [尾类工作点分差](specs/s7/plan.md#尾类工作点分差)的固定事件、分数与误报摘要和输入身份；不复制原预测、原图或一次性计算脚本。

`runs/tram_flow/` 留 [有轨车辆候选流失](specs/s7/plan.md#有轨车辆候选流失)的两臂逐真值训练侧候选阶段摘要、匹配计数与身份，不留原图、密集输出或临时程序。

`runs/truck_pair/` 留 [卡车工作点互补](specs/s7/plan.md#卡车工作点互补)的匿名逐真值覆盖、误报迁移和输入封存身份，不复制原预测或原图。

`runs/truck_shift/` 留 [卡车抬分与排序诊断](specs/s7/plan.md#卡车抬分与排序诊断)的共同框重放与匿名边界证据，不复制原预测或原图。

`scripts/night_frontier.py` 是卡车预算边界排序的训练侧入口：`probe` 为原 32 批局部历史，`probe-full` 为[全程历史](specs/s7/plan.md#卡车全程历史排序)，`verify-probe` 校验两种封存。`runs/frontier_<编号>/` 留原训练点、配对与梯度摘要、执行代码和身份，不留特征或内存队列。`tests/train/test_night_frontier.py` 保护配对范围、两端梯度、平移不变性和折回时的偏置隔离。全程历史正式对照由 `scripts/night_expand.py run --frontier-sha256` 接收过门的封存，结果仍归 `runs/expand_<编号>/`。

`runs/frontier_diag/` 留 [卡车配对支持分解](specs/s7/plan.md#卡车配对支持分解)的匿名候选与逐层配对摘要，不留原图或特征。

`runs/rival_<编号>/` 留 [双尾类同锚竞争](specs/s7/plan.md#双尾类同锚竞争)的训练侧折／风险组支持、父封存和实测口径；隔离复验若实际执行另留收据。不复制图、标签内容、权重、特征或临时脚本；正式对照仍用 `runs/expand_<编号>/`。

`runs/agree_<编号>/` 留 [双尾类跨步长几何](specs/s7/plan.md#双尾类跨步长几何)的匿名固定框池重放摘要、折统计及身份；不留图、标签正文、坐标、密集张量或探针。正式对照若入围仍用 `runs/expand_<编号>/`。

`runs/ceiling_<编号>/` 留 [固定框池上限](specs/s7/plan.md#双尾类固定框池上限)的逐图漏检分解、对照／上限证据、折统计与父封存身份；边和原分只引用父封存，不复制预测、坐标或原始材料。

`runs/neighbor_<编号>/` 留 [尾类邻锚支持](specs/s7/plan.md#尾类邻锚支持)的候选顺序绑定、匿名邻格支持计数、固定五折读数及父封存；拟合仅在实际过门后单独留收据。不留图、坐标、标签正文、密集分数或探针。

`runs/roi_<编号>/` 留 [尾类框内冻结特征](specs/s7/plan.md#尾类框内冻结特征)的候选顺序绑定、各折组边界、匿名留折分、复算指标和系数身份；不留原图、标签正文、坐标、特征缓存、未入围系数或临时脚本。

`runs/roi_002/` 是 [框内分数蒸馏](specs/s7/plan.md#尾类框内分数蒸馏)的原排程无更新子型，只留对齐身份、匿名监督与实际梯度摘要；不复制原组外读出或其候选池，不存特征或临时程序。

`runs/roi_003/` 留 [组外反转支持](specs/s7/plan.md#尾类组外反转支持)的匿名真实纠正边、折统计及父封存；只引用原候选和组外分数，不复制其内容或保存临时脚本。

`runs/roi_004/` 留 [残差只降分](specs/s7/plan.md#尾类组外残差只降分)的逐图命中、固定折摘要、减分审计与父封存；不复制原预测、组外分、特征或一次性脚本。

`runs/roi_005/` 留 [高分误报判别](specs/s7/plan.md#尾类高分误报判别)的折与拟合边界身份、匿名风险分、逐图命中及父封存；不留 ROI 描述、模型副本或临时程序。

`runs/block_<编号>/` 留 [预算阻挡分解](specs/s7/plan.md#尾类预算阻挡分解)的固定事件、匿名误报类型与组、原命中和可达预算见证；源框、图、预测及权重只引用父封存。

`scripts/night_truck.py` 是 [卡车独立分数迁移](specs/s7/plan.md#卡车独立分数迁移)的同锚校正与预检入口。`runs/truck_<编号>/` 留支持、身份和代码快照；正式对照复用 `scripts/night_expand.py run --truck-sha256` 与 `expand_` 封存。`tests/train/test_night_truck.py` 保护作用范围、索引、支持门槛及折回隔离。

`scripts/night_floor.py` 是 [有轨车辆正分保底](specs/s7/plan.md#有轨车辆正分保底)的损失与训练侧 `probe` / `verify` 入口；`runs/floor_<编号>/` 留汇总、身份及执行代码快照。正式对照复用 `scripts/night_expand.py run --floor-sha256` 与 `expand_` 封存，`coverage.json` 附同名单训练侧逐层匿名覆盖。`tests/train/test_night_floor.py` 保护索引、同锚贴合、梯度与覆盖门槛。

`scripts/night_pairloss.py` 实现 [尾类跨图分差](specs/s7/plan.md#尾类跨图分差)并提供训练侧 `probe` / `verify`。`runs/pairloss_<编号>/` 仅留预检汇总、隔离收据、代码快照及清单；内存队列不落盘。正式训练复用 `scripts/night_expand.py run --pairloss-sha256` 及 `expand_` 封存。`tests/train/test_night_pairloss.py` 保护匹配索引、异组配对、梯度隔离及队列生命周期。

`scripts/night_tal.py` 执行和核验 [TAL 正样本分配预检](specs/s7/plan.md#尾类正样本分配)；`runs/tal_<编号>/` 留该次训练侧统计、抽样／环境身份和代码快照，不保留样本或预测。`tests/train/test_night_tal.py` 保护作用范围以及否决门槛。

`runs/tal_002/` 是 [有轨车辆候选边界](specs/s7/plan.md#有轨车辆候选边界)的观测证据，同类目录只留逐曝光摘要与清单，不留临时观测脚本或候选张量。

`scripts/night_tal.py ring` 复用训练侧预检入口执行 [有轨车辆候选区补位](specs/s7/plan.md#有轨车辆候选区补位)的无更新信号门。正式对照复用 `scripts/night_expand.py run --ring-sha256` 与 `expand_` 封存；`run.json` 绑定已通过预检，`train.json` 留实际补位监督。相关回归在 `tests/train/test_night_tal.py` 与 `tests/train/test_night_expand.py`。

`runs/contour_<编号>/` 留 [训练内轮廓筛查](specs/s7/plan.md#局部轮廓的训练内筛查)的支持、工作点和归档身份，不保存暂存图、裁图、原标签、特征或未入围权重。

`runs/quality_<编号>/` 留 [定位不确定性](specs/s7/plan.md#定位不确定性筛查)的训练内折支持、度量和身份，不保存密集张量、一次性程序或未入围权重。

[`configs/margin.json`](configs/margin.json) 锁 [S7 原生车型分差](specs/s7/plan.md#原生车型分差)的亲本和强度；`scripts/night_margin.py` 提供只触及夜间车辆分配正例的训练损失，回归归 `tests/train/test_night_margin.py`。复用 `night_expand.py` 与新 `runs/expand_<编号>/`，额外的 `margin.json`、实际信号和同名单配对绑定本次处理；原平衡臂不回写。

[`specs/s6/plan.md`](specs/s6/plan.md) 维护对照阶段状态、共同配方和 E1→E2 的衔接；[`specs/s6/e1.md`](specs/s6/e1.md) 单独维护 E1 机制对照的假设、配置、执行细节和收尾门。计划不作实验收据，结果归封存 run 及报告。

`scripts/night_train.py` 是随机臂本机训练及开发评价入口，`configs/adapt.json` 锁配方和基线身份；`tests/train/test_night_train.py` 保护轮序、用途与框几何。`runs/d_<编号>/` 保存正式随机臂原始记录：各 seed 的 `inputs.json`、`train.json`、`last.pt`、`dev/` 和采样 GPU 数据；批次预检仅保存证据，不保留预检权重、图片、标签内容或缓存。`protocol.json`、`code/`、`run.json` 及 `manifest.json` 绑定状态和复查范围；代码快照非活动入口。

`scripts/night_adapt.py` 为 E1/E2 本机配对入口，活动身份归 `configs/pair.json`，复用 D 的训练核心和五类评价。`align` 只读名单和 D 实际轮序，`run` 只启动指定策略臂。`runs/e1_<编号>/`、`runs/e2_<编号>/` 保存 `slots.json`（选样顺序、D 原槽位、映射及复算身份）、`preflight.json`、逐 seed 的输入、训练日志、`last.pt` 与两域开发预测/评价，以及 `comparison.json`（实际完成三个 seed 后产生的 D 配对结果）。`protocol.json`、`run.json`、`code/`、`manifest.json` 绑定身份和已完成范围；失败 run 不冒称完成。标签准备仅给指定 arm/seed 的窄输入在项目外暂存；不保留原图、标签内容、预检权重或缓存。

`configs/ratio.json` 锁定 [S6 单种子比例探索](specs/s6/plan.md#单种子比例探索)，既有选样与训练入口通过独立 `ratio` 模式执行。`runs/pool_004/` 的 `lists.json` 保存四份探索名单、排序证据和重合数，`ratio.json`、`run.json`、`code/`、`manifest.json` 绑定首版回放和读取边界，不复制特征或分数。策略训练同系列的新 run 用 `ratio` 身份区分，`comparison.json` 改记单 seed 相对 D 和同策略首版的两套配对；不是三种子复验结果。

[`specs/s6/rank.md`](specs/s6/rank.md) 维护低误报排序的诊断、训练动机和取舍。`configs/rank.json` 绑定实验；`scripts/night_rank.py` 复用封存 E1 名单及已锁开发评价，只写独立 `runs/rank_<编号>/` 的输入、训练、末轮权重、开发输出、配对指标、实际代码/配置及清单。原图、标签和特征副本不保留；长期回归在 `tests/train/test_night_rank.py`。

[`specs/s6/replay.md`](specs/s6/replay.md) 维护低照白天回放的机制对照；`configs/replay.json` 锁亮度变换和父封存，`scripts/night_replay.py` 复用原适配及开发入口，`runs/replay_<编号>/` 保留原槽位、实际曝光、固定末轮及开发证据。长期回归在 `tests/train/test_night_replay.py`，不留处理后的白天图。

[`specs/s6/tail.md`](specs/s6/tail.md) 维护尾类名额对照，`configs/tail.json` 锁配额，`scripts/night_tail.py` 仅用已封存的图片特征和预测复算；实际 `runs/tail_<编号>/selection.json` 保留仅图步骤、名单与读取回执。训练与开发产物同前，长期回归在 `tests/train/test_night_tail.py`。

`scripts/verify_night.py` 是 S6 上述新训练 run 的只读复核入口：封存文件、白名单、固定开发口径与各 seed 配对复算，不启动模型或创建重复结果。长期回归在 `tests/train/diagnose/test_verify_night.py`。

[`specs/s6/repeat.md`](specs/s6/repeat.md) 维护同名单尾类曝光对照，`configs/repeat.json` 与 `scripts/night_repeat.py` 锁白名单、已选标注的使用和两段采样；`runs/repeat_<编号>/` 留同一百图的实际重复次数、训练和开发证据。长期回归在 `tests/train/test_night_repeat.py`。

[`specs/s6/presence.md`](specs/s6/presence.md) 维护尾类共识选样对照。`configs/presence.json`／`scripts/night_presence.py` 用 A 的仅图跨视图配对生成分数；实际 `runs/presence_<编号>/` 留复算分数、名单和训练指标，不重复归档特征、图或标签。长期回归在 `tests/train/test_night_presence.py`。

[`specs/s6/focus.md`](specs/s6/focus.md) 维护原生夜间分类的难例权重对照；`configs/focus.json`／`scripts/night_focus.py` 保持原名单、槽位、框和开发定义。`runs/focus_<编号>/` 只保存必要训练、两域开发与封存证据；长期回归归 `tests/train/test_night_focus.py`。

[`specs/s6/style.md`](specs/s6/style.md) 维护频域外观回放，`configs/style.json` 与 `scripts/night_style.py` 锁定保留训练标签的白天外观变换，`runs/style_<编号>/` 留实际训练和开发证据，不复制变换图片。长期回归在 `tests/train/test_night_style.py`。

[`specs/s6/long.md`](specs/s6/long.md) 维护原轨迹延长的对照，`configs/long.json` 与 `scripts/night_long.py` 锁额外日程及原训练前缀，`runs/long_<编号>/` 只留固定末轮和实际证据。长期回归在 `tests/train/test_night_long.py`。

[`specs/s6/width.md`](specs/s6/width.md) 维护保留原名单的尺度补位，`configs/width.json`／`scripts/night_width.py` 锁仅图尺度与名单边界，`runs/width_<编号>/` 保存复算分数和效果。长期回归在 `tests/train/test_night_width.py`。

[`specs/s6/target.md`](specs/s6/target.md) 维护原名单上的夜间单域对照，`configs/target.json`／`scripts/night_target.py` 锁零白天梯度和原夜间顺序，`runs/target_<编号>/` 留实际消费和两域验证。长期回归归 `tests/train/test_night_target.py`。

[`specs/s6/paste.md`](specs/s6/paste.md) 维护现有白天尾类供体与夜景合成回放边界，`configs/paste.json`／`scripts/night_paste.py` 锁标签保持和梯度安排；`runs/paste_<编号>/` 留事件汇总与开发证据，不保存合成图。长期回归在 `tests/train/test_night_paste.py`。

`scripts/seed_train.py` 是原白天 A 的受限重复入口，`scripts/seed_queue.py` 是现有 BitaHub 实例的本机队列与收尾入口。`runs/replica_<编号>/` 形成后只保留逐 seed 权重与必要训练记录、清单和云交接收据；排队过程中的本机锁、部署副本与传输包不入 Git。

24 轮配方按臂保留 `epoch08/ema.pt`、`epoch16/ema.pt` 及 `last.pt`，每个完整轮以 `last.pt` 恢复。第 8、16、24 轮统一使用 `epoch<两位轮数>/dev960/` 存放评价证据，例如 `epoch24/dev960/metrics.json`。`comparison.json` 的 `by_epoch` 对应各观测点，原 R80 参照只保存一次。`runs/s3_17/scale24_preflight.json` 保存本机预检与包身份。入口、命令和范围见[延长微调](specs/s3/scale.md#延长微调)。

`runs/s3_17/scale24_queue.json` 保存本次排队、预算、控制器身份、平台 job、部署路径、训练交接和实际关机状态。`gpu_queue.py` 只更新这份收据，拒绝覆写后重复提交。控制台输出放仓库外临时目录；有效状态以收据为准。云端 `job/` 的命令、控制器代码、退出状态和训练日志在成功交接时取回 `runs/scale_003/job/`，与原训练清单分开。评价仍使用既有本机入口。

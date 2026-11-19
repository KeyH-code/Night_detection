# 结构

## 目录

| 位置 | 用途 | 允许内容 | 命名 |
|---|---|---|---|
| 根目录 | 项目入口 | README.md、AGENTS.md、STRUCTURE.md、RULES.md、plan.md、Git规则 | 固定短名 |
| specs/ | 阶段文档 | 每个计划阶段单独建子目录；根目录不散放阶段文件 | 小写阶段ID，如s0/、s2/、s10/；与根plan阶段表对应 |
| specs/s0/、specs/s1/ | 已完成阶段索引 | review.md链接已有数据/首训证据，注明事后整理 | 固定短名review.md；不补造同期plan或复制原始记录 |
| specs/s2/—specs/s10/ | 各阶段开发安排 | plan.md；有独立用途时增加roadmap.md、changemap.md等，职责见下表 | 固定短名；s2已执行待阶段复核，s3已审、本地实现/回归完成，s4—s10为待细化框架；不机械创建空文档 |
| scripts/ | 活动实现 | 数据、训练、校验、打包入口及共用模块 | 小写；必要时下划线；不加临时时间或实例端口 |
| tests/data/ | 数据回归 | 归档安全、框转换、分组、冻结身份 | test_<功能>.py |
| tests/train/ | 训练回归 | 白名单、选模、续训、资源、运行和发布边界 | test_<功能>.py |
| tests/train/diagnose/ | 固定权重诊断回归 | 原生匹配/导出、工作点、尺寸、风险组区间与审核身份与抽查/自动草稿边界 | test_<功能>.py；完整pytest递归覆盖，训练包顶层glob不带入本机诊断回归 |
| configs/ | 配置 | 类别映射、实验参数、README.md参数说明 | 按功能/实验命名；JSON/YAML |
| environment/ | 环境 | 数据依赖、平台锁、安装入口 | 按平台/用途；旧有效锁名保留 |
| data/soda10m/ | 数据 | 下表限定内容 | 数据集/版本/用途分层 |
| runs/a_001/ | 首训A基线 | 一份原始训练记录、代码/配置/环境快照、权重及清单；不等于最终A已冻结 | 实验名_序号；如 a_001 |
| runs/a_001_train/ | 固定best的训练集评价 | metrics.json、comparison.csv、run.json、必要日志及评价代码快照；不复制数据/权重 | 关联实验名_用途；原训练清单不改 |
| `runs/<实验>/`、`runs/<实验>_diag/` | 候选、诊断、对照及冻结证据 | 实际执行后保存配置/代码/环境身份、必要日志、指标、权重及冻结记录；固定权重诊断保存可复查预测/匹配/PR，不复制原权重 | 训练用a_002、d_001等短ID；诊断用`<实验>_diag`；seed写身份/配置；不预建空目录 |
| reports/ | 结论 | data.md数据报告、a.md源模型报告；后续形成有效结果后维护对应主题报告 | 按主题/实验；不留过程流水 |
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
| STRUCTURE.md | 维护目录的职责、允许内容与命名模式 |
| RULES.md | 实验指标、保护、证据、选模、停止及冻结标准和依据；长期留存，不是临时审核材料 |
| plan.md | 全项目目标、范围、阶段状态、衔接、文档导航和实验矩阵 |
| specs/s<编号>/plan.md | 该阶段目标、输入、实施步骤、文件范围、验证、产物、失败分支和审核状态；引用RULES |
| specs/s<编号>/roadmap.md | 可选；复杂阶段内部的里程碑、顺序和依赖，不复制根plan路线或plan任务正文 |
| specs/s<编号>/changemap.md | 可选；拟改文件/接口、变更原因、引用影响和验证对应关系；简单改动合入阶段plan |
| specs/s3/scale.md | 行人分辨率专项路线、训练尺度对照与执行准备 |
| specs/s<编号>/review.md | 已完成阶段的证据索引、退出结论及必要限制；无同期计划时明确事后整理，不替代reports或runs |

后续实验/图目录行描述生成模式，不表示这些目录或结果已经存在。只有实际形成可复查证据时创建；本次规则与规划不预建实验目录或空报告。

`plan.md`维护全项目路线和阶段衔接；`specs/`按阶段组织具体安排；`RULES.md`定义验收标准；`reports/`记录执行后的有效结论。阶段文档不兼作启动prompt、操作流水或原始实验存储。每个职责只有一个主要位置，使用链接衔接，不通过复制正文凑齐文件。

当前S0—S10各有目录：S0/S1为事后证据索引，S2计划已审核通过并进入实施，S3采样对照计划已审、本地实现与回归完成，S4—S10仍须在进入阶段前结合前置结果细化。当前不需要独立roadmap/changemap，不创建空模板；阶段建档不表示其他阶段执行已获授权。

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

`runs/a_001/manifest.json` 是当前正式A留存清单，逐项记录相对路径、哈希、大小及旧来源路径。它只声明实际保留内容，不声称原60份文件全部还在。

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

本地收尾前将原云端封存清单原样保存在job/cloud_manifest.json，保留下载身份；train/manifest.json随后管理新增本地评价证据。最后一条run的job/cloud_completion.json记录两臂完成与云端进程/GPU核实，不代替平台关机或账单确认。配对结果使用runs/s3_<seed>/（如s3_17/），保存comparison.json/csv、逐臂intervals.json、采样汇总sampling.json/csv及绑定输入/代码的run.json；它是对照结果，不是新训练run。对应plots/s3_<seed>/的metrics.png、effects.png、sampling.png可用scripts/plot.py --source-pair重建。

配对记录复核继续维护同一目录：transitions.json保存GT迁移及最终框的分数/IoU/匹配证据，同名csv是行人迁移GT的查阅简表；trajectory.csv/json保存原云端逐轮指标、描述性窗口及选模敏感性，不能伪装成逐轮逐GT预测或独立重复。trajectory.png用scripts/plot.py --trajectory重建。图表代码更替时，按实际被图引用的code/versions/<SHA前12位>/保存必要版本，不回写既有图所引用的代码快照。一次性统计程序仍放仓库外；机器证据与视觉/因果结论分开。

固定 R 的输入尺度对照归 `runs/s3_17/scale.json`，保存权重及数据身份、三臂输入几何、逐目标覆盖和工作点迁移、六类 AP 及风险组区间。原图、权重和临时缩放图只引用冻结源或放在临时目录。

`scripts/person_fusion.py` 是行人多视图融合的本机白天开发复现入口。无梯度，单方案只用同一 checkpoint；`a1` 保持 `a_001 best` 的 640 非行人输出，`r80` 保持 R80 的 960 非行人输出。`tests/train/diagnose/test_person_fusion.py` 保护合并、类别隔离、上限、镜像坐标和身份语义。`runs/scale_<编号>/` 的固定权重子型只留筛查摘要、最终候选及原生参照的比较、可复查的融合对象及必要身份，不存重复原图、权重或验证器临时目录。

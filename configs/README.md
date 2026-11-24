# 配置

- `soda10m_categories.json`：官方1..6到训练0..5的映射；pedestrian、car为主指标，cyclist不并入pedestrian。当前冻结状态以数据READY为准。
- `source_a.yaml`：A的复用配方。dataset限定source_train/source_dev；environment限定Python和框架版本；model限定COCO初始化来源；training为实际训练参数；selection为主类索引、严格改进和checkpoint周期；smoke为小样本验证；cloud为资源下限。

原始A使用YOLO11m/640、seed17、AMP、SGD、最多100轮、patience20、batch16、workers4，配方完整值见YAML。每轮以pedestrian/car AP50:95宏平均选模；并列保留较早best。正式最终值见`../runs/a_001/train/audit_initial/`，历史配置快照在`../runs/a_001/code/configs/source_a.yaml`。

显式CLI值和库默认值可能改变实际行为，不能仅凭YAML宣称参数生效。新实验必须记录传入参数、最终trainer参数和原始库默认值。改配置应创建新实验身份，不修改历史快照或冻结数据。当前A的预检仅允许OOM时batch16转8；不能把本机smoke的batch2当正式配置。

## 验收

后续A候选的六类质量、保护、证据、选模与停止标准统一见[规则](../RULES.md)，阶段安排见[计划](../plan.md)。本文不重复阈值。

`source_a.yaml`继续保留旧主类宏平均/旧patience路径。`source_s3.yaml`显式锁定原A梯度配方，R/S支持seed17/29/43，U当前仅登记seed17；4000槽位、batch16/workers4、100轮、horizon_only。R为无放回排列，S为2:1有放回，U为1:1有放回。六类资格与证据筛查后选best；无合格时独立保留diagnostic，不自动冻结。规则直接解析RULES.md，不在配置重复门槛。

## source_a兼容观察

monitor.enabled=true、every=1表示以后preflight/formal每轮额外对完整source_train的当轮EMA做无增强AP/P/R评价；batch16、workers0仅用于这个独立评价加载器，梯度训练仍batch16/workers4。原生train/val loss每轮已记录，不需补造。train_metrics.csv含六类AP50/AP50:95/P/R、同轮dev AP及耗时；train_monitor.json绑定身份。观察只写日志，不参与best/早停，不更新参数；不静默改抽样或频率。

当前活动配置增加了观察，原A当时的配置仍以runs/a_001/code快照为准；云端必须为新代码/配置重做预检与配方冻结，将评价的真实时间计入预算。当前没有启动新训练。

## S3

训练入口以`--config configs/source_s3.yaml --arm R --seed 17`显式选择；相同seed两臂除采样以外的训练参数一致，CLI不能静默改强度、批量或日程。checkpoint_period=0表示只保留有效角色与last；其他原梯度参数逐键对照原始args.yaml，差异拒绝启动。

S3当前profile=dev_only_local_final_v1：每轮仅完整500 dev，batch1/workers0、FP16及S2锁定的匹配/PR口径；4000 train观察和最终导出移到本地，使用保存的云端dev阈值，不逐轮做train推理。AMP初始缩放固定1（1024在S预检第26批出现溢出），后续GradScaler状态随checkpoint恢复；两臂一致。默认F1指标单独保留。preflight为100轮日程下完成2轮后停止，使用完整数据，不是把epochs改成2。S3不使用小样本smoke或OOM自动降批量；失败预检从COCO重启，formal完整轮checkpoint可恢复。

新云端调用仍经run_source_cloud_job.py；各arm/seed须新runtime身份与完整预检收据，旧A凭证、其他臂/seed或缺完整观察均拒绝。用户已授权新安排并自行停机；当前完成本机改造，待用户重开机再按新身份预检。旧全train观察收据不可复用。

## U控制

`--config configs/source_s3.yaml --arm U --seed 17`解析出有效权重1/1及indices_rng_scope=S：同一PCG64流和显式概率choice，保留U身份，不复用S凭证。R/S的解析结果和原采样流保持不变，U不接受29/43或其他强度；后续重复需先登记。

`source_s3_control.json`是复用旧R/S的精确兼容配置，绑定原清单、旧/新源码哈希、原配方/数据/规则及锁环境，供source_control.py独立核验。它只允许已审U增量，不能自动接纳新公共训练代码；改训练/评价/库/数据须重新判断是否仍可复用。原compare_runs仍要求严格R/S身份一致。运行身份额外绑定本配置及source_control.py，不将本地兼容通过视为Linux/4090预检通过。

## 夜间

`night.json` 登记 `night_001`、`d_001` 绑定的[历史两类评价口径](../specs/s5/plan.md#一协议评审)与数据、共同权重身份，不复用白天 Precision 工作点，不包含选样超参或训练预算。`scripts/night_eval.py` 的 `reference` 与 `verify` 均要求 `--protocol-sha256` 对照外部确认的配置 SHA；打印 JSON，不训练、不调用模型。真实用途须按 S5 的批次授权。

获准的 `run` 同样校验配置身份，另需 `--scratch`（同卷仓库外任务目录）、`--output`（全新 `runs/night_NNN`）、`--reference` 与 `--reference-sha256`。仅用冻结 A 进行 `night_dev` 小批预检和完整无梯度评价，不以预检计分。不得把全量数据验证命令作为替代入口。

`select.json` 只锁定[随机对照](../specs/s5/plan.md#已锁随机对照)，不含策略参数。`scripts/night_select.py` 要求显式 `--protocol-sha256` 与 `--images-sha256`，使用同卷的仓库外 `--scratch` 及全新 `runs/pool_NNN` 作为 `--output`。仅核验冻结候选池的仅图清单、图片及有序抽签，不访问标签或调用模型。

`strategy.json` 单独锁定 [E1/E2](../specs/s5/select.md) 的参数、图像清单、D 名单和冻结 A 身份。`scripts/night_strategy.py` 的三个公开模式均要求审核后确认的 `--protocol-sha256` 和已存在的项目外 `--scratch`。须按阶段授权后才执行：

```powershell
.venv-train/Scripts/python.exe -B scripts/night_strategy.py e1-preflight --protocol-sha256 '<审核确认的 strategy.json SHA-256>' --scratch '<项目外任务目录>'
.venv-train/Scripts/python.exe -B scripts/night_strategy.py e1 --protocol-sha256 '<审核确认的 strategy.json SHA-256>' --scratch '<同卷项目外任务目录>' --output runs/pool_NNN
.venv-train/Scripts/python.exe -B scripts/night_strategy.py e2 --protocol-sha256 '<审核确认的 strategy.json SHA-256>' --scratch '<同卷项目外任务目录>' --e1-run '<已封存的 E1 run 路径>' --e1-manifest-sha256 '<E1 manifest.json SHA-256>' --output runs/pool_NNN
```

`e1-preflight` 只采集固定八图并打印回执，不接受 `--output`；完整 `e1` 封存三份名单和分离的特征、分数。`e2` 校验 E1 父封存清单，只消费共享原图特征及对齐的窄元数据，不读取 E1 分数、名单或候选原图。`e1` 和 `e2` 各用独立的全新输出，不覆盖封存 run。

`adapt.json` 冻结 [S6 随机臂](../specs/s6/plan.md#随机臂-d)的梯度预算及基线身份。`scripts/night_train.py run` 显式校验 `--config-sha256`，要求项目外 `--scratch` 和全新的 `runs/d_NNN`；只有实际授权后才能执行。内部数据准备子进程不作选样入口。

### 五类统一评价

[`accept.json`](accept.json) 锁定新 [五类规则](../specs/s5/plan.md#恢复目标)、固定开发清单和 A/D 封存来源，不改历史 `night.json`。`replay` 只读这些封存开发预测及必要收据，离线重匹配和统计，不打开原图、标签文件、权重或最终留出。使用已存在的同卷项目外临时目录及全新输出，首次封存见 [`accept_001`](../runs/accept_001/manifest.json)：

```powershell
.venv-train/Scripts/python.exe -B scripts/night_accept.py replay --protocol-sha256 '<审核确认的 accept.json SHA-256>' --scratch '<同卷项目外任务目录>' --output runs/accept_NNN
```

### 策略配对训练

[`pair.json`](pair.json) 绑定 D 训练配方、D/E1/E2 名单封存、A/D 五类参照和每臂 60 分钟本机窗口。`align` 不读取候选标签，验证六份映射、实际 D 顺序和增强哈希。`run` 再核验对齐，只启动 `--arm` 的三个种子，seed 17 先预检；仅在获得训练授权后执行。`prepare` 为独立标签准备子进程，不是公开运行入口。

```powershell
.venv-train/Scripts/python.exe -B scripts/night_adapt.py align --protocol-sha256 '<审核确认的 pair.json SHA-256>'
.venv-train/Scripts/python.exe -B scripts/night_adapt.py run --arm e1 --protocol-sha256 '<审核确认的 pair.json SHA-256>' --scratch '<项目外任务目录>' --output runs/e1_NNN
```

E2 使用 `--arm e2` 和新 `runs/e2_NNN`。在同一台设备上按 E1→E2 顺序；异常停止，不静默转云或改配方。

### 单种子比例探索

[`ratio.json`](ratio.json) 锁 seed 43 和 40/60、60/40 两档，绑定原策略、配方及封存父结果。`ratio` 选样重放旧特征，E1 单独读旧分数；`ratio-align` 在不打开原图或标签文件的情况下验证四份名单和 D 槽位。正式训练须有单独授权，按新名单封存 SHA 和一组比例启动，输出不可覆盖。

```powershell
.venv-train/Scripts/python.exe -B scripts/night_strategy.py ratio --protocol-sha256 '<ratio.json SHA-256>' --scratch '<同卷项目外任务目录>' --output runs/pool_NNN
.venv-train/Scripts/python.exe -B scripts/night_adapt.py ratio-align --protocol-sha256 '<ratio.json SHA-256>' --pool-run runs/pool_NNN --pool-sha256 '<名单 manifest.json SHA-256>'
.venv-train/Scripts/python.exe -B scripts/night_adapt.py ratio --protocol-sha256 '<ratio.json SHA-256>' --pool-run runs/pool_NNN --pool-sha256 '<名单 manifest.json SHA-256>' --arm e1 --random-count 40 --scratch '<项目外任务目录>' --output runs/e1_NNN
```

### 分类排序适配

[`rank.json`](rank.json) 锁定[低误报排序实验](../specs/s6/rank.md)的亲本清单、种子、白天回放和拟合参数。入口只使用封存 E1 的对应百图及其原白天训练顺序，不接受任意数据目录，开发评价使用合规的原生 640 和行人多视图。

```powershell
.venv-train/Scripts/python.exe -B scripts/night_rank.py --config-sha256 '<rank.json SHA-256>' --scratch '<项目外任务目录>' --output runs/rank_NNN
```

### 低照回放

[`replay.json`](replay.json) 锁 [低照回放单因素对照](../specs/s6/replay.md)，复用 E1 父封存名单、原白天槽位及 200 更新，从冻结 A 重启，只在每两张白天回放中的第二张叠加低照变换。

```powershell
.venv-train/Scripts/python.exe -B scripts/night_replay.py --config-sha256 '<replay.json SHA-256>' --scratch '<项目外任务目录>' --output runs/replay_NNN
```

### 尾类选样

[`tail.json`](tail.json) 锁 [尾类配额对照](../specs/s6/tail.md)。排序进程只打开封存特征、分数和仅图元数据，验证原 E1 等额重放，在独立新名单冻结后才暂存其 100 个标签；训练配方与原 E1、D 相同。

```powershell
.venv-train/Scripts/python.exe -B scripts/night_tail.py --config-sha256 '<tail.json SHA-256>' --scratch '<项目外任务目录>' --output runs/tail_NNN
```

### 同名单尾类曝光

[`repeat.json`](repeat.json) 锁 [固定夜间百图的曝光对照](../specs/s6/repeat.md)。训练前半按 E1 原顺序，后半只依据这 100 张已获准训练图的标签确定加权重采样；白天和梯度更新数不变。

```powershell
.venv-train/Scripts/python.exe -B scripts/night_repeat.py --config-sha256 '<repeat.json SHA-256>' --scratch '<项目外任务目录>' --output runs/repeat_NNN
```

### 新训练证据复算

```powershell
.venv-train/Scripts/python.exe -B scripts/verify_night.py runs/replay_002
```

对 `rank`、`replay`、`tail`、`repeat`、`presence`、`focus`、`style`、`long`、`width`、`target`、`paste` 的正式封存读取清单、白名单与开发预测，不打开图片、标签内容或启动模型。同一命令可接多个不同 seed run；拒绝不同配方、代码、环境、评价或重复 seed。区分三种子点估计全过和证据完整的稳定恢复。

### 共识选样

[`presence.json`](presence.json) 锁[共识选样对照](../specs/s6/presence.md)。先在仅图沙箱采集冻结 A 的原图、提亮、镜向同类支持，再与已封存的 E1 特征一起排序；训练原生 200 更新，不改开发评价。

```powershell
.venv-train/Scripts/python.exe -B scripts/night_presence.py --config-sha256 '<presence.json SHA-256>' --scratch '<项目外任务目录>' --output runs/presence_NNN
```

### 低误报训练

[`focus.json`](focus.json) 锁[夜间两类困难候选的原生分类损失对照](../specs/s6/focus.md)。固定 E1 名单、输入顺序和原生分配，只对夜间卡车／有轨车辆的部分 BCE 元素增加梯度权重。

```powershell
.venv-train/Scripts/python.exe -B scripts/night_focus.py --config-sha256 '<focus.json SHA-256>' --scratch '<项目外任务目录>' --output runs/focus_NNN
```

### 频域白天回放

[`style.json`](style.json) 锁[频域外观对照](../specs/s6/style.md)。已选的同微批夜间图仅为低频外观提供参照，不改该夜图、不改训练标签，也不直接输入候选池其他图。

```powershell
.venv-train/Scripts/python.exe -B scripts/night_style.py --config-sha256 '<style.json SHA-256>' --scratch '<项目外任务目录>' --output runs/style_NNN
```

### 延长适配

[`long.json`](long.json) 锁[原轨迹延长对照](../specs/s6/long.md)：从共同 A 重新启动，前 200 更新对比封存 E1 的逐步损失；只有前缀一致才执行已登记的后 400 更新。只评价末轮，不重选。若后续入围，每个重复种子分别用独立 run，不共用一个 45 分钟训练窗口。

```powershell
.venv-train/Scripts/python.exe -B scripts/night_long.py --config-sha256 '<long.json SHA-256>' --scratch '<项目外任务目录>' --output runs/long_NNN
```

### 尺度补位

[`width.json`](width.json) 锁[尾类尺度覆盖对照](../specs/s6/width.md)：不读候选标签，保留 E1 的前 80 张，以中等尺度候选和多样性选新的 20 张；训练仅暂存锁定百图。

```powershell
.venv-train/Scripts/python.exe -B scripts/night_width.py --config-sha256 '<width.json SHA-256>' --scratch '<项目外任务目录>' --output runs/width_NNN
```

### 夜间单域对照

[`target.json`](target.json) 锁[去掉白天梯度的反事实](../specs/s6/target.md)：原 E1 夜间名单、增强和 200 更新不变，训练只暂存已选夜间图；白天开发仍完整评价。

```powershell
.venv-train/Scripts/python.exe -B scripts/night_target.py --config-sha256 '<target.json SHA-256>' --scratch '<项目外任务目录>' --output runs/target_NNN
```

### 尾类合成回放

[`paste.json`](paste.json) 锁[白天供体与已选夜景的合成对照](../specs/s6/paste.md)。仅在已获准训练输入之间粘贴尾类局部，碰到已有标签或供体无效则保留原白天槽位；真实夜图、原生损失、梯度更新数及开发评价不变。

```powershell
.venv-train/Scripts/python.exe -B scripts/night_paste.py --config-sha256 '<paste.json SHA-256>' --scratch '<项目外任务目录>' --output runs/paste_NNN
```

### 300 张嵌套对照

[`expand.json`](expand.json) 锁定 [S7](../specs/s7/plan.md#300-张探索) 的随机先行和同梯度预算。策略入口额外要求已封存随机臂的路径及清单 SHA。

```powershell
.venv-train/Scripts/python.exe -B scripts/night_expand.py run --arm d --config-sha256 '<expand.json SHA-256>' --scratch '<项目外任务目录>' --output runs/expand_001
.venv-train/Scripts/python.exe -B scripts/night_expand.py verify --output runs/expand_001
.venv-train/Scripts/python.exe -B scripts/night_expand.py run --arm e1 --d300 runs/expand_001 --d300-sha256 '<随机封存清单 SHA-256>' --config-sha256 '<expand.json SHA-256>' --scratch '<项目外任务目录>' --output runs/expand_002
```

### 300 张 E2 追加配比

[`e2_300.json`](e2_300.json) 锁定 [三档配比](../specs/s7/e2.md)。`mix-check` 只读重放三档、平衡引用和槽位；在正常 E2 入口额外传 `--mix-sha256` 与 `--representatives 133` 或 `67`，才运行一档新训练。只读 `verify` 兼容原扩量与新配比。

```powershell
.venv-train/Scripts/python.exe -B scripts/night_expand.py mix-check --config-sha256 '<expand.json SHA-256>' --mix-sha256 '<e2_300.json SHA-256>' --scratch '<项目外任务目录>'
.venv-train/Scripts/python.exe -B scripts/night_expand.py mix-pair --diverse-run runs/expand_005 --representative-run runs/expand_004
```

### 尾类跨视图复核

[`night_consensus.py`](../scripts/night_consensus.py) 按 [S7 预登记](../specs/s7/plan.md#尾类跨视图复核) 固定 `expand_003/43` 及开发验收参照。`run` 只在两域开发集上采集 640 镜向和 960 原向辅助视图；`verify` 只读封存复算两臂，不启动模型。

```powershell
.venv-train/Scripts/python.exe -B scripts/night_consensus.py run --scratch '<同盘项目外任务目录>' --output runs/consensus_NNN
```

[亲本复核](../specs/s7/plan.md#尾类亲本复核)用 `parents`，仅打开嵌套亲本和冻结 A 的封存预测及身份，不打开图像或启动 GPU。

```powershell
.venv-train/Scripts/python.exe -B scripts/night_consensus.py parents --scratch '<同盘项目外任务目录>' --output runs/consensus_001
.venv-train/Scripts/python.exe -B scripts/night_consensus.py verify --output runs/consensus_001
```

### 原生车型分差

[`margin.json`](margin.json) 锁定 [S7 单臂](../specs/s7/plan.md#原生车型分差)，只对原平衡 E2-300 同名单、同顺序的已分配夜间车辆正例加车型分差项。与 `--mix-sha256` 排他；同一 `verify` 核验训练消费及对原平衡臂的开发配对。

```powershell
.venv-train/Scripts/python.exe -B scripts/night_expand.py run --arm e2 --config-sha256 '<expand.json SHA-256>' --margin-sha256 '<margin.json SHA-256>' --d300 runs/expand_001 --d300-sha256 '<随机封存清单 SHA-256>' --scratch '<项目外任务目录>' --output runs/expand_006
.venv-train/Scripts/python.exe -B scripts/night_expand.py verify --output runs/expand_006
```

### 困难尾类百图

[`hard.json`](hard.json) 固定仅图分数来源和 [S6 名单与支持门](../specs/s6/hard.md)。`select` 生成 `pool_007`；`support` 要求该名单的封存清单 SHA，之后才以只读方式核验入选标签并生成 `hard_001`；`verify` 要求两个清单 SHA，重复隔离验证，不读任何其他标签或启动模型。

```powershell
.venv-train/Scripts/python.exe -B scripts/night_hard.py select --config-sha256 '<hard.json SHA-256>' --scratch D:/Temp
.venv-train/Scripts/python.exe -B scripts/night_hard.py support --config-sha256 '<hard.json SHA-256>' --selection-sha256 '<pool_007 SHA-256>' --scratch D:/Temp
.venv-train/Scripts/python.exe -B scripts/night_hard.py verify --config-sha256 '<hard.json SHA-256>' --selection-sha256 '<pool_007 SHA-256>' --support-sha256 '<hard_001 SHA-256>' --scratch D:/Temp
```

[`joint.json`](joint.json) 将这两份封存绑定到同名单训练：先原损失控制，再联合损失，两臂从 A 独立重启，结束后用 `verify` 从封存的昼夜对象重算效果而不重训或推理。

```powershell
.venv-train/Scripts/python.exe -B scripts/night_joint.py run --config-sha256 '<joint.json SHA-256>' --scratch D:/Temp
.venv-train/Scripts/python.exe -B scripts/night_joint.py verify --config-sha256 '<joint.json SHA-256>' --manifest-sha256 '<hard_002 SHA-256>'
```

[`split.json`](split.json) 复用 `hard_002/control`，依次训练 `loc`、`negative` 两臂。`verify-split` 只读完整开发证据，重算与同名单原损失的对比及两层白天保护。无须重跑百图排序或历史控制。

```powershell
.venv-train/Scripts/python.exe -B scripts/night_joint.py split --config-sha256 '<split.json SHA-256>' --scratch D:/Temp
.venv-train/Scripts/python.exe -B scripts/night_joint.py verify-split --config-sha256 '<split.json SHA-256>' --manifest-sha256 '<hard_003 SHA-256>'
```

[`quality.json`](quality.json) 锁定 [仅 IoU 质量分类](../specs/s6/hard.md#仅质量分类)，复用同一原损失控制；不重新训练历史臂。新封存为 `hard_004`。

```powershell
.venv-train/Scripts/python.exe -B scripts/night_joint.py quality --config-sha256 '<quality.json SHA-256>' --scratch D:/Temp
.venv-train/Scripts/python.exe -B scripts/night_joint.py verify-quality --config-sha256 '<quality.json SHA-256>' --manifest-sha256 '<hard_004 SHA-256>'
```

[`distill.json`](distill.json) 锁定 [白天回放蒸馏](../specs/s6/hard.md#白天回放蒸馏)，先在白天训练子集验证冻结教师信号，再运行同名单处理臂并封存 `hard_005`；历史原损失控制只读复用。

```powershell
.venv-train/Scripts/python.exe -B scripts/night_joint.py distill --config-sha256 '<distill.json SHA-256>' --scratch D:/Temp
.venv-train/Scripts/python.exe -B scripts/night_joint.py verify-distill --config-sha256 '<distill.json SHA-256>' --manifest-sha256 '<hard_005 SHA-256>'
```

### 夜间局部放大

[`crop.json`](crop.json) 绑定 [训练局部放大](../specs/s7/plan.md#夜间局部放大支持)的几何收据与固定原平衡臂；原白天回放与开发推理不变。参数处理与配比、车型分差排他。

```powershell
.venv-train/Scripts/python.exe -B scripts/night_expand.py run --arm e2 --config-sha256 '<expand.json SHA-256>' --crop-sha256 '<crop.json SHA-256>' --d300 runs/expand_001 --d300-sha256 '<随机封存清单 SHA-256>' --scratch D:/Temp --output runs/expand_007
.venv-train/Scripts/python.exe -B scripts/night_expand.py verify --output runs/expand_007
```

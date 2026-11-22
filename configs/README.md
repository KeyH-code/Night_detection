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

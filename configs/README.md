# 配置

- `soda10m_categories.json`：官方1..6到训练0..5的映射；pedestrian、car为主指标，cyclist不并入pedestrian。当前冻结状态以数据READY为准。
- `source_a.yaml`：A的复用配方。dataset限定source_train/source_dev；environment限定Python和框架版本；model限定COCO初始化来源；training为实际训练参数；selection为主类索引、严格改进和checkpoint周期；smoke为小样本验证；cloud为资源下限。

原始A使用YOLO11m/640、seed17、AMP、SGD、最多100轮、patience20、batch16、workers4，配方完整值见YAML。每轮以pedestrian/car AP50:95宏平均选模；并列保留较早best。正式最终值见`../runs/a_001/train/audit_initial/`，历史配置快照在`../runs/a_001/code/configs/source_a.yaml`。

显式CLI值和库默认值可能改变实际行为，不能仅凭YAML宣称参数生效。新实验必须记录传入参数、最终trainer参数和原始库默认值。改配置应创建新实验身份，不修改历史快照或冻结数据。当前A的预检仅允许OOM时batch16转8；不能把本机smoke的batch2当正式配置。

## 验收

后续A候选的六类质量、保护、证据、选模与停止标准统一见[规则](../RULES.md)，阶段安排见[计划](../plan.md)。本文不重复阈值。

当前`source_a.yaml`和活动训练入口仍使用原来的两主类宏平均严格提升选模及100轮/patience20配方。新增规则尚未实现为六类资格筛选、受约束PR工作点或有意义改进的早停逻辑；S3必须先完成实现/验证、新身份与云预检，再按实际安排训练。本文发布不修改YAML、原A代码快照或任何权重。

## 观察

monitor.enabled=true、every=1表示以后preflight/formal每轮额外对完整source_train的当轮EMA做无增强AP/P/R评价；batch16、workers0仅用于这个独立评价加载器，梯度训练仍batch16/workers4。原生train/val loss每轮已记录，不需补造。train_metrics.csv含六类AP50/AP50:95/P/R、同轮dev AP及耗时；train_monitor.json绑定身份。观察只写日志，不参与best/早停，不更新参数；不静默改抽样或频率。

当前活动配置增加了观察，原A当时的配置仍以runs/a_001/code快照为准；云端必须为新代码/配置重做预检与配方冻结，将评价的真实时间计入预算。当前没有启动新训练。

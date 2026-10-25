---
version: alpha
name: "夜间道路数据复核台"
description: "用原始图片、官方标签与来源链支持人工抽样核对的本地工具"
colors:
  primary: "#175A78"
  background: "#EDF3F5"
  surface: "#FFFFFF"
  text: "#182F3C"
  muted: "#49606E"
  border: "#CBD8DF"
  success: "#256148"
  warning: "#885A15"
typography:
  sans:
    fontFamily: '"Microsoft YaHei", "Segoe UI", sans-serif'
  display:
    fontFamily: 'Bahnschrift, "Microsoft YaHei", sans-serif'
  mono:
    fontFamily: 'Consolas, "Courier New", monospace'
rounded:
  DEFAULT: "0.5rem"
spacing:
  section-gap: "1.5rem"
  page-max: "96rem"
components:
  button:
    rounded: "{rounded.DEFAULT}"
    backgroundColor: "{colors.primary}"
    textColor: "{colors.surface}"
    typography: "{typography.sans}"
  image-card:
    rounded: "{rounded.DEFAULT}"
    backgroundColor: "{colors.surface}"
    textColor: "{colors.text}"
  caption:
    textColor: "{colors.muted}"
    typography: "{typography.mono}"
  heading:
    typography: "{typography.display}"
  source-state:
    textColor: "{colors.success}"
  caveat:
    textColor: "{colors.warning}"
  divider:
    backgroundColor: "{colors.border}"
  page:
    backgroundColor: "{colors.background}"
---

# 数据复核台

## Overview

2026-10-25 相似场景复核更新：昼夜沿用官方 period，时段视图改为只读。首页默认本轮 50 个待确认新组；已有 12 组独立只读展示，不挤占本轮名额。沿用现有色彩、字体、来源凭条和自然滚动，新增的签名交互为两张原图并排对照，可切换组内最接近／哈希差异最大的配对，并展开完整组。

面向本项目作者，在 Windows 本机查看 SODA10M 来源、昼夜标签和相似组。依据当前用户要求与 `plan.md`，本界面只记录审核意见，不修改原始数据、标签或实验划分。

这是数据工具。视觉参考为图像质检台：图片占主要面积，每张图紧邻不可省略的来源条。用蓝灰背景、白色工作面、深青色操作，避免营销式大标题和装饰性统计卡。没有既有应用或兄弟页面；三个审核视图共享同一套图片卡、按钮与状态反馈。

签名元素是每张图片下的“来源凭条”：官方 split、图片 ID、原文件名、原始 period，保证看图和核对来源在同一位置完成。简体中文界面，技术文件名保留原文。仅本机研究使用，不发布到 Sites 托管。

## Colors

运行时唯一值源为 `review_app/styles.css` 的 `:root` CSS 变量；本文件镜像这些值。`--color-primary` 用于当前导航、焦点与主要操作；`--color-text`、`--color-muted` 表示文本层级；`--color-success` 仅表示来源校验通过，不表示标签正确。`--color-warning` 表示待人工复核。状态同时提供文字。无主题切换，系统高对比模式保留可操作性。

## Typography

标题使用 display 字体，正文使用 sans，文件名和校验值使用 mono；正文 16px、1.65 行高。长文件名换行，不省略影响溯源的信息。不加载外部字体。

## Layout

最大宽度 96rem，桌面图片卡三列，1000px 以下两列，640px 以下一列。图片固定 16:9 占位；页面自然滚动。来源详情为单列阅读面；相似组按组分页，每组初始显示最多六张，显式展开其余图片。过滤后页码归零，翻页保留过滤与审核结果。

## Elevation & Depth

白色工作面与细边框建立层级，不使用复杂阴影。没有模态弹窗；原图通过同源链接在新页打开。

## Shapes

按钮和工作面统一 0.5rem 圆角。截图本身保持矩形完整视野，不裁掉道路画面。

## Components

本轮相似组采用“几乎相同／同地点，有变化／仅类型相似／组内混杂／无法判断”五种人工判断，可填写并保存中文说明。旧 similar/scene 结果保留原义，不自动变成新的去重判定；本轮进度只统计相同 campaign 的新判断。未选择始终是未审核，不解释为通过。随机抽样与风险覆盖的区别、各组入选原因在页面可见。

原生 button、a、label、select 和 input 是共享控件。Select 为 native，接受 Windows 平台弹出菜单的外观与键盘行为；日期选择、Table Selection 和模态框不适用。

导航使用链接并保持 URL 状态；搜索为本地过滤，支持中文输入法和显式清除；按钮有 hover、focus-visible、active 与 disabled 状态。所有反馈通过共享 `role=status` 区域，并在相关控件旁保留关键说明。

图像卡的 loading、解码失败、来源校验失败都保留 16:9 几何。审核采用按钮组，并通过 aria-pressed 表明选择。审核草稿保存在当前浏览器，导出 JSON 包含数据指纹；不能把 Agent 初审标记当作用户判断。空记录导出按钮禁用。

无装饰性动画，降低动态效果偏好下不平滑滚动。图像来源字段、组内边距离和分层抽样说明使用正文解释，不只用颜色编码。

## Do's and Don'ts

- Do：将来源对齐和语义标签正确性分开呈现。
- Do：在随机覆盖样本和有意挑出的可疑样本之间明确标注抽样方式。
- Don't：把候选连通组说成同车同日 sequence，或说组内每对图片都满足阈值。
- Don't：自动改标、删除原图、上传图片或自动把人工审核意见用于训练。

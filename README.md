# DHSA 增强 RT-DETR 课堂行为/表情分析

[English](./README_EN.md) | 中文

本仓库包含论文对应的模型侧实现与实验证据，面向课堂多模态学习分析场景。

核心贡献是：在 RT-DETR 基线结构上引入 DHSA（Dynamic-range Histogram Self-Attention）模块，用于学生行为与表情识别。

## 核心改进（供评审快速定位）

### 相比 RT-DETR 基线做了什么改进

- 在 RT-DETR 编码器相关路径中引入 **DHSA 注意力机制**。
- DHSA 思路来自 Histoformer，并结合本项目场景完成适配与集成。
- 目标是提升课堂视频中小目标、遮挡、光照变化等情况下的特征表达能力。

### DHSA 代码位置

- DHSA 主要实现：`ultralytics/nn/extra_modules/transformer.py`
- 模型构建相关路径：`ultralytics/nn/tasks.py`
- RT-DETR 配置目录：`ultralytics/cfg/models/rt-detr`

DHSA 原始参考：
- [Histoformer](https://github.com/sunshangquan/Histoformer)

### 架构示意图

![DHSA-enhanced RT-DETR](./改进RT-DETR.png)

## 可复现实验证据

### 数据配置文件

- `dataset/action-1-721.yaml`（行为任务）
- `dataset/exp-1-721.yaml`（表情任务）

因隐私与伦理要求，原始课堂视频和标注不公开。

### 训练/评估日志

- `runs/train/.../results.csv`
- `runs/train/.../args.yaml`

用于展示训练参数、收敛过程与指标结果。

## 推理测试脚本（工程补充）

仓库中新增了可直接用于测试的小项目脚本：

- `classroom_dual_model_sampler.py`

流程：
- 第 1 阶段：整帧人体检测；
- 第 2 阶段：按人体框裁剪并分别做行为/表情识别；
- 每 30 帧采样一次（可配置）；
- 每 30 秒聚合一次类别百分比（可配置）；
- 支持导出每次采样的人体框可视化图片。

测试文档：
- 中文：`TESTING_GUIDE_CN.md`
- English: `TESTING_GUIDE_EN.md`

## 给论文评审的说明

本仓库基于 Ultralytics 框架，重点保留 DHSA 改进实现与实验证据，便于评审核验方法贡献与可复现性。

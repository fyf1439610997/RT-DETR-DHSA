# RT-DETR-DHSA 独立测试说明

这个目录用于独立测试视频目标检测，不会改动 `MIDO-Chat` 主项目代码。

## 1) 环境准备

在 `RT-DETR-DHSA` 目录下执行：

```powershell
python -m venv .venv
.\.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

> 如果你是 CPU 测试，把最后一行改成官方 CPU 版本安装命令即可。

## 2) 三模型两阶段采样统计脚本

已新增脚本：`classroom_dual_model_sampler.py`

能力：
- 第 1 阶段：人体检测模型在整帧检测学生框
- 第 2 阶段：对每个学生框做行为 + 表情识别
- 每 `30` 帧采样一次（可改）
- 每 `30` 秒输出一次全班统计（可改）
- 输出 `CSV + JSON + 运行元数据`
- 可输出每次采样的人体框可视化图片

## 3) 运行示例

默认不需要传任何参数，直接运行：

```powershell
python classroom_dual_model_sampler.py
```

参数统一在脚本顶部 `CONFIG` 区域修改：
- 文件：`classroom_dual_model_sampler.py`
- 位置：文件开头注释 `Local fixed configuration` 下方
- 常改项：`video`、`person_weights`、`behavior_weights`、`expression_weights`

如需临时覆盖，也可以继续用命令行参数：

```powershell
python classroom_dual_model_sampler.py `
  --video .\demo\classroom.mp4 `
  --person-weights .\weights\yolov8n.pt `
  --behavior-weights .\weights\behavior_best.pt `
  --expression-weights .\weights\expression_best.pt `
  --sample-every-frames 30 `
  --window-seconds 30 `
  --person-class-id 0 `
  --person-conf 0.25 `
  --action-conf 0.25 `
  --behavior-labels "listen,raise_hand,write" `
  --expression-labels "neutral,happy,confused" `
  --output-dir .\runs\classroom_stats
```

## 4) 输出文件

- `runs/classroom_stats/classroom_30s_stats.csv`
- `runs/classroom_stats/classroom_30s_stats.json`
- `runs/classroom_stats/run_meta.json`
- `runs/classroom_stats/person_boxes/*.jpg`（每次采样一张，带人体框）

其中每个 30s 窗口会包含：
- 时间范围（开始秒、结束秒）
- 该窗口内被采样的帧数
- 人体检测总框数（该窗口内）
- 行为类别百分比（3 类加和 = 100%）
- 表情类别百分比（3 类加和 = 100%）

## 5) 注意事项

- 目前统计的是“按学生框得到的分类百分比汇总”，用于快速验证两阶段流程可用性。
- 若后续你希望“按学生 ID 去重统计”，可在此脚本上接入 tracking（如 ByteTrack）再做每 30s 统计。

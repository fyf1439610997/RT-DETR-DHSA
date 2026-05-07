# RT-DETR-DHSA Standalone Testing Guide

English | [中文](./TESTING_GUIDE_CN.md)

This directory is for standalone video detection testing and does not modify the main `MIDO-Chat` project code.

## 1) Environment Setup

Run in the `RT-DETR-DHSA` directory:

```powershell
python -m venv .venv
.\.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
```

If you want CPU-only testing, replace the last command with the official CPU wheels.

## 2) Three-model, Two-stage Sampling Script

Script:
- `classroom_dual_model_sampler.py`

Capabilities:
- Stage 1: detect student person boxes on the full frame.
- Stage 2: classify behavior + expression on each person crop.
- Sample every 30 frames (configurable).
- Aggregate whole-class statistics every 30 seconds (configurable).
- Export CSV + JSON + run metadata.
- Export person-box visualization images for each sampled frame.

## 3) Run

By default, no CLI arguments are required:

```powershell
python classroom_dual_model_sampler.py
```

Modify parameters in the `CONFIG` section at the top of:
- `classroom_dual_model_sampler.py`

Commonly updated fields:
- `video`
- `person_weights`
- `behavior_weights`
- `expression_weights`
- `device`

## 4) Output Files

- `test_output/classroom_stats/classroom_30s_stats.csv`
- `test_output/classroom_stats/classroom_30s_stats.json`
- `test_output/classroom_stats/run_meta.json`
- `test_output/classroom_stats/person_boxes/*.jpg`

Each 30s window includes:
- start/end time range
- sampled frame count
- total detected person boxes
- behavior percentages (3 classes sum to 100% when valid samples exist)
- expression percentages (3 classes sum to 100% when valid samples exist)

## 5) Notes

- Current statistics are percentage summaries from person-crop predictions.
- If you need per-student de-duplicated statistics, add tracking (e.g., ByteTrack) before 30-second aggregation.

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import cv2
from ultralytics import RTDETR
from ultralytics import YOLO

# =========================
# Local fixed configuration
# Edit parameters here only.
# =========================
CONFIG = {
    # I/O
    "video": r"./test/classroom_1min.mp4",
    "person_weights": r"./weights/yolov8n.pt",
    "behavior_weights": r"./weights/RTDETR-DHSA-B.pt",
    "expression_weights": r"./weights/RTDETR-DHSA-E.pt",
    "output_dir": r"test_output/classroom_stats",
    # Sampling and aggregation
    "sample_every_frames": 30,
    "window_seconds": 30,
    # Detection thresholds
    "person_conf": 0.20,
    "action_conf": 0.25,
    # Person detector settings
    "person_class_id": 0,
    "min_person_box": 12,
    "person_imgsz": 1280,
    "person_iou": 0.60,
    # Person box visualization
    "save_person_vis": True,
    "person_vis_dirname": "person_boxes",
    # Label mapping
    "behavior_labels": "High,Medium,Low",
    "expression_labels": "Neutral,Negative,Positive",
    # Inference device: "0", "cpu", or None
    "device": "0",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Two-stage classroom analytics: detect persons on full frame, then classify "
            "behavior/expression on each person crop."
        )
    )
    parser.add_argument("--video", type=str, default=CONFIG["video"], help="Input classroom video path.")
    parser.add_argument(
        "--person-weights",
        type=str,
        default=CONFIG["person_weights"],
        help="Person detector weights (.pt).",
    )
    parser.add_argument(
        "--behavior-weights",
        type=str,
        default=CONFIG["behavior_weights"],
        help="Behavior model weights (.pt).",
    )
    parser.add_argument(
        "--expression-weights",
        type=str,
        default=CONFIG["expression_weights"],
        help="Expression model weights (.pt).",
    )
    parser.add_argument("--output-dir", type=str, default=CONFIG["output_dir"], help="Output directory.")
    parser.add_argument(
        "--sample-every-frames",
        type=int,
        default=CONFIG["sample_every_frames"],
        help="Infer once every N frames.",
    )
    parser.add_argument(
        "--window-seconds",
        type=int,
        default=CONFIG["window_seconds"],
        help="Aggregate counts every M seconds.",
    )
    parser.add_argument(
        "--person-conf",
        type=float,
        default=CONFIG["person_conf"],
        help="Person detector confidence threshold.",
    )
    parser.add_argument(
        "--action-conf",
        type=float,
        default=CONFIG["action_conf"],
        help="Behavior and expression detector confidence threshold.",
    )
    parser.add_argument(
        "--person-class-id",
        type=int,
        default=CONFIG["person_class_id"],
        help="Class id of person in person detector labels (COCO person is 0).",
    )
    parser.add_argument(
        "--min-person-box",
        type=int,
        default=CONFIG["min_person_box"],
        help="Ignore person boxes whose width/height are both smaller than this value.",
    )
    parser.add_argument(
        "--person-imgsz",
        type=int,
        default=CONFIG["person_imgsz"],
        help="Person detector input size, e.g. 640/960/1280.",
    )
    parser.add_argument(
        "--person-iou",
        type=float,
        default=CONFIG["person_iou"],
        help="NMS IoU threshold for person detector.",
    )
    parser.add_argument(
        "--save-person-vis",
        action=argparse.BooleanOptionalAction,
        default=CONFIG["save_person_vis"],
        help="Save person-box visualization for each sampled frame.",
    )
    parser.add_argument(
        "--person-vis-dirname",
        type=str,
        default=CONFIG["person_vis_dirname"],
        help="Subfolder name under output-dir for sampled person-box images.",
    )
    parser.add_argument(
        "--behavior-labels",
        type=str,
        default=CONFIG["behavior_labels"],
        help="Comma-separated class names for behavior model.",
    )
    parser.add_argument(
        "--expression-labels",
        type=str,
        default=CONFIG["expression_labels"],
        help="Comma-separated class names for expression model.",
    )
    parser.add_argument("--device", type=str, default=CONFIG["device"], help="Device string, e.g. '0', 'cpu'.")
    return parser.parse_args()


def parse_labels(raw: str) -> dict[int, str]:
    labels = [x.strip() for x in raw.split(",") if x.strip()]
    return {i: name for i, name in enumerate(labels)}


def detect_person_boxes(
    person_model: YOLO,
    frame,
    conf: float,
    iou: float,
    imgsz: int,
    person_class_id: int,
    min_person_box: int,
    device: str | None,
) -> list[tuple[int, int, int, int]]:
    result = person_model.predict(
        source=frame,
        conf=conf,
        iou=iou,
        imgsz=imgsz,
        classes=[person_class_id],
        verbose=False,
        device=device,
    )[0]
    boxes = result.boxes
    if boxes is None or boxes.xyxy is None:
        return []

    h, w = frame.shape[:2]
    person_boxes: list[tuple[int, int, int, int]] = []
    for xyxy in boxes.xyxy.cpu().tolist():
        x1, y1, x2, y2 = [int(v) for v in xyxy]
        x1 = max(0, min(x1, w - 1))
        y1 = max(0, min(y1, h - 1))
        x2 = max(0, min(x2, w))
        y2 = max(0, min(y2, h))
        bw = x2 - x1
        bh = y2 - y1
        if bw <= 0 or bh <= 0:
            continue
        if bw < min_person_box and bh < min_person_box:
            continue
        person_boxes.append((x1, y1, x2, y2))
    return person_boxes


def predict_top1_class(model: RTDETR, image, conf: float, device: str | None) -> int | None:
    result = model.predict(source=image, conf=conf, verbose=False, device=device)[0]
    cls_tensor = result.boxes.cls if result.boxes is not None else None
    conf_tensor = result.boxes.conf if result.boxes is not None else None
    if cls_tensor is None or conf_tensor is None or len(cls_tensor) == 0:
        return None

    best_idx = int(conf_tensor.argmax().item())
    return int(cls_tensor[best_idx].item())


def decode_counts(counts: Counter, labels: dict[int, str]) -> dict[str, int]:
    decoded = {}
    for idx, count in sorted(counts.items(), key=lambda x: x[0]):
        decoded[labels.get(idx, f"class_{idx}")] = int(count)
    return decoded


def convert_counts_to_percentages(counts: Counter, labels: dict[int, str]) -> dict[str, float]:
    ordered_indices = sorted(labels.keys())
    label_names = [labels[idx] for idx in ordered_indices]
    values = [int(counts.get(idx, 0)) for idx in ordered_indices]
    total = sum(values)

    if total <= 0:
        return {name: 0.0 for name in label_names}

    raw = [(v * 100.0) / total for v in values]
    rounded = [round(x, 2) for x in raw]
    diff = round(100.0 - sum(rounded), 2)
    if rounded:
        max_idx = max(range(len(values)), key=lambda i: values[i])
        rounded[max_idx] = round(rounded[max_idx] + diff, 2)
    return {name: rounded[i] for i, name in enumerate(label_names)}


def write_csv(rows: list[dict], path: Path) -> None:
    fieldnames = [
        "window_index",
        "start_second",
        "end_second",
        "sampled_frames",
        "detected_person_boxes",
        "behavior_percentages_json",
        "expression_percentages_json",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def save_person_visualization(
    frame,
    person_boxes: list[tuple[int, int, int, int]],
    frame_idx: int,
    second: float,
    output_path: Path,
) -> None:
    vis = frame.copy()
    for x1, y1, x2, y2 in person_boxes:
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 255), 2)
    label = f"frame={frame_idx} sec={second:.2f} persons={len(person_boxes)}"
    cv2.putText(vis, label, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.imwrite(str(output_path), vis)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    person_vis_dir = output_dir / args.person_vis_dirname
    if args.save_person_vis:
        person_vis_dir.mkdir(parents=True, exist_ok=True)

    video_path = Path(args.video)
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    behavior_labels = parse_labels(args.behavior_labels)
    expression_labels = parse_labels(args.expression_labels)

    person_model = YOLO(args.person_weights)
    behavior_model = RTDETR(args.behavior_weights)
    expression_model = RTDETR(args.expression_weights)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    window_frames = max(1, int(args.window_seconds * fps))
    sample_every = max(1, args.sample_every_frames)

    frame_idx = -1
    window_idx = 0
    behavior_window_counter: Counter = Counter()
    expression_window_counter: Counter = Counter()
    samples_in_window = 0
    people_in_window = 0
    rows = []
    json_windows = []

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_idx += 1

        if frame_idx % sample_every == 0:
            person_boxes = detect_person_boxes(
                person_model=person_model,
                frame=frame,
                conf=args.person_conf,
                iou=args.person_iou,
                imgsz=args.person_imgsz,
                person_class_id=args.person_class_id,
                min_person_box=args.min_person_box,
                device=args.device,
            )
            people_in_window += len(person_boxes)
            if args.save_person_vis:
                sec = frame_idx / fps if fps > 0 else 0.0
                vis_path = person_vis_dir / f"frame_{frame_idx:07d}_sec_{sec:08.2f}.jpg"
                save_person_visualization(frame, person_boxes, frame_idx, sec, vis_path)

            for x1, y1, x2, y2 in person_boxes:
                crop = frame[y1:y2, x1:x2]
                if crop.size == 0:
                    continue

                behavior_cls = predict_top1_class(behavior_model, crop, args.action_conf, args.device)
                expression_cls = predict_top1_class(expression_model, crop, args.action_conf, args.device)
                if behavior_cls is not None:
                    behavior_window_counter.update([behavior_cls])
                if expression_cls is not None:
                    expression_window_counter.update([expression_cls])

            samples_in_window += 1

        is_window_end = ((frame_idx + 1) % window_frames == 0)
        is_last_frame = total_frames > 0 and (frame_idx + 1 >= total_frames)
        if is_window_end or is_last_frame:
            start_second = round((window_idx * window_frames) / fps, 2)
            end_second = round((frame_idx + 1) / fps, 2)
            behavior_pct = convert_counts_to_percentages(behavior_window_counter, behavior_labels)
            expression_pct = convert_counts_to_percentages(expression_window_counter, expression_labels)
            row = {
                "window_index": window_idx,
                "start_second": start_second,
                "end_second": end_second,
                "sampled_frames": samples_in_window,
                "detected_person_boxes": people_in_window,
                "behavior_percentages_json": json.dumps(behavior_pct, ensure_ascii=False),
                "expression_percentages_json": json.dumps(expression_pct, ensure_ascii=False),
            }
            rows.append(row)
            json_windows.append(
                {
                    "window_index": window_idx,
                    "start_second": start_second,
                    "end_second": end_second,
                    "sampled_frames": samples_in_window,
                    "detected_person_boxes": people_in_window,
                    "behavior_percentages": behavior_pct,
                    "expression_percentages": expression_pct,
                }
            )
            window_idx += 1
            behavior_window_counter = Counter()
            expression_window_counter = Counter()
            samples_in_window = 0
            people_in_window = 0

    cap.release()

    csv_path = output_dir / "classroom_30s_stats.csv"
    json_path = output_dir / "classroom_30s_stats.json"
    meta_path = output_dir / "run_meta.json"

    write_csv(rows, csv_path)
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(json_windows, f, ensure_ascii=False, indent=2)
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "video": str(video_path),
                "fps": fps,
                "total_frames": total_frames,
                "sample_every_frames": sample_every,
                "window_seconds": args.window_seconds,
                "window_frames": window_frames,
                "person_weights": args.person_weights,
                "behavior_weights": args.behavior_weights,
                "expression_weights": args.expression_weights,
                "person_conf": args.person_conf,
                "action_conf": args.action_conf,
                "person_class_id": args.person_class_id,
                "min_person_box": args.min_person_box,
                "person_imgsz": args.person_imgsz,
                "person_iou": args.person_iou,
                "save_person_vis": args.save_person_vis,
                "person_vis_dirname": args.person_vis_dirname,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(f"Done. CSV: {csv_path}")
    print(f"Done. JSON: {json_path}")
    print(f"Done. Meta: {meta_path}")


if __name__ == "__main__":
    main()

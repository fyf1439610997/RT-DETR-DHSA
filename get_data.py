import json
import cv2
import pandas as pd
from ultralytics import RTDETR

# 定义从 Labelme JSON 文件读取 student_areas 的函数
def load_student_areas(json_file):
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    student_areas = []
    for shape in data['shapes']:
        label = shape['label']  # 学生 ID
        points = shape['points']  # 标注框的顶点坐标 [[x1, y1], [x2, y2]]
        x1, y1 = points[0]
        x2, y2 = points[1]

        # 确保坐标顺序正确 (x1, y1) 是左上角，(x2, y2) 是右下角
        x1, x2 = min(x1, x2), max(x1, x2)
        y1, y2 = min(y1, y2), max(y1, y2)

        student_areas.append((x1, y1, x2, y2, label))

    return student_areas

# 加载训练好的 RTDETR 模型
model = RTDETR('runs/train/exp-1-721/rtdetr-AIFI-DHSA/weights/best.pt')

# 视频路径
video_path = "/root/autodl-fs/711.mp4"
cap = cv2.VideoCapture(video_path)
# 视频信息
fps = int(cap.get(cv2.CAP_PROP_FPS))

# 定义每个学生的固定区域（手动定义）
# 格式为 [(x1, y1, x2, y2, "Student_ID")]
json_file = "/root/autodl-fs/20.json"
student_areas = load_student_areas(json_file)

# 创建一个字典存储每个学生的数据
student_data = {area[4]: [] for area in student_areas}

# 遍历视频帧
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    # 获取当前帧号和时间戳
    frame_number = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
    timestamp = frame_number / fps

    if frame_number%30 == 0:
        # 遍历每个学生区域
        for x1, y1, x2, y2, student_id in student_areas:
            # 裁剪出当前学生区域的图片
            student_frame = frame[int(y1):int(y2), int(x1):int(x2)]
    
            # 使用模型检测
            results = model(student_frame)
    
            # 获取检测结果（只保留置信度最高的行为）
            expression = "Unknown"
            confidence = 0
            for result in results[0].boxes.data:
                cls_name = model.names[int(result[5])]  # 行为类别
                conf = result[4].item()  # 置信度
                if conf > confidence:
                    expression = cls_name
                    confidence = conf
    
            # 记录结果
            student_data[student_id].append({
                "Timestamp (s)": timestamp,
                "Frame Number": frame_number,
                "Expression": expression,
                "Confidence": round(confidence, 2)
            })

# 释放视频资源
cap.release()

# 保存结果到 Excel 的不同工作簿
output_file = "/root/autodl-fs/student_expression_log.xlsx"
with pd.ExcelWriter(output_file) as writer:
    for student_id, data in student_data.items():
        df = pd.DataFrame(data)
        df.to_excel(writer, sheet_name=student_id, index=False)

print(f"结果已保存到 {output_file}")
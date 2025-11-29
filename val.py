import warnings

warnings.filterwarnings("ignore")
from ultralytics import RTDETR

# 最终论文的参数量和计算量统一以这个脚本运行出来的为准
gaijin = "rtdetr-r50"
dataset = "action-1-721"

if __name__ == "__main__":
    model = RTDETR(f"runs/train/{dataset}/{gaijin}/weights/best.pt")
    model.val(
        data=f"dataset/{dataset}.yaml",
        split="test",  # split可以选择train、val、test 根据自己的数据集情况来选择.
        imgsz=640,
        batch=16,
        #   save_json=True, # if you need to cal coco metrice
        project=f"runs/val/{dataset}",
        name=gaijin,
    )

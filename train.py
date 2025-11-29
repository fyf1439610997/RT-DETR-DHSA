import warnings

warnings.filterwarnings("ignore")
from ultralytics import RTDETR

gaijin = "rtdetr-l"
dataset = "exp-1-721"

if __name__ == "__main__":
    model = RTDETR(f"ultralytics/cfg/models/rt-detr/{gaijin}.yaml")
    model.load("weights/rtdetr-r18.pt")  # loading pretrain weights
    model.train(
        data=f"dataset/{dataset}.yaml",
        cache=False,
        imgsz=640,
        epochs=300,
        batch=16,
        workers=4,  # Windows下出现莫名其妙卡主的情况可以尝试把workers设置为0
        device="0",  # 指定显卡和多卡训练问题 统一都在<使用说明.md>下方常见错误和解决方案。
        # resume='', # last.pt path
        project=f"runs/train/{dataset}",
        name=gaijin,
    )

# ProtoHGF-Net

Official implementation of **ProtoHGF-Net: Prototype HyperGraph Fusion with Intra-modal Calibration for RGBT Object Detection** (ACM MM 2026).

论文链接：https://doi.org/10.1145/3767308.3835030

## 环境

原始实验环境：

- Python 3.12.3
- PyTorch 2.3.0+cu121
- torchvision 0.18.0+cu121
- torchaudio 2.3.0+cu121
- CUDA 12.1

```bash
conda create -n protohgf python=3.12 -y
conda activate protohgf
pip install torch==2.3.0+cu121 torchvision==0.18.0+cu121 \
  torchaudio==2.3.0+cu121 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

PyTorch 与 torchvision 必须成套安装，否则验证阶段的 `torchvision.ops.nms` 会报 C++ 算子不兼容。

## 数据组织

可见光和红外图像必须逐像素对齐、文件名相同，并分别放在同级的 `visible` 与 `infrared` 目录中。数据 YAML 的 `train`、`val` 只填写 visible 路径，加载器会自动定位对应 infrared 图像。

```text
dataset/
├── images/
│   ├── visible/train/000001.jpg
│   ├── visible/val/000002.jpg
│   ├── infrared/train/000001.jpg
│   └── infrared/val/000002.jpg
└── labels/
    └── visible/
        ├── train/000001.txt
        └── val/000002.txt
```

修改以下配置中的 `path`：

- `ultralytics/cfg/datasets/DVTOD.yaml`
- `ultralytics/cfg/datasets/FLIR_aligned.yaml`
- `ultralytics/cfg/datasets/DroneVehicle.yaml`

## 训练

TM-Calib 需要两个类别顺序一致的 YOLOv8m 单模态教师权重；训练时教师保持冻结。
请自行准备教师权重，并通过命令行传入。

```bash
# DVTOD（HBB）
CUDA_VISIBLE_DEVICES=0 python train_protohgf.py \
  --dataset dvtod \
  --rgb-teacher runs/DVTOD_RGB/weights/best.pt \
  --thermal-teacher runs/DVTOD_IR/weights/best.pt \
  --device 0

# DroneVehicle（OBB）
CUDA_VISIBLE_DEVICES=0 python train_protohgf.py \
  --dataset dronevehicle \
  --rgb-teacher runs/DroneVehicle_RGB/weights/best.pt \
  --thermal-teacher runs/DroneVehicle_IR/weights/best.pt \
  --device 0
```

FLIR 使用 `--dataset flir`。可用 `--data`、`--model`、`--epochs`、`--batch`、`--imgsz` 覆盖默认配置。当前 TM-Calib 训练支持单进程、单 GPU。

## DVTOD 最终评估

```bash
python val_RGFNet.py runs/ProtoHGF-Net/dvtod/weights/best.pt \
  --data ultralytics/cfg/datasets/DVTOD.yaml \
  --project runs/val_RGFNet --name dvtod --device 0
```

训练期间默认验证阈值不同，比较结果时应使用同一套最终评估设置。

## License 与引用

代码沿用 Ultralytics 的 AGPL-3.0 许可。使用本项目时请引用论文，完整元数据见 `CITATION.cff`。

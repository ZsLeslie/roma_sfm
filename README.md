# roma_sfm — RoMa Outdoor 批量图像匹配（阶段 1）

本目录是独立项目，放在官方 `RoMa` 仓库的**同级**目录，不修改官方代码。当前仅输出逐对匹配的 NPZ 与统计，不会写入 COLMAP 数据库，也不构造跨图像的关键点轨迹。

## 目录

```text
~/src/
  RoMa/             # 官方仓库
  roma_sfm/         # 本项目（这 7 个 .py 文件）
```

预训练权重：`~/models/RoMa/roma_outdoor.pth` 与 `~/models/RoMa/dinov2_vitl14_pretrain.pth`。

## 环境

在已经通过单对图像测试的 Jetson PyTorch 环境中运行。**不要重新 `pip install torch`。** 如果 RoMa 尚未以 `-e` 方式安装，请在同一终端执行：

```bash
export PYTHONPATH="$HOME/src/RoMa:${PYTHONPATH:-}"
```

## 运行示例

```bash
cd ~/src/roma_sfm
# 只查看生成的图像对，不加载 GPU 模型
python batch_match.py \
  --images-dir ~/colmap_data/DjiData_1367/images \
  --overlap 5 --max-pairs 30 --dry-run

# 第一次小批量验证：仅前 30 对，保存可视化
python batch_match.py \
  --images-dir ~/colmap_data/DjiData_1367/images \
  --output-dir ~/colmap_data/DjiData_1367/roma_matches \
  --weights-dir ~/models/RoMa \
  --overlap 5 --max-pairs 30 \
  --coarse-res 448 --upsample-res 672 \
  --num-matches 3000 --min-confidence 0.05 \
  --visualize-every 10

# 扩展到全部顺序图像对。参数保持一致，已完成对会跳过。
python batch_match.py \
  --images-dir ~/colmap_data/DjiData_1367/images \
  --output-dir ~/colmap_data/DjiData_1367/roma_matches \
  --weights-dir ~/models/RoMa \
  --overlap 5 --max-pairs 0 \
  --coarse-res 448 --upsample-res 672 \
  --num-matches 3000 --min-confidence 0.05
```

> 可传入自己已有的 `--pairs-file pairs-sequential.txt`，每行 `image0.jpg image1.jpg`，名称相对于 `--images-dir`。使用该选项时不生成顺序对。所有已存在匹配将跳过；`--overwrite` 重算所选图像对。

## 输出

```text
roma_matches/
  run_config.json          # 防止不同模型/阈值结果混在一起
  pairs_used.txt           # 本次命令选择的图像对
  statistics.csv           # 每对图像的处理状态与统计
  batch_match.log          # 日志
  matches/<SHA1前20位>.npz # 每对图像一个结果（原子写入）
  visualizations/*.jpg     # 仅 --visualize-every > 0 时创建
```

NPZ 字段：`image0`、`image1`、`keypoints0`、`keypoints1`、`confidence`、`inliers`、`sampled`、`after_confidence`、`elapsed_s`、`peak_cuda_gb`。其中 `keypoints*` 为原始输入图像像素坐标，`confidence` 是 RoMa 未被 threshold 采样覆盖的 sigmoid 置信度，`inliers` 是对应位置的 MAGSAC 布尔值。

### 关键约束

- 模型初始化一次，图像对逐个处理，适合 Orin NX 16GB；每对采用独立随机种子，断点续跑的采样不依赖前面已完成的图像对。
- `--max-pairs 30` 是按已生成配对顺序取前 30 对，非随机抽样。
- 原始图像若未去畸变，基础矩阵几何验证对大畸变区域可能不够准确。
- `statistics.csv` 中的 RANSAC 内点比例是经过置信度与空间去重后的比例，不是全部 RoMa 稠密预测的内点比例。
- 断点续跑以已存在、原子完成的 NPZ 为依据；非 OOM 的异常默认继续下一对，失败对下次会重试。
- 改动权重、分辨率、匹配数量、阈值等关键参数时，必须使用新的 `--output-dir`，避免结果混用。
- 本阶段不解决 COLMAP 多图像的关键点统一索引和轨迹关联问题。

## 检查一个匹配文件

```bash
python - <<'PY'
import numpy as np
from pathlib import Path
p = next((Path.home()/"colmap_data/DjiData_1367/roma_matches/matches").glob("*.npz"))
with np.load(p, allow_pickle=False) as d:
    print("image pair:", d["image0"], d["image1"])
    print("keypoints0:", d["keypoints0"].shape)
    print("keypoints1:", d["keypoints1"].shape)
    print("conf range:", d["confidence"].min() if len(d["confidence"]) else "empty",
          d["confidence"].max() if len(d["confidence"]) else "empty")
    print("inliers:", int(d["inliers"].sum()))
PY
```

# CIFAR-10 图像分类（当代人工智能·实验三）

基于 PyTorch 的 CIFAR-10 图像分类实验：从零设计并训练自定义 CNN 与 ResNet-20，
对比数据增强、学习率调度、标签平滑等训练策略，并完成混淆矩阵、误分类样本与
Grad-CAM 可解释性分析。

## 文件结构

```
├── model.py      # SimpleCNN（轻量 VGG 风格）与 ResNet-20（CIFAR 版）定义
├── train.py      # 训练入口：数据划分、训练循环、早停、调度器、日志与检查点
├── analysis.py   # 测试集评估：混淆矩阵、每类准确率、训练曲线、
│                 #   误分类样本网格、Grad-CAM 热力图
├── make_report_docx.py   # 实验报告 md -> docx 排版脚本（可选）
├── export_docx_pdf.py    # 用本机 Word 将 docx 导出为 PDF（可选）
├── README.md
└── .gitignore
```

数据集与训练产物不入库：CIFAR-10 由 torchvision 自动下载到 `data/`，训练日志、
检查点与图表分别写入 `results/` 与 `figures/`。

## 实验结果

在 20 核 CPU 上按上述设置复现（每组约 1-2.5 小时）：

| 实验组 | 模型 | 策略 | 验证集最优准确率 | 测试集准确率 |
| --- | --- | --- | --- | --- |
| cnn_baseline | SimpleCNN | 无增强、固定学习率 0.01 | 86.70% | 85.35% |
| cnn_improved | SimpleCNN | 数据增强 + 余弦退火 + 标签平滑 0.1 | 89.74% | **90.15%** |
| resnet20 | ResNet-20 | 同 cnn_improved | 90.06% | 89.96% |

数据增强、余弦退火与标签平滑使同一网络提升 4.8 个百分点，并把训练/验证准确率
差距从约 10.5 个百分点收窄到 4.4 个百分点；ResNet-20 在有限训练预算下与增强后的
SimpleCNN 基本持平。更多分析（混淆矩阵、每类准确率、误分类样本、Grad-CAM）见
实验报告与 `analysis.py`。

## 环境

- Python 3.13，PyTorch 2.12（CPU 即可运行，有 GPU 自动使用）
- 依赖：`torch`, `torchvision`, `numpy`, `matplotlib`

```bash
pip install torch torchvision numpy matplotlib
```

## 快速复现

三组实验依次运行（在 CPU 上总计约需数小时，可减少 `--epochs` 快速验证）：

```bash
# 1) 基础 CNN，无数据增强，固定学习率
python train.py --run cnn_baseline --model simplecnn --epochs 50

# 2) 基础 CNN + 数据增强 + 余弦退火 + 标签平滑 + 早停
python train.py --run cnn_improved --model simplecnn --epochs 80 \
    --augment --scheduler cosine --label-smoothing 0.1 --patience 15

# 3) ResNet-20，策略同 2
python train.py --run resnet20 --model resnet20 --epochs 80 \
    --augment --scheduler cosine --label-smoothing 0.1 --patience 15
```

每个 run 在 `results/<run>/` 下输出 `best.pt`（最佳验证集模型）与
`history.csv`（逐轮 loss/accuracy/lr）。

## 分析与可视化

```bash
python analysis.py --run cnn_baseline --run cnn_improved --run resnet20
```

在 `figures/` 下生成训练曲线对比、各模型混淆矩阵、误分类样本网格与
Grad-CAM 热力图，并输出 `summary.csv` 汇总各项指标。

## 训练设置

- 数据：训练集 45,000 / 验证集 5,000（按类别分层抽样，seed=42），测试集 10,000
- 预处理：ToTensor + 按 CIFAR-10 均值/标准差归一化；
  增强组额外使用随机裁剪（pad=4）与水平翻转
- 优化：SGD（momentum=0.9，weight_decay=5e-4），batch size 128
- 基准组学习率固定 0.01；改进组使用余弦退火 + 标签平滑 0.1 + 早停（patience=15）
- 结果可复现：固定随机种子，测试集仅用于最终评估

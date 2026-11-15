# -*- coding: utf-8 -*-
"""训练后分析：测试集评估、曲线图、混淆矩阵、误分类样本与 Grad-CAM。

用法：
    python analysis.py --run cnn_baseline --run cnn_improved --run resnet20

- 对每个 run 输出：测试集总准确率、每类准确率、混淆矩阵图、
  误分类样本网格图、Grad-CAM 热力图；
- 输出 figures/curves.png：所有 run 的训练/验证 loss 与准确率对比；
- 输出 figures/summary.csv：每个 run 的最终验证/测试指标汇总。
"""

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader

from model import build_model
from train import MEAN, STD

CLASSES = ("airplane", "automobile", "bird", "cat", "deer",
           "dog", "frog", "horse", "ship", "truck")

plt.rcParams["font.family"] = ["Microsoft YaHei", "SimHei", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False


def load_model(run: str, device):
    """加载 best.pt 并解析 run 名中的模型名。"""
    model_name = "resnet20" if run.startswith("resnet") else "simplecnn"
    model = build_model(model_name).to(device)
    ckpt = Path("results") / run / "best.pt"
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval()
    return model


@torch.no_grad()
def predict_all(model, loader, device):
    """返回全体预测 (pred, label) 与逐样本置信度。"""
    preds, labels = [], []
    for images, labs in loader:
        logits = model(images.to(device))
        preds.append(logits.argmax(1).cpu())
        labels.append(labs)
    return torch.cat(preds), torch.cat(labels)


def test_report(model, run: str, device):
    """测试集评估：总准确率、每类准确率、混淆矩阵，保存到 figures/。"""
    test_ds = torchvision.datasets.CIFAR10(
        "data", train=False, transform=T.Compose([T.ToTensor(),
                                                  T.Normalize(MEAN, STD)]))
    loader = DataLoader(test_ds, batch_size=256, num_workers=0)
    preds, labels = predict_all(model, loader, device)

    acc = (preds == labels).float().mean().item()
    n = len(CLASSES)
    per_class, conf = [], np.zeros((n, n), dtype=int)
    for i in range(n):
        mask = labels == i
        per_class.append((preds[mask] == i).float().mean().item())
    for t, p in zip(labels.tolist(), preds.tolist()):
        conf[t][p] += 1

    # 混淆矩阵（像素值，右侧加颜色条）
    fig, ax = plt.subplots(figsize=(5.2, 4.4), dpi=150)
    im = ax.imshow(conf, cmap="Blues")
    ax.set_xticks(range(n), CLASSES, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(n), CLASSES, fontsize=8)
    for i in range(n):
        for j in range(n):
            ax.text(j, i, conf[i, j], ha="center", va="center", fontsize=6,
                    color="white" if conf[i, j] > conf.max() / 2 else "black")
    ax.set_xlabel("预测类别")
    ax.set_ylabel("真实类别")
    ax.set_title(f"{run} 混淆矩阵（测试集）", fontsize=10)
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(Path("figures") / f"confusion_{run}.png")
    plt.close(fig)

    with open(Path("results") / run / "per_class.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["class", "accuracy"])
        w.writerows(zip(CLASSES, per_class))
    return acc, per_class


# ------------------------------------------------------------------ 曲线图

def plot_curves(runs):
    """所有 run 的训练/验证 loss 与准确率曲线对比，保存 figures/curves.png。"""
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.3), dpi=150)
    colors = plt.cm.tab10(np.linspace(0, 1, len(runs)))
    for run, c in zip(runs, colors):
        hist = np.genfromtxt(Path("results") / run / "history.csv",
                             delimiter=",", names=True)
        axes[0].plot(hist["epoch"], hist["train_loss"], color=c,
                     linestyle="--", label=f"{run} train")
        axes[0].plot(hist["epoch"], hist["val_loss"], color=c,
                     label=f"{run} val")
        axes[1].plot(hist["epoch"], hist["train_acc"], color=c,
                     linestyle="--", label=f"{run} train")
        axes[1].plot(hist["epoch"], hist["val_acc"], color=c,
                     label=f"{run} val")
    for ax, ylab, title in zip(axes, ("loss", "accuracy"),
                               ("训练/验证损失曲线", "训练/验证准确率曲线")):
        ax.set_xlabel("epoch")
        ax.set_ylabel(ylab)
        ax.set_title(title, fontsize=10)
        ax.legend(fontsize=7, ncol=2)
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(Path("figures") / "curves.png")
    plt.close(fig)


# ------------------------------------------------------------- 误分类样本

def misclassified_grid(model, run: str, device, n_grid: int = 12):
    """随机抽取 n_grid 个测试集误分类样本拼成网格，保存到 figures/。"""
    test_ds = torchvision.datasets.CIFAR10(
        "data", train=False, transform=T.Compose([T.ToTensor(),
                                                  T.Normalize(MEAN, STD)]))
    loader = DataLoader(test_ds, batch_size=256, num_workers=0)
    preds, labels = predict_all(model, loader, device)

    rng = np.random.RandomState(0)
    wrong = np.where(preds.numpy() != labels.numpy())[0]
    picked = rng.choice(wrong, size=min(n_grid, len(wrong)), replace=False)

    cols = 4
    rows = int(np.ceil(len(picked) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(9, rows * 2.3), dpi=150)
    # 反归一化回原图用于显示
    inv = T.Normalize(mean=[-m / s for m, s in zip(MEAN, STD)],
                      std=[1 / s for s in STD])
    for k, idx in enumerate(picked):
        ax = axes.flat[k]
        img = inv(test_ds[idx][0]).permute(1, 2, 0).clamp(0, 1).numpy()
        ax.imshow(img)
        ax.set_title(f"真:{CLASSES[labels[idx]]}\n预:{CLASSES[preds[idx]]}",
                     fontsize=8)
        ax.axis("off")
    for ax in axes.flat[len(picked):]:
        ax.axis("off")
    fig.suptitle(f"{run} 误分类样本示例", fontsize=11)
    fig.tight_layout()
    fig.savefig(Path("figures") / f"misclassified_{run}.png")
    plt.close(fig)


# ---------------------------------------------------------------- Grad-CAM

class _Hook:
    def __init__(self):
        self.acts, self.grads = None, None

    def __call__(self, module, inp, out):
        self.acts = out.detach()
        out.register_hook(lambda g: setattr(self, "grads", g.detach()))


def gradcam(model, images, target_idx):
    """对一批图像计算 Grad-CAM 热力图（需模型已含 target 层，梯度由 target
    类别反向传播）。返回与输入同尺寸的归一化热力图数组。"""
    hooks = []
    last_conv = None
    for m in model.modules():
        if isinstance(m, torch.nn.Conv2d):
            last_conv = m
    hook = _Hook()
    last_conv.register_forward_hook(hook)
    hooks.append(hook)

    logits = model(images)
    model.zero_grad()
    logits[:, target_idx].sum().backward()

    w = hook.grads.mean(dim=(2, 3), keepdim=True)      # [B, C, 1, 1]
    cams = (w * hook.acts).sum(1).clamp(min=0)         # [B, H, W]
    for i in range(cams.size(0)):
        cams[i] -= cams[i].min()
        m = cams[i].max()
        if m > 0:
            cams[i] /= m
    return cams


def gradcam_figure(model, run: str, device, n: int = 6):
    """测试集前若干张图像（含正误例）的 Grad-CAM 可视化。"""
    test_ds = torchvision.datasets.CIFAR10(
        "data", train=False, transform=T.Compose([T.ToTensor(),
                                                  T.Normalize(MEAN, STD)]))
    rng = np.random.RandomState(1)
    idx = rng.choice(len(test_ds), size=n, replace=False)

    images = torch.stack([test_ds[i][0] for i in idx]).to(device)
    labels = [test_ds[i][1] for i in idx]
    cams = gradcam(model, images, [labels[i] for i in range(n)])
    logits = model(images).argmax(1).cpu()

    inv = T.Normalize(mean=[-m / s for m, s in zip(MEAN, STD)],
                      std=[1 / s for s in STD])
    fig, axes = plt.subplots(2, n, figsize=(2.1 * n, 4.4), dpi=150)
    for i in range(n):
        img = inv(images[i].cpu()).permute(1, 2, 0).clamp(0, 1).numpy()
        axes[0, i].imshow(img)
        axes[0, i].set_title(
            f"真:{CLASSES[labels[i]]}\n预:{CLASSES[logits[i]]}", fontsize=8)
        axes[0, i].axis("off")
        axes[1, i].imshow(img)
        axes[1, i].imshow(cams[i].cpu().numpy(), cmap="jet", alpha=0.5)
        axes[1, i].axis("off")
    axes[0, 0].set_ylabel("原图")
    axes[1, 0].set_ylabel("Grad-CAM")
    fig.suptitle(f"{run} Grad-CAM 可解释性可视化", fontsize=11)
    fig.tight_layout()
    fig.savefig(Path("figures") / f"gradcam_{run}.png")
    plt.close(fig)


# ------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="append", required=True,
                    help="results 下的训练目录名，可多次传入")
    ap.add_argument("--n-mis", type=int, default=12)
    ap.add_argument("--n-cam", type=int, default=6)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Path("figures").mkdir(exist_ok=True)

    plot_curves(args.run)

    rows = []
    for run in args.run:
        model = load_model(run, device)
        acc, per_class = test_report(model, run, device)
        misclassified_grid(model, run, device, args.n_mis)
        gradcam_figure(model, run, device, args.n_cam)
        hist = np.genfromtxt(Path("results") / run / "history.csv",
                             delimiter=",", names=True)
        best_epoch = int(hist["epoch"][np.argmax(hist["val_acc"])])
        best_val = float(hist["val_acc"].max())
        rows.append([run, best_epoch, best_val, acc,
                     min(per_class), max(per_class)])
        print(f"{run}: best_val_acc={best_val:.4f}(ep{best_epoch}) "
              f"test_acc={acc:.4f} per_class∈[{min(per_class):.3f},"
              f"{max(per_class):.3f}]")

    with open(Path("figures") / "summary.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["run", "best_epoch", "best_val_acc", "test_acc",
                    "worst_class_acc", "best_class_acc"])
        w.writerows(rows)


if __name__ == "__main__":
    main()

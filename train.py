# -*- coding: utf-8 -*-
"""CIFAR-10 图像分类训练脚本。

用法示例：
    # 1) 基础 CNN，不做数据增强，固定学习率
    python train.py --run cnn_baseline --model simplecnn --epochs 50

    # 2) 基础 CNN + 数据增强 + 余弦退火 + 标签平滑 + 早停
    python train.py --run cnn_improved --model simplecnn --epochs 80 \\
        --augment --scheduler cosine --label-smoothing 0.1 --patience 15

    # 3) ResNet-20 + 与上相同的增强策略
    python train.py --run resnet20 --model resnet20 --epochs 80 \\
        --augment --scheduler cosine --label-smoothing 0.1 --patience 15

输出（results/<run>/）：best.pt（最佳验证集模型）、history.csv（逐轮指标）。
"""

import argparse
import csv
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader, Subset

from model import build_model

MEAN = (0.4914, 0.4822, 0.4465)   # CIFAR-10 各通道均值
STD = (0.2470, 0.2435, 0.2616)    # CIFAR-10 各通道标准差


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_transforms(augment: bool):
    """测试/验证只用标准化；训练在开启增强时叠加随机裁剪与水平翻转。"""
    base = [T.ToTensor(), T.Normalize(MEAN, STD)]
    if augment:
        train_trans = T.Compose([
            T.RandomCrop(32, padding=4),
            T.RandomHorizontalFlip(),
            *base,
        ])
    else:
        train_trans = T.Compose(base)
    eval_trans = T.Compose(base)
    return train_trans, eval_trans


def stratified_split(dataset, val_size: int, seed: int):
    """按类别分层抽样出验证集，其余作为训练集，保证各类比例一致。"""
    targets = np.array(dataset.targets)
    rng = np.random.RandomState(seed)
    val_idx, train_idx = [], []
    for c in range(10):
        idx = np.where(targets == c)[0]
        rng.shuffle(idx)
        val_idx.extend(idx[:val_size // 10])
        train_idx.extend(idx[val_size // 10:])
    return Subset(dataset, train_idx), Subset(dataset, val_idx)


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    """返回 (平均损失, 准确率)。"""
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        total_loss += criterion(logits, labels).item() * labels.size(0)
        correct += (logits.argmax(1) == labels).sum().item()
        total += labels.size(0)
    return total_loss / total, correct / total


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * labels.size(0)
        correct += (logits.argmax(1) == labels).sum().item()
        total += labels.size(0)
    return total_loss / total, correct / total


def main():
    parser = argparse.ArgumentParser(description="CIFAR-10 CNN 训练")
    parser.add_argument("--run", required=True, help="输出目录名 results/<run>")
    parser.add_argument("--model", default="simplecnn", choices=["simplecnn", "resnet20"])
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--scheduler", default="none",
                        choices=["none", "cosine", "step"])
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument("--augment", action="store_true", help="训练时使用数据增强")
    parser.add_argument("--patience", type=int, default=0,
                        help="验证准确率连续不提升多少轮后早停，0 表示关闭")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--val-size", type=int, default=5000)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path("results") / args.run
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- 数据：训练集 45k / 验证集 5k（分层抽样），测试集 10k
    train_trans, eval_trans = build_transforms(args.augment)
    full_train = torchvision.datasets.CIFAR10(
        args.data_root, train=True, download=True, transform=train_trans)
    train_ds, val_ds = stratified_split(full_train, args.val_size, args.seed)
    test_ds = torchvision.datasets.CIFAR10(
        args.data_root, train=False, download=True, transform=eval_trans)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=256, shuffle=False,
                            num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False,
                             num_workers=0)

    # ---- 模型 / 优化器 / 损失 / 调度器
    model = build_model(args.model).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[{args.run}] model={args.model} params={n_params:,} "
          f"device={device} epochs={args.epochs} augment={args.augment}")

    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

    if args.scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs)
    elif args.scheduler == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=args.epochs // 3, gamma=0.2)
    else:
        scheduler = None

    # ---- 训练循环（每轮记录 train/val 指标，按验证准确率保存最佳模型）
    best_acc, best_epoch, patience_left = 0.0, -1, args.patience
    history = []
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        lr = optimizer.param_groups[0]["lr"]
        if scheduler is not None:
            scheduler.step()
        history.append([epoch, lr, train_loss, train_acc, val_loss, val_acc])
        print(f"epoch {epoch:3d}/{args.epochs}  "
              f"train_loss={train_loss:.4f} train_acc={train_acc:.4f}  "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}  "
              f"lr={lr:.5f}  ({time.time()-t0:.0f}s)")

        if val_acc > best_acc:
            best_acc, best_epoch = val_acc, epoch
            torch.save(model.state_dict(), out_dir / "best.pt")
            patience_left = args.patience
        elif args.patience > 0:
            patience_left -= 1
            if patience_left <= 0:
                print(f"验证准确率连续 {args.patience} 轮未提升，早停。")
                break

    # ---- 持久化训练历史，并给出测试集最终评估
    with open(out_dir / "history.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "lr", "train_loss", "train_acc",
                         "val_loss", "val_acc"])
        writer.writerows(history)

    model.load_state_dict(torch.load(out_dir / "best.pt", map_location=device))
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    print(f"[{args.run}] best_val_acc={best_acc:.4f} (epoch {best_epoch})  "
          f"test_acc={test_acc:.4f} test_loss={test_loss:.4f}")


if __name__ == "__main__":
    main()

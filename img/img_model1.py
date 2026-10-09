# ==============================================================================
# EfficientNet-B0 融合改进模型（实验 b0_17：SE+SA + FPN + 细粒度纹理增强）
# （从头训练，无预训练）
#
#   主要参数调整：
#     1. FPN 通道数 fpn_channels=128，编码器输出 4*128=512 维；
#     2. 分类头输入维度改为 512，隐藏层 128，Dropout=0.5；
#     3. AdamW weight_decay=0.05；
#     4. 余弦调度总周期 TOTAL_EPOCHS=100，
#     5. 固定随机种子，尽量复现。
#
# 原融合改进：
#   【改进1 · 来自 b0_5】Block 级 SE+SA
#   【改进2 · 来自 b0_11】FPN 跨层多尺度融合
#   【改进3 · 来自 b0_14】细粒度纹理增强分支
# ==============================================================================
import os
import sys
import time
import random
from PIL import Image

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import efficientnet_b0

# 尝试导入 thop 用于计算 FLOPs，若未安装则跳过
try:
    from thop import profile
    THOP_AVAILABLE = True
except ImportError:
    THOP_AVAILABLE = False
    print("提示：未安装 thop，将跳过 FLOPs 计算（可运行 pip install thop）")


# ============================
# 0. 固定随机种子
# ============================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ============================
# 0.1 日志输出工具（同时写控制台与文件）
# ============================
class TeeLogger:
    """把 stdout 同时输出到控制台与日志文件"""
    def __init__(self, log_path, stream=None):
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        self.file = open(log_path, 'w', encoding='utf-8')
        self.stream = stream if stream is not None else sys.__stdout__

    def write(self, data):
        self.stream.write(data)
        self.file.write(data)

    def flush(self):
        self.stream.flush()
        self.file.flush()

    def close(self):
        self.file.close()


# ============================
# 1. 空间注意力模块（CBAM 的 Spatial 部分，来自 b0_5）
# ============================
class SpatialAttention(nn.Module):
    """
    CBAM 的空间注意力部分。
    对特征图在通道维做 max-pool 和 avg-pool，拼接后经 7x7 卷积 + Sigmoid，
    得到空间注意力图 [B, 1, H, W]，再与原特征图逐元素相乘。
    """
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: [B, C, H, W]
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        attn = self.sigmoid(self.conv(torch.cat([avg_out, max_out], dim=1)))
        return attn * x, attn


class SpatialAttentionSE(nn.Module):
    """
    与 SE 接口一致：forward(x: [B,C,H,W]) -> [B,C,H,W]。
    把本次注意力图缓存到 self.last_attn，供编码器聚合透传。
    """
    def __init__(self, kernel_size=7):
        super().__init__()
        self.spatial_attn = SpatialAttention(kernel_size=kernel_size)
        self.last_attn = None

    def forward(self, x):
        out, attn = self.spatial_attn(x)
        self.last_attn = attn
        return out


# ============================
# 2. 细粒度纹理增强分支（高频残差，来自 b0_14）
# ============================
class FineGrainedTextureBranch(nn.Module):
    """
    细粒度纹理/高频增强分支（Laplacian -> Depthwise 3x3 Conv -> BN -> 残差相加）。
    """
    def __init__(self, channels):
        super().__init__()
        laplacian = torch.tensor([[0., -1., 0.],
                                   [-1., 4., -1.],
                                   [0., -1., 0.]])
        self.register_buffer(
            'laplacian',
            laplacian.view(1, 1, 3, 3).expand(channels, 1, 3, 3).contiguous()
        )
        self.detail_conv = nn.Conv2d(
            channels, channels, kernel_size=3, padding=1, groups=channels, bias=False
        )
        self.bn = nn.BatchNorm2d(channels)
        self.channels = channels

    def forward(self, x):
        high_freq = F.conv2d(x, self.laplacian, padding=1, groups=self.channels)
        detail = self.detail_conv(high_freq)
        detail = self.bn(detail)
        return x + detail


# ============================
# 3. FPN 模块（标准自上而下融合，来自 b0_11）
# ============================
class FPN(nn.Module):
    """
    标准 FPN。输入多层特征（自底向上），输出融合后的多层特征列表。
    """
    def __init__(self, in_channels_list, out_channels=256):
        super().__init__()
        self.in_channels_list = in_channels_list
        self.out_channels = out_channels
        self.laterals = nn.ModuleList([
            nn.Conv2d(c, out_channels, 1, bias=False) for c in in_channels_list
        ])
        self.smooths = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.SiLU(inplace=True),
            ) for _ in in_channels_list
        ])

    def forward(self, feats):
        lats = [l(f) for l, f in zip(self.laterals, feats)]
        for i in range(len(lats) - 2, -1, -1):
            up = F.interpolate(lats[i + 1], size=lats[i].shape[-2:], mode='nearest')
            lats[i] = lats[i] + up
        outs = [s(l) for s, l in zip(self.smooths, lats)]
        return outs


# ============================
# 4. 图像编码器（EfficientNet-B0 + SE后SA + 纹理增强 + FPN，随机初始化）
# ============================
class ImageEncoder(nn.Module):
    """
    融合模型编码器：
      - 每个 MBConv 的 SE 之后插入 SpatialAttentionSE
      - Stage3 输出后接细粒度纹理增强分支
      - Stage3/5/7/9 四层特征送入 FPN 融合
      - 各层融合特征分别全局池化后 concat
    """
    STAGE_INDICES = [2, 4, 6, 8]
    IN_CHANNELS = [24, 80, 192, 1280]

    def __init__(self, fpn_channels=128):
        super().__init__()
        backbone = efficientnet_b0(weights=None)
        self.features = backbone.features
        backbone.classifier = nn.Identity()
        self.avgpool = backbone.avgpool

        # ---- 改进1：在每个 MBConv 内 SE 之后插入空间注意力 ----
        self._spatial_modules = []
        for stage in self.features:
            for block in stage.children():
                if type(block).__name__ == 'MBConv':
                    se_idx = None
                    for j, mod in enumerate(block.block):
                        if type(mod).__name__ == 'SqueezeExcitation':
                            se_idx = j
                            break
                    if se_idx is None:
                        continue
                    sa = SpatialAttentionSE(kernel_size=7)
                    old_block = block.block
                    modules = []
                    for j, mod in enumerate(old_block):
                        modules.append(mod)
                        if j == se_idx:
                            modules.append(sa)
                    block.block = nn.Sequential(*modules)
                    self._spatial_modules.append(sa)

        # ---- 改进3：Stage3 细粒度纹理增强分支 ----
        self.texture_stage3 = FineGrainedTextureBranch(channels=24)

        # ---- 改进2：FPN 四层融合 ----
        self.fpn = FPN(in_channels_list=self.IN_CHANNELS, out_channels=fpn_channels)
        self.out_channels = fpn_channels * len(self.STAGE_INDICES)

    def forward(self, x):
        feats = []
        stage_iter = iter(self.STAGE_INDICES)
        next_capture = next(stage_iter)

        for i in range(len(self.features)):
            x = self.features[i](x)
            if i == 2:
                x = self.texture_stage3(x)
            if i == next_capture:
                feats.append(x)
                try:
                    next_capture = next(stage_iter)
                except StopIteration:
                    pass

        outs = self.fpn(feats)
        pooled = [self.avgpool(o).flatten(1) for o in outs]
        feat = torch.cat(pooled, dim=1)

        attn_maps = []
        for sa in self._spatial_modules:
            if sa.last_attn is not None:
                if sa.last_attn.shape[-2:] != (7, 7):
                    a = F.interpolate(
                        sa.last_attn, size=(7, 7),
                        mode='bilinear', align_corners=False
                    )
                else:
                    a = sa.last_attn
                attn_maps.append(a)

        if attn_maps:
            attn_map = torch.stack(attn_maps, dim=0).mean(dim=0)
        else:
            attn_map = None

        return feat, attn_map


# ============================
# 5. 图像分类器（编码器 + 分类头）
# ============================
class ImageClassifier(nn.Module):
    def __init__(self, num_classes=5):
        super().__init__()
        # 关键修复：FPN 通道数改为 128，编码器输出 4*128=512
        self.encoder = ImageEncoder(fpn_channels=128)

        # 分类头输入维度 512
        self.classifier = nn.Sequential(
            nn.Dropout(0.6),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.6),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        feature, attn_map = self.encoder(x)
        logits = self.classifier(feature)
        return logits, attn_map, feature


# ============================
# 6. Dataset
# ============================
class ImageDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        import pandas as pd
        self.data = pd.read_csv(csv_file, sep='\t')
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        image_path = os.path.join(self.root_dir, row['file_path'])
        label = int(row['label'])
        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label


# ============================
# 7. 参数量统计工具
# ============================
def print_model_params(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    encoder_params = sum(p.numel() for p in model.encoder.parameters())
    cls_params = sum(p.numel() for p in model.classifier.parameters())

    print("=" * 60)
    print(f"模型总参数量: {total_params:,} ({total_params / 1e6:.2f} M)")
    print(f"可训练参数量: {trainable_params:,} ({trainable_params / 1e6:.2f} M)")
    print(f"编码器参数量: {encoder_params:,} ({encoder_params / 1e6:.2f} M)")
    print(f"分类头参数量: {cls_params:,} ({cls_params / 1e3:.2f} K)")
    print("=" * 60)

    return {
        'total_params': total_params,
        'trainable_params': trainable_params,
        'encoder_params': encoder_params,
        'classifier_params': cls_params,
    }


# ============================
# 8. 评价函数
# ============================
def evaluate_model(model, dataloader, device, num_classes=5):
    from sklearn.metrics import (
        accuracy_score, classification_report,
        precision_score, recall_score, f1_score
    )
    model.eval()
    all_preds = []
    all_labels = []

    if device.type == 'cuda':
        torch.cuda.synchronize()
    start_time = time.time()
    total_images = 0

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)
            logits, attn_map, _ = model(images)
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            total_images += images.size(0)

    if device.type == 'cuda':
        torch.cuda.synchronize()
    elapsed_time = time.time() - start_time
    throughput = total_images / elapsed_time

    accuracy = accuracy_score(all_labels, all_preds) * 100
    precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0) * 100
    recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0) * 100
    f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0) * 100
    report = classification_report(all_labels, all_preds, digits=4)

    flops = None
    if THOP_AVAILABLE:
        try:
            dummy_input = torch.randn(1, 3, 224, 224).to(device)
            flops, _ = profile(model, inputs=(dummy_input,), verbose=False)
            flops = flops / 1e9
        except Exception:
            flops = None

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'classification_report': report,
        'throughput': throughput,
        'flops': flops,
        'total_images': total_images
    }


# ============================
# 9. 主程序
# ============================
if __name__ == '__main__':
    set_seed(42)

    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    LOG_PATH = os.path.join(SCRIPT_DIR, '日志', 'img_model1_95.txt')
    BEST_MODEL_PATH = os.path.join(
        SCRIPT_DIR, 'best_image_classifier_scratch_img_model1_95.pth'
    )

    logger = TeeLogger(LOG_PATH)
    sys.stdout = logger

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    print(f"日志文件: {LOG_PATH}")
    print(f"模型保存路径: {BEST_MODEL_PATH}")
    print("融合改进实验 b0_17（约95%参数版）: SE后插SA + FPN四层融合 + 细粒度纹理增强")
    print("参数调整：fpn_channels=128, dropout=0.5, wd=0.05, TRAIN_EPOCHS=50, patience=8")

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3)
    ])
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3)
    ])

    DATA_ROOT = r"D:\pythonlearning\lunwen\628\data\pic913\pic9.13"
    TRAIN_CSV = r'D:/pythonlearning/lunwen/628/data/pic913/train.csv'
    TEST_CSV = r'D:/pythonlearning/lunwen/628/data/pic913/test.csv'

    train_dataset = ImageDataset(TRAIN_CSV, DATA_ROOT, train_transform)
    test_dataset = ImageDataset(TEST_CSV, DATA_ROOT, test_transform)

    train_loader = DataLoader(
        train_dataset, batch_size=64, shuffle=True,
        num_workers=4, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=64, shuffle=False,
        num_workers=4, pin_memory=True
    )

    model = ImageClassifier(num_classes=5).to(device)
    print_model_params(model)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.08)
    
    TRAIN_EPOCHS = 100
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=TOTAL_EPOCHS, eta_min=1e-6
    )

    LAMBDA_FOCUS = 0.1

    def topk_focus_loss(attn_map, ratio=0.25):
        b = attn_map.shape[0]
        flat = attn_map.view(b, -1)
        k = max(1, int(flat.shape[1] * ratio))
        topk, _ = torch.topk(flat, k, dim=1)
        return 1.0 - topk.sum(dim=1) / (flat.sum(dim=1) + 1e-8)

    best_acc = 0.0
    best_epoch = -1
    patience = 8
    early_stop_counter = 0

    from sklearn.metrics import accuracy_score

    print(f"实验: b0_17 融合模型 | "
          f"Top-k 聚焦辅助损失(LAMBDA_FOCUS={LAMBDA_FOCUS})")
    print()

    for epoch in range(TRAIN_EPOCHS):
        # 训练
        model.train()
        total_loss = 0
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            logits, attn_map, _ = model(images)

            cls_loss = criterion(logits, labels)
            if attn_map is not None:
                aux_loss = topk_focus_loss(attn_map, ratio=0.25)
                loss = cls_loss + LAMBDA_FOCUS * aux_loss.mean()
            else:
                loss = cls_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # 验证
        model.eval()
        y_true, y_pred = [], []
        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(device)
                logits, _, _ = model(images)
                preds = torch.argmax(logits, dim=1).cpu().numpy()
                y_pred.extend(preds)
                y_true.extend(labels.numpy())

        acc = accuracy_score(y_true, y_pred) * 100
        print(f"Epoch:{epoch:2d} Loss:{total_loss:.4f} "
              f"Acc:{acc:.2f}% LR:{scheduler.get_last_lr()[0]:.2e}")

        # 保存最佳模型
        if acc > best_acc:
            best_acc = acc
            best_epoch = epoch
            early_stop_counter = 0
            torch.save(model.state_dict(), BEST_MODEL_PATH)
            print(f"  保存最佳模型 (epoch {epoch}, acc {acc:.2f}%)")
        else:
            early_stop_counter += 1
            print(f"  连续 {early_stop_counter}/{patience} 轮未提升")
            if early_stop_counter >= patience:
                print(f"\n早停触发！连续 {patience} 轮准确率未提升，训练提前结束")
                break

        scheduler.step()

    print(f"\n训练完成！最佳准确率: {best_acc:.2f}% (Epoch {best_epoch})")

    # ============================
    # 10. 最终评估（加载最佳模型）
    # ============================
    print("\n正在加载最佳模型进行最终评估...")
    best_model = ImageClassifier(num_classes=5).to(device)

    if os.path.exists(BEST_MODEL_PATH):
        best_model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=device))
        print("成功加载最佳模型权重")
    else:
        print("未找到最佳模型文件，使用当前模型")
        best_model = model

    results = evaluate_model(best_model, test_loader, device, num_classes=5)

    print("\n" + "=" * 60)
    print("测试集最终评价指标（EfficientNet-B0 + SE后SA + FPN + 细粒度纹理增强）")
    print("=" * 60)
    print(f"准确率 (Accuracy)   : {results['accuracy']:.2f}%")
    print(f"加权精确率 (Precision): {results['precision']:.2f}%")
    print(f"加权召回率 (Recall)   : {results['recall']:.2f}%")
    print(f"加权 F1-score        : {results['f1']:.2f}%")
    print("\n详细分类报告：")
    print(results['classification_report'])
    print(f"吞吐量 (Throughput) : {results['throughput']:.2f} images/sec "
          f"(测试集共 {results['total_images']} 张)")
    if results['flops'] is not None:
        print(f"FLOPs (G)          : {results['flops']:.2f} GFLOPs")
    else:
        print("FLOPs 未计算（请安装 thop）")

    print("\n" + "=" * 60)
    print("模型参数量统计：")
    print_model_params(best_model)

    print("=" * 60)
    print(f"完整模型已保存至 '{BEST_MODEL_PATH}'")
    print(f"日志已保存至 '{LOG_PATH}'")

    sys.stdout = logger.stream
    logger.close()

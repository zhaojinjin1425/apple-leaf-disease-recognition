# ============================================================
# qk_img_model1.py - 三元组级 Query-Key 选择融合 SINet
# 视觉编码器改用 img_model1.py 中的 ImageEncoder（FPN 通道 128，输出 512 维）
# ============================================================
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

# 尝试导入 thop 用于计算 FLOPs
try:
    from thop import profile
    THOP_AVAILABLE = True
except ImportError:
    THOP_AVAILABLE = False
    print("提示：未安装 thop，将跳过 FLOPs 计算（可运行 pip install thop）")

# ---- 动态路径，导入 ImageEncoder ----
current_dir = os.path.dirname(os.path.abspath(__file__))      # .../lunwen/628/fusion
parent_dir = os.path.dirname(current_dir)                     # .../lunwen/628
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# 改为从 img.img_model1 导入 ImageEncoder
from img.img_model1 import ImageEncoder


# ============================================================
# 固定随机种子
# ============================================================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ============================================================
# 日志输出工具
# ============================================================
class TeeLogger:
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


# ============================================================
# KG 数据加载
# ============================================================
def load_kg_data(kg_dir):
    """
    加载 KG 嵌入矩阵 M、三元组索引 triple_indices、类-三元组标签 class_triple_labels。
    """
    # 1. 加载嵌入矩阵
    M = np.load(os.path.join(kg_dir, 'kg_embedding_M.npy'))  # (E, d)

    # 2. 解析 id.txt 得到 name -> id 映射
    name2id = {}
    id_path = os.path.join(kg_dir, 'id.txt')
    with open(id_path, 'r', encoding='utf-8') as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split('\t') if '\t' in line else line.split()
            if parts[0] in ('索引', 'id', 'ID', 'index'):
                continue
            if len(parts) < 2:
                print(f"[警告] id.txt 第 {line_no} 行格式异常，已跳过: {line}")
                continue
            try:
                idx = int(parts[0])
            except ValueError:
                print(f"[警告] id.txt 第 {line_no} 行 ID 无法解析，已跳过: {line}")
                continue
            name = parts[1].strip()
            name2id[name] = idx

    print(f"[信息] 从 id.txt 载入 {len(name2id)} 个实体")

    # 3. 解析 triples.txt 得到三元组索引
    triples = []
    triple_path = os.path.join(kg_dir, 'triples.txt')
    with open(triple_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split('\t') if '\t' in line else line.split()
            if len(parts) >= 3:
                h, r, t = parts[0].strip(), parts[1].strip(), parts[2].strip()
                triples.append((h, r, t))

    N = len(triples)
    triple_indices = np.zeros((N, 3), dtype=np.int64)
    for i, (h, r, t) in enumerate(triples):
        triple_indices[i, 0] = name2id[h]
        triple_indices[i, 1] = name2id[r]
        triple_indices[i, 2] = name2id[t]

    # 4. 构建类-三元组标签矩阵 (5, N)
    class_name2label = {
        '苹果斑点落叶病': 0,
        '苹果褐斑病': 1,
        '苹果花叶病': 2,
        '苹果白粉病': 3,
        '苹果锈病': 4,
    }
    num_classes = 5
    class_triple_labels = np.zeros((num_classes, N), dtype=np.float32)
    for i, (h, r, t) in enumerate(triples):
        if h in class_name2label:
            label = class_name2label[h]
            class_triple_labels[label, i] = 1.0
        else:
            print(f"警告：三元组头实体 '{h}' 不在类别映射中，已忽略")

    return M, triple_indices, class_triple_labels


# ============================================================
# 三元组级 Query-Key 知识模块
# ============================================================
class TripleAttentionKnowledgeModule(nn.Module):
    def __init__(self, kg_embed_path, triple_indices, class_triple_labels,
                 visual_feat_dim=512, kg_embed_dim=16, d_k=128, dropout=0.1):
        super().__init__()
        M = np.load(kg_embed_path)
        self.register_buffer('M', torch.tensor(M, dtype=torch.float32))
        self.register_buffer('triple_indices', torch.tensor(triple_indices, dtype=torch.long))
        self.register_buffer('class_triple_labels', torch.tensor(class_triple_labels, dtype=torch.float32))

        self.N = triple_indices.shape[0]

        self.W_g = nn.Linear(3 * kg_embed_dim, d_k)
        self.W_q = nn.Linear(visual_feat_dim, d_k)
        self.W_k = nn.Linear(d_k, d_k)
        self.W_v = nn.Linear(d_k, d_k)
        self.projector = nn.Sequential(
            nn.Linear(d_k, visual_feat_dim),
            nn.ReLU()
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, visual_feat):
        B = visual_feat.size(0)
        h = self.M[self.triple_indices[:, 0]]
        r = self.M[self.triple_indices[:, 1]]
        t = self.M[self.triple_indices[:, 2]]
        G = self.W_g(torch.cat([h, r, t], dim=-1))

        q = self.W_q(visual_feat)
        K = self.W_k(G)
        V = self.W_v(G)

        logits = torch.matmul(q, K.transpose(0, 1)) / (K.size(-1) ** 0.5)
        alpha = F.softmax(logits, dim=-1)

        z_kg = torch.matmul(alpha, V)
        z_kg = self.dropout(z_kg)
        kg_feat = self.projector(z_kg)

        return kg_feat, logits, alpha


# ============================================================
# 双线性融合（Mutan）
# ============================================================
class MutanFusion(nn.Module):
    def __init__(self, visual_dim, kg_dim, rank=200, output_dim=512):
        super().__init__()
        self.fc_v = nn.Linear(visual_dim, rank, bias=False)
        self.fc_k = nn.Linear(kg_dim, rank, bias=False)
        self.fc_out = nn.Linear(rank, output_dim, bias=True)

    def forward(self, visual_feat, kg_feat):
        v = self.fc_v(visual_feat)
        k = self.fc_k(kg_feat)
        fused = v * k
        out = self.fc_out(fused)
        return out


# ============================================================
# 完整 QK-SINet 模型（使用 img_model1 的 ImageEncoder）
# ============================================================
class QKSINet(nn.Module):
    def __init__(self, kg_dir, num_classes=5,
                 visual_feat_dim=512, kg_embed_dim=16, d_k=128,
                 fusion_rank=200, fusion_out=512,
                 freeze_encoder=False, encoder_weights_path=None):
        super().__init__()
        M, triple_indices, class_triple_labels = load_kg_data(kg_dir)
        self.num_triples = triple_indices.shape[0]
        self.num_classes = num_classes

        # 1. 视觉编码器（来自 img_model1）
        self.encoder = ImageEncoder()   # 输出维度 512
        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False
        if encoder_weights_path is not None and os.path.exists(encoder_weights_path):
            state_dict = torch.load(encoder_weights_path, map_location='cpu')
            new_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith('encoder.'):
                    new_state_dict[k[8:]] = v
                elif k.startswith('features') or k.startswith('texture_stage3') or k.startswith('fpn'):
                    new_state_dict[k] = v
            self.encoder.load_state_dict(new_state_dict, strict=False)
            print(f"已加载编码器预训练权重: {encoder_weights_path}")

        # 2. 知识模块
        self.knowledge = TripleAttentionKnowledgeModule(
            kg_embed_path=os.path.join(kg_dir, 'kg_embedding_M.npy'),
            triple_indices=triple_indices,
            class_triple_labels=class_triple_labels,
            visual_feat_dim=visual_feat_dim,
            kg_embed_dim=kg_embed_dim,
            d_k=d_k,
            dropout=0.1
        )

        # 3. 融合模块
        self.fusion = MutanFusion(visual_feat_dim, visual_feat_dim,
                                  rank=fusion_rank, output_dim=fusion_out)

        # 4. 分类头（参考 img_model1 的配置：Dropout 0.6 + 隐藏层 128）
        self.classifier = nn.Sequential(
            nn.Dropout(0.6),
            nn.Linear(fusion_out, 128),
            nn.ReLU(),
            nn.Dropout(0.6),
            nn.Linear(128, num_classes)
        )

        # 注册类-三元组标签
        self.register_buffer('class_triple_labels_tensor',
                             torch.tensor(class_triple_labels, dtype=torch.float32))

    def forward(self, x):
        visual_feat, attn_map = self.encoder(x)          # (B, 512), attn_map: (B,1,7,7)
        kg_feat, logits_kg, alpha = self.knowledge(visual_feat)
        fused = self.fusion(visual_feat, kg_feat)        # (B, 512)
        logits_cls = self.classifier(fused)              # (B, num_classes)
        return logits_cls, logits_kg, alpha, attn_map

    def get_class_triple_labels(self):
        return self.class_triple_labels_tensor


# ============================================================
# Dataset
# ============================================================
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


# ============================================================
# 参数量统计
# ============================================================
def print_model_params(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("=" * 60)
    print(f"模型总参数量: {total_params:,} ({total_params / 1e6:.2f} M)")
    print(f"可训练参数量: {trainable_params:,} ({trainable_params / 1e6:.2f} M)")
    print("=" * 60)
    return {
        'total_params': total_params,
        'trainable_params': trainable_params,
    }


# ============================================================
# 评价函数
# ============================================================
def evaluate_model(model, dataloader, device, num_classes=5):
    from sklearn.metrics import (accuracy_score, classification_report,
                                 precision_score, recall_score, f1_score)
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
            logits_cls, _, _, _ = model(images)
            preds = torch.argmax(logits_cls, dim=1)
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


# ============================================================
# 主程序
# ============================================================
if __name__ == '__main__':
    set_seed(42)

    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    LOG_PATH = os.path.join(SCRIPT_DIR, '日志', 'qk_img_model1_95.txt')
    BEST_MODEL_PATH = os.path.join(SCRIPT_DIR, 'best_qk_img_model1_95.pth')

    logger = TeeLogger(LOG_PATH)
    sys.stdout = logger

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    print(f"日志文件: {LOG_PATH}")
    print(f"模型保存路径: {BEST_MODEL_PATH}")
    print("QK-SINet 融合模型（视觉编码器来自 img_model1：FPN 128 通道，输出 512 维）")
    print("参数：visual_feat_dim=512, fusion_out=512, dropout=0.6, wd=0.08, TRAIN_EPOCHS=45, patience=8")

    # 数据增强
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

    # 数据路径（与 img_model1 保持一致）
    DATA_ROOT = r"D:\pythonlearning\lunwen\628\data\pic913\pic9.13"
    TRAIN_CSV = r'D:/pythonlearning/lunwen/628/data/pic913/train.csv'
    TEST_CSV = r'D:/pythonlearning/lunwen/628/data/pic913/test.csv'

    train_dataset = ImageDataset(TRAIN_CSV, DATA_ROOT, train_transform)
    test_dataset = ImageDataset(TEST_CSV, DATA_ROOT, test_transform)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True,
                              num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False,
                             num_workers=4, pin_memory=True)

    # KG 目录
    KG_DIR = os.path.join(parent_dir, 'kg')

    # 图像编码器预训练权重（来自 img_model1 训练保存的模型）
    ENCODER_WEIGHTS = os.path.join(SCRIPT_DIR, 'best_image_classifier_scratch_img_model1_95.pth')

    model = QKSINet(
        kg_dir=KG_DIR,
        num_classes=5,
        visual_feat_dim=512,
        kg_embed_dim=16,
        d_k=128,
        fusion_rank=200,
        fusion_out=512,
        freeze_encoder=False,
        encoder_weights_path=ENCODER_WEIGHTS
    ).to(device)

    print_model_params(model)

    # 损失函数
    criterion_cls = nn.CrossEntropyLoss()
    criterion_kg = nn.BCEWithLogitsLoss()

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.08)

    TOTAL_EPOCHS = 100
    TRAIN_EPOCHS = 45
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=TOTAL_EPOCHS, eta_min=1e-6
    )

    LAMBDA_FOCUS = 0.1
    LAMBDA_KG = 0.1

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

    print(f"开始训练 QK-SINet | LAMBDA_FOCUS={LAMBDA_FOCUS}, LAMBDA_KG={LAMBDA_KG}")
    print()

    for epoch in range(TOTAL_EPOCHS):
        model.train()
        total_loss = 0
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            logits_cls, logits_kg, alpha, attn_map = model(images)

            # 分类损失
            cls_loss = criterion_cls(logits_cls, labels)

            # 三元组选择损失
            class_triple_labels = model.get_class_triple_labels()  # (C, N)
            target_kg = class_triple_labels[labels]                # (B, N)
            kg_loss = criterion_kg(logits_kg, target_kg)

            # Top-k 聚焦辅助损失
            if attn_map is not None:
                aux_loss = topk_focus_loss(attn_map, ratio=0.25).mean()
            else:
                aux_loss = 0.0

            loss = cls_loss + LAMBDA_KG * kg_loss + LAMBDA_FOCUS * aux_loss

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
                logits_cls, _, _, _ = model(images)
                preds = torch.argmax(logits_cls, dim=1).cpu().numpy()
                y_pred.extend(preds)
                y_true.extend(labels.numpy())

        acc = accuracy_score(y_true, y_pred) * 100
        print(f"Epoch:{epoch:2d} Loss:{total_loss:.4f} "
              f"Acc:{acc:.2f}% LR:{scheduler.get_last_lr()[0]:.2e}")

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
    # 最终评估
    # ============================
    print("\n正在加载最佳模型进行最终评估...")
    best_model = QKSINet(
        kg_dir=KG_DIR,
        num_classes=5,
        visual_feat_dim=512,
        kg_embed_dim=16,
        d_k=128,
        fusion_rank=200,
        fusion_out=512,
        freeze_encoder=False,
        encoder_weights_path=None
    ).to(device)

    if os.path.exists(BEST_MODEL_PATH):
        best_model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=device))
        print("成功加载最佳模型权重")
    else:
        print("未找到最佳模型文件，使用当前模型")
        best_model = model

    results = evaluate_model(best_model, test_loader, device, num_classes=5)

    print("\n" + "=" * 60)
    print("测试集最终评价指标（QK-SINet + img_model1 编码器）")
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

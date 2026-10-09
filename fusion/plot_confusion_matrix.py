# ============================================================
# plot_confusion_matrix.py
# 加载 QK-SINet 最优权重，在测试集上生成混淆矩阵
# ============================================================
import os
import sys
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (confusion_matrix, classification_report,
                             accuracy_score, precision_score,
                             recall_score, f1_score)
# ---- 动态路径 ----
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
# 直接复用你训练代码中的模型与数据集定义
from qk_img_model1 import QKSINet, ImageDataset, set_seed

if __name__ == '__main__':
    # ============================================================
    # 输出目录（不存在则自动创建）
    # ============================================================
    OUTPUT_DIR = r"D:\pythonlearning\lunwen\628\fusion\matrix"
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"输出目录: {OUTPUT_DIR}")
    # ============================================================
    # 设置中文字体（避免图中中文乱码）
    # ============================================================
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
    plt.rcParams['axes.unicode_minus'] = False
    # ============================================================
    # 配置
    # ============================================================
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    SCRIPT_DIR = current_dir
    BEST_MODEL_PATH = os.path.join(SCRIPT_DIR, 'best_qk_img_model1_95.pth')
    KG_DIR = os.path.join(parent_dir, 'kg')
    DATA_ROOT = r"D:\pythonlearning\lunwen\628\data\pic913\pic9.13"
    TEST_CSV = r'D:/pythonlearning/lunwen/628/data/pic913/test.csv'
    CLASS_NAMES = ['斑点落叶病', '褐斑病', '花叶病', '白粉病', '锈病']
    NUM_CLASSES = 5
    # ============================================================
    # 测试集（与训练一致：只做确定性预处理，不做数据增强）
    # ============================================================
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3)
    ])
    test_dataset = ImageDataset(TEST_CSV, DATA_ROOT, test_transform)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False,
                             num_workers=4, pin_memory=True)
    print(f"测试集样本数: {len(test_dataset)}")
    # ============================================================
    # 加载模型
    # ============================================================
    model = QKSINet(
        kg_dir=KG_DIR,
        num_classes=NUM_CLASSES,
        visual_feat_dim=512,
        kg_embed_dim=16,
        d_k=128,
        fusion_rank=200,
        fusion_out=512,
        freeze_encoder=False,
        encoder_weights_path=None
    ).to(device)
    if os.path.exists(BEST_MODEL_PATH):
        state_dict = torch.load(BEST_MODEL_PATH, map_location=device)
        model.load_state_dict(state_dict)
        print(f"成功加载模型权重: {BEST_MODEL_PATH}")
    else:
        raise FileNotFoundError(f"未找到模型权重文件: {BEST_MODEL_PATH}")
    # ============================================================
    # 推理，收集真实标签与预测标签
    # ============================================================
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            logits_cls, _, _, _ = model(images)
            preds = torch.argmax(logits_cls, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    # ============================================================
    # 计算指标
    # ============================================================
    accuracy = accuracy_score(all_labels, all_preds) * 100
    precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0) * 100
    recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0) * 100
    f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0) * 100
    print("\n" + "=" * 60)
    print("测试集评价指标")
    print("=" * 60)
    print(f"Accuracy : {accuracy:.2f}%")
    print(f"Precision: {precision:.2f}%")
    print(f"Recall   : {recall:.2f}%")
    print(f"F1-score : {f1:.2f}%")
    print("\n分类报告：")
    print(classification_report(all_labels, all_preds,
                                target_names=CLASS_NAMES, digits=4))
    # ============================================================
    # 混淆矩阵
    # ============================================================
    cm = confusion_matrix(all_labels, all_preds)
    # 归一化混淆矩阵（按真实类别归一化，用于看召回率）
    cm_norm = cm.astype('float') / cm.sum(axis=1, keepdims=True)
    # 保存数值文件
    np.save(os.path.join(OUTPUT_DIR, 'confusion_matrix.npy'), cm)
    np.save(os.path.join(OUTPUT_DIR, 'confusion_matrix_norm.npy'), cm_norm)
    print(f"\n混淆矩阵原始值已保存: {os.path.join(OUTPUT_DIR, 'confusion_matrix.npy')}")
    print(f"混淆矩阵归一化值已保存: {os.path.join(OUTPUT_DIR, 'confusion_matrix_norm.npy')}")
    # 保存分类报告为 txt
    report_txt_path = os.path.join(OUTPUT_DIR, 'classification_report.txt')
    with open(report_txt_path, 'w', encoding='utf-8') as f:
        f.write(f"Accuracy : {accuracy:.4f}%\n")
        f.write(f"Precision: {precision:.4f}%\n")
        f.write(f"Recall   : {recall:.4f}%\n")
        f.write(f"F1-score : {f1:.4f}%\n\n")
        f.write(classification_report(all_labels, all_preds,
                                      target_names=CLASS_NAMES, digits=4))
    print(f"分类报告已保存: {report_txt_path}")
    # ============================================================
    # 绘图 1：原始计数混淆矩阵
    # ============================================================
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
                cbar_kws={'label': '样本数'})
    plt.xlabel('预测类别', fontsize=12)
    plt.ylabel('真实类别', fontsize=12)
    plt.title('QK-SINet 混淆矩阵（测试集）', fontsize=14)
    plt.xticks(rotation=30, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    save_path1 = os.path.join(OUTPUT_DIR, 'confusion_matrix.png')
    plt.savefig(save_path1, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"混淆矩阵图已保存: {save_path1}")
    # ============================================================
    # 绘图 2：归一化混淆矩阵（百分比）
    # ============================================================
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_norm, annot=True, fmt='.3f', cmap='Blues',
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
                vmin=0, vmax=1, cbar_kws={'label': '归一化比例'})
    plt.xlabel('预测类别', fontsize=12)
    plt.ylabel('真实类别', fontsize=12)
    plt.title('QK-SINet 归一化混淆矩阵（测试集）', fontsize=14)
    plt.xticks(rotation=30, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    save_path2 = os.path.join(OUTPUT_DIR, 'confusion_matrix_norm.png')
    plt.savefig(save_path2, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"归一化混淆矩阵图已保存: {save_path2}")
    # ============================================================
    # 打印每类 Precision / Recall / F1
    # ============================================================
    print("\n" + "=" * 60)
    print("单类病害性能")
    print("=" * 60)
    report = classification_report(all_labels, all_preds,
                                   target_names=CLASS_NAMES,
                                   digits=4, output_dict=True)
    for i, name in enumerate(CLASS_NAMES):
        print(f"{name:8s}  Precision: {report[name]['precision']*100:6.2f}%  "
              f"Recall: {report[name]['recall']*100:6.2f}%  "
              f"F1: {report[name]['f1-score']*100:6.2f}%")
    print(f"\n所有文件已保存到: {OUTPUT_DIR}")

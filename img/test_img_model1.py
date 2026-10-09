# ==============================================================================
# 读取 img_model1.py 训练好的模型，对 test_input 文件夹中的图片进行预测并
# 生成多层级 FPN 融合的 Grad-CAM++ 热力图可视化结果，保存到 test_output 文件夹。
 随机种子固定，模型权重文件为 best_image_classifier_scratch_efficientnet_b0_17_controlled1.pth
#
# 可视化逻辑与 test_efficientnet_b0_17.py 相同：
#   - 对 FPN 的 4 个 smooth 层分别计算 Grad-CAM++
#   - 根据各层梯度强度加权融合，得到最终热力图
# ==============================================================================
import os
import sys
import glob

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib import cm

from torchvision import transforms

# 从新训练脚本导入模型类（结构完全一致，仅参数不同）
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from img_model1 import ImageClassifier, ImageEncoder


# ============================
# Grad-CAM++ 实现（与原始可视化一致）
# ============================
class GradCAMPlusPlus:
    """
    Grad-CAM++：用二阶导数（梯度的平方）对 CAM 权重做加权，
    能更好地定位多个分散的激活峰值。
    """
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, inputs, outputs):
            self.activations = outputs

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0]

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate(self, input_tensor, target_class=None):
        self.model.eval()
        logits, attn_map, feature = self.model(input_tensor)

        if target_class is None:
            target_class = logits.argmax(dim=1).item()

        self.model.zero_grad()
        score = logits[0, target_class]
        score.backward()

        gradients = self.gradients
        activations = self.activations

        # ---- Grad-CAM++ 权重计算 ----
        grad2 = gradients.pow(2)
        denom = 2 * grad2 + (grad2 * activations).sum(dim=(2, 3), keepdim=True) + 1e-8
        alpha = grad2 / denom
        weights = (alpha * F.relu(gradients)).sum(dim=(2, 3), keepdim=True)
        cam = (weights * activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = torch.squeeze(cam)
        cam = cam.cpu().detach().numpy()
        cam_min, cam_max = cam.min(), cam.max()
        if cam_max - cam_min > 1e-8:
            cam = (cam - cam_min) / (cam_max - cam_min)
        return cam, target_class, logits


# ============================
# 工具函数
# ============================
def overlay_heatmap(img_pil, cam, alpha=0.5):
    """把热力图叠加到原图上，返回 RGB numpy 和纯热力图"""
    img = np.array(img_pil.convert('RGB').resize((224, 224))).astype(np.float32) / 255.0
    cam_resized = np.array(Image.fromarray(cam).resize((224, 224), resample=Image.BILINEAR))
    heatmap = cm.jet(cam_resized)[..., :3]
    overlay = heatmap * alpha + img * (1 - alpha)
    overlay = np.clip(overlay, 0, 1)
    return overlay, heatmap


# ============================
# 核心：多层级 FPN 融合 CAM（与原始逻辑一致）
# ============================
def generate_multi_level_cam(model, input_tensor, target_class=None, return_layer_cams=False):
    """
    对 FPN 的 4 个 smooth 层分别计算 Grad-CAM++，然后按梯度强度加权融合。

    参数：
        model: ImageClassifier 实例（必须与 img_model1.py 结构一致）
        input_tensor: 预处理后的输入张量 [1,3,224,224]
        target_class: 目标类别索引，None 则取模型预测
        return_layer_cams: 是否返回每层的 CAM 和权重（用于调试）

    返回：
        fused_cam: 融合后的归一化热力图 (224,224) numpy
        target_class: 预测类别
        logits: 模型原始 logits
        (可选) layer_cams, layer_weights
    """
    model.eval()
    logits, _, _ = model(input_tensor)
    if target_class is None:
        target_class = logits.argmax(dim=1).item()

    # 用于存储各层 CAM 和梯度强度
    layer_cams = []
    layer_weights = []

    # 遍历 FPN 的 4 个 smooth 层
    for idx in range(4):
        target_layer = model.encoder.fpn.smooths[idx]
        grad_cam = GradCAMPlusPlus(model, target_layer)
        cam, _, _ = grad_cam.generate(input_tensor, target_class)

        # 计算该层梯度强度（绝对值均值）作为权重
        grad_strength = grad_cam.gradients.abs().mean().item()
        layer_weights.append(grad_strength)

        # 将 CAM resize 到 224x224
        cam_resized = np.array(Image.fromarray(cam).resize((224, 224), resample=Image.BILINEAR))
        layer_cams.append(cam_resized)

    # 权重归一化（softmax，防止某一层权重为0）
    weights = np.array(layer_weights)
    if weights.sum() == 0:   # 防止所有梯度为0
        weights = np.ones_like(weights) / len(weights)
    else:
        # 使用指数归一化，让大的权重更突出
        weights = np.exp(weights) / np.sum(np.exp(weights))

    # 加权融合
    fused_cam = np.zeros_like(layer_cams[0])
    for i, cam in enumerate(layer_cams):
        fused_cam += weights[i] * cam

    # 归一化到 [0,1]
    fused_cam = (fused_cam - fused_cam.min()) / (fused_cam.max() - fused_cam.min() + 1e-8)

    if return_layer_cams:
        return fused_cam, target_class, logits, layer_cams, weights
    else:
        return fused_cam, target_class, logits


# ============================
# 主程序
# ============================
if __name__ == '__main__':
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

    MODEL_PATH = os.path.join(SCRIPT_DIR, 'best_image_classifier_scratch_efficientnet_b0_17_controlled1.pth')
    INPUT_DIR = os.path.join(SCRIPT_DIR, 'test_input')
    OUTPUT_DIR = os.path.join(SCRIPT_DIR, 'test_output', 'test_img_model1_multilevel_cam')

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if not os.path.exists(MODEL_PATH):
        print(f"错误：找不到训练好的模型文件 {MODEL_PATH}")
        print("请先运行 img_model1.py 完成训练。")
        sys.exit(1)
    if not os.path.exists(INPUT_DIR):
        print(f"错误：找不到输入图片文件夹 {INPUT_DIR}")
        print("请创建该文件夹并放入若干测试图片。")
        sys.exit(1)

    CLASS_NAMES = ['alternaria leaf spot', 'brown spot', 'mosaic', 'powdery mildew', 'rust']
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    FPN_CHANNELS = 40      # 训练时设定的值
    DROPOUT_RATE = 0.45    # 训练时设定的值

    # 实例化模型并加载权重
    model = ImageClassifier(num_classes=5, fpn_channels=FPN_CHANNELS, dropout_rate=DROPOUT_RATE).to(device)
    state_dict = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    print(f"已加载模型权重: {MODEL_PATH}")
    print(f"模型配置: fpn_channels={FPN_CHANNELS}, dropout_rate={DROPOUT_RATE}")
    print("可视化方式: 多层级 FPN 融合 Grad-CAM++ (4层加权)")

    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])

    exts = ('*.jpg', '*.jpeg', '*.png', '*.bmp', '*.JPG', '*.JPEG', '*.PNG')
    image_files = []
    for ext in exts:
        image_files.extend(glob.glob(os.path.join(INPUT_DIR, ext)))
    image_files = sorted(set(image_files))

    if len(image_files) == 0:
        print(f"错误：输入图片文件夹 {INPUT_DIR} 中没有找到图片")
        sys.exit(1)
    print(f"共找到 {len(image_files)} 张测试图片\n")

    for img_path in image_files:
        fname = os.path.basename(img_path)
        try:
            img_pil = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"  跳过 {fname}（读取失败：{e}）")
            continue

        input_tensor = test_transform(img_pil).unsqueeze(0).to(device)
        input_tensor.requires_grad_(True)

        # ---- 生成融合 CAM ----
        fused_cam, pred_class, logits = generate_multi_level_cam(
            model, input_tensor, target_class=None, return_layer_cams=False
        )
        pred_label = CLASS_NAMES[pred_class] if pred_class < len(CLASS_NAMES) else str(pred_class)
        prob = F.softmax(logits, dim=1)[0, pred_class].item()

        # ---- 可视化 ----
        overlay, heatmap = overlay_heatmap(img_pil, fused_cam, alpha=0.5)

        # 三子图：原图 / 融合热力图 / 叠加图
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        axes[0].imshow(img_pil.resize((224, 224)))
        axes[0].set_title('Original')
        axes[0].axis('off')

        axes[1].imshow(heatmap)
        axes[1].set_title('Fused CAM\n(4 FPN layers)')
        axes[1].axis('off')

        axes[2].imshow(overlay)
        axes[2].set_title(f'Overlay\nPred: {pred_label} ({prob*100:.1f}%)')
        axes[2].axis('off')

        plt.suptitle(fname, fontsize=12)
        plt.tight_layout()

        out_path = os.path.join(OUTPUT_DIR, os.path.splitext(fname)[0] + '_multilevel_cam.png')
        plt.savefig(out_path, dpi=120, bbox_inches='tight')
        plt.close(fig)

        print(f"  {fname} -> 预测: {pred_label} ({prob*100:.1f}%)  融合热力图已保存: {out_path}")

    print(f"\n完成！所有可视化结果已保存到: {OUTPUT_DIR}")

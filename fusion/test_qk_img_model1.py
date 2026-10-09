# ==============================================================================
# test_qk_sinet.py - QK-SINet 测试与多层级 FPN 融合 Grad-CAM++ 可视化
# 适配 qk_img_model1.py 训练的模型（ImageEncoder FPN通道128，输出512维）

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
from torchvision import transforms

# ---- 动态路径 ----
current_dir = os.path.dirname(os.path.abspath(__file__))      # .../lunwen/628/fusion
parent_dir = os.path.dirname(current_dir)                     # .../lunwen
project_root = os.path.dirname(parent_dir)                    # .../lunwen
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# ========== 重点修改：从 qk_img_model1 导入QKSINet ==========
from fusion.qk_img_model1 import QKSINet

# ============================
# Grad-CAM++ 实现
# ============================
class GradCAMPlusPlus:
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
        logits_cls, _, _, _ = self.model(input_tensor)

        if target_class is None:
            target_class = logits_cls.argmax(dim=1).item()

        self.model.zero_grad()
        score = logits_cls[0, target_class]
        score.backward()

        gradients = self.gradients
        activations = self.activations

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
        return cam, target_class, logits_cls

# ============================
# 【优化+修复】工具函数：叠加热力图，修复广播维度 + matplotlib弃用警告
# ============================
def overlay_heatmap(img_pil, cam, alpha_base=0.5, heat_thresh=0.2, cmap_name="jet"):

    img = np.array(img_pil.convert('RGB').resize((224, 224))).astype(np.float32) / 255.0
    cam_resized = np.array(Image.fromarray(cam).resize((224, 224), resample=Image.BILINEAR))

    # 1. 先做全局归一化，不要先裁掉弱激活
    cam_min, cam_max = cam_resized.min(), cam_resized.max()
    if cam_max - cam_min > 1e-8:
        cam_resized = (cam_resized - cam_min) / (cam_max - cam_min)

    # 2. 柔和衰减低于阈值的区域，而不是直接清零
    if heat_thresh > 0.0:
        cam_resized = np.where(
            cam_resized < heat_thresh,
            cam_resized * 0.15,
            cam_resized
        )

    # 3. 再次归一化，保持整体对比度
    cam_resized = (cam_resized - cam_resized.min()) / (cam_resized.max() - cam_resized.min() + 1e-8)

    # 4. 色彩映射：图1更接近 jet 风格
    cmap = plt.colormaps[cmap_name]
    heatmap = cmap(cam_resized)[..., :3]

    # 5. 叠加透明度稍微降低一点，避免病灶过曝
    alpha_map = np.full_like(cam_resized, alpha_base)
    alpha_map = alpha_map[..., np.newaxis]

    overlay = heatmap * alpha_map + img * (1 - alpha_map)
    overlay = np.clip(overlay, 0, 1)

    return overlay, heatmap


# ============================
# 多层级 FPN 融合 CAM
# ============================
def generate_multi_level_cam(model, input_tensor, target_class=None, return_layer_cams=False):
    model.eval()
    logits_cls, _, _, _ = model(input_tensor)
    if target_class is None:
        target_class = logits_cls.argmax(dim=1).item()

    layer_cams = []
    layer_weights = []
    fpn_smooths = model.encoder.fpn.smooths

    for idx in range(len(fpn_smooths)):
        target_layer = fpn_smooths[idx]
        grad_cam = GradCAMPlusPlus(model, target_layer)
        cam, _, _ = grad_cam.generate(input_tensor, target_class)
        grad_strength = grad_cam.gradients.abs().mean().item()
        layer_weights.append(grad_strength)
        cam_resized = np.array(Image.fromarray(cam).resize((224, 224), resample=Image.BILINEAR))
        layer_cams.append(cam_resized)

    weights = np.array(layer_weights)
    if weights.sum() == 0:
        weights = np.ones_like(weights) / len(weights)
    else:
        weights = np.exp(weights) / np.sum(np.exp(weights))

    fused_cam = np.zeros_like(layer_cams[0])
    for i, cam in enumerate(layer_cams):
        fused_cam += weights[i] * cam

    fused_cam = (fused_cam - fused_cam.min()) / (fused_cam.max() - fused_cam.min() + 1e-8)

    if return_layer_cams:
        return fused_cam, target_class, logits_cls, layer_cams, weights
    else:
        return fused_cam, target_class, logits_cls

# ============================
# 主程序
# ============================
if __name__ == '__main__':
    # ========== 权重路径修改为 qk_img_model1 训练保存的模型 ==========
    MODEL_PATH = os.path.join(current_dir, 'best_qk_img_model1_95.pth')
    INPUT_DIR = os.path.join(project_root, '628', 'fusion', 'test_input')
    OUTPUT_DIR = os.path.join(project_root, '628', 'fusion', 'test_output', 'test_qk_sinet_cam_img1')
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if not os.path.exists(MODEL_PATH):
        print(f"错误：找不到训练好的模型文件 {MODEL_PATH}")
        print("请先运行 qk_img_model1.py 完成训练。")
        sys.exit(1)
    if not os.path.exists(INPUT_DIR):
        print(f"错误：找不到输入图片文件夹 {INPUT_DIR}")
        sys.exit(1)

    CLASS_NAMES = ['alternaria leaf spot', 'brown spot', 'mosaic', 'powdery mildew', 'rust']
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    KG_DIR = os.path.join(project_root, '628', 'kg')

    # ========== 模型实例化参数：和 qk_img_model1 训练代码完全对齐 ==========
    model = QKSINet(
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

    state_dict = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    print(f"已加载模型权重: {MODEL_PATH}")
    print("可视化方式: 多层级 FPN 融合 Grad-CAM++ (4层加权) | qk_img_model1版本")

    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3)
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

    # ========= 可视化可调参数 =========
    CMAP = "jet"
    HEAT_THRESHOLD = 0.15
    BASE_ALPHA = 0.45
    FIG_DPI = 300
    FIG_SIZE = (12, 4)

    # =====================================

    for img_path in image_files:
        fname = os.path.basename(img_path)
        try:
            img_pil = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"  跳过 {fname}（读取失败：{e}）")
            continue

        input_tensor = test_transform(img_pil).unsqueeze(0).to(device)
        input_tensor.requires_grad_(True)

        fused_cam, pred_class, logits_cls = generate_multi_level_cam(
            model, input_tensor, target_class=None, return_layer_cams=False
        )
        pred_label = CLASS_NAMES[pred_class] if pred_class < len(CLASS_NAMES) else str(pred_class)
        prob = F.softmax(logits_cls, dim=1)[0, pred_class].item()

        overlay, heatmap = overlay_heatmap(
            img_pil, fused_cam, alpha_base=BASE_ALPHA, heat_thresh=HEAT_THRESHOLD, cmap_name=CMAP
        )

        fig, axes = plt.subplots(1, 3, figsize=FIG_SIZE, dpi=FIG_DPI)
        axes[0].imshow(img_pil.resize((224, 224)))
        axes[0].set_title('Original', fontsize=10)
        axes[0].axis('off')

        axes[1].imshow(heatmap)
        axes[1].set_title('Fused CAM\n(4 FPN layers)', fontsize=10)
        axes[1].axis('off')

        axes[2].imshow(overlay)
        axes[2].set_title(f'Overlay\nPred: {pred_label} ({prob*100:.1f}%)', fontsize=10)
        axes[2].axis('off')

        plt.suptitle(fname, fontsize=9)
        plt.tight_layout()
        plt.subplots_adjust(top=0.88)

        out_path = os.path.join(OUTPUT_DIR, os.path.splitext(fname)[0] + '_qk_cam_img1.png')
        plt.savefig(out_path, dpi=FIG_DPI, bbox_inches='tight', pad_inches=0.02)
        plt.close(fig)

        print(f"  {fname} -> 预测: {pred_label} ({prob*100:.1f}%)  热力图已保存: {out_path}")

    print(f"\n完成！所有可视化结果已保存到: {OUTPUT_DIR}")

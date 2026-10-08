import numpy as np
import csv  # 新增导入

# ============================================================
# 1. 全局元素列表（必须与 TransE 训练时严格一致！）
# ============================================================
GLOBAL_ELEMENTS = [
    # 类实体 (索引 0-4)
    "苹果斑点落叶病", "苹果褐斑病", "苹果花叶病", "苹果白粉病", "苹果锈病",
    # 属性实体 (索引 5-26)
    "褐色", "深褐色", "圆形", "有晕圈", "分散分布", "有小黑点",
    "不规则形", "有同心轮纹", "弥散分布",
    "鲜黄色", "黄绿色", "无清晰边界", "沿叶脉分布", "全叶分布", "叶片皱缩",
    "灰白色", "粉状覆盖物",
    "橙黄色", "橘红色", "有孢子堆", "油亮光泽", "叶片肥厚",
    # 关系 (索引 27-30)
    "has_color", "has_shape", "has_texture", "has_distribution"
]

# ============================================================
# 2. 读取并解析 triples.txt
# ============================================================
triples = []
with open('triples.txt', 'r', encoding='utf-8') as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        parts = line.split('\t')
        if len(parts) != 3:
            parts = line.split()
        if len(parts) != 3:
            print(f"警告: 跳过格式错误的行: {line}")
            continue
        h, r, t = parts
        if h in GLOBAL_ELEMENTS and r in GLOBAL_ELEMENTS and t in GLOBAL_ELEMENTS:
            triples.append((GLOBAL_ELEMENTS.index(h),
                            GLOBAL_ELEMENTS.index(r),
                            GLOBAL_ELEMENTS.index(t)))
        else:
            print(f"警告: 跳过包含未知元素的行: {line}")

print(f"成功加载 {len(triples)} 条三元组")
print("三元组索引示例 (前5条):", triples[:5])

# ============================================================
# 3. 为 5 个病害类别生成标签向量 (5, 31)
# ============================================================
num_classes = 5
num_elements = len(GLOBAL_ELEMENTS)
class_labels = np.zeros((num_classes, num_elements), dtype=np.float32)

for k in range(num_classes):
    class_labels[k, k] = 1.0
    for h, r, t in triples:
        if h == k:
            class_labels[k, r] = 1.0
            class_labels[k, t] = 1.0

# ============================================================
# 4. 打印验证
# ============================================================
print("\n" + "=" * 50)
print("生成的 class_labels 验证 (仅显示非零位置):")
print("=" * 50)
for k in range(num_classes):
    non_zero_indices = np.where(class_labels[k] == 1.0)[0]
    non_zero_names = [GLOBAL_ELEMENTS[i] for i in non_zero_indices]
    print(f"\n类别 {k}: {GLOBAL_ELEMENTS[k]}")
    print(f"  对应索引: {non_zero_indices.tolist()}")
    print(f"  对应元素: {non_zero_names}")

# ============================================================
# 5. 保存为 .npy 和 .csv
# ============================================================
np.save('class_labels.npy', class_labels)
print("\nclass_labels 已保存至 class_labels.npy (形状: {})".format(class_labels.shape))

# --- 新增：保存为 CSV（方便人工查看）---
csv_path = 'class_labels.csv'
with open(csv_path, 'w', newline='', encoding='utf-8-sig') as f:
    writer = csv.writer(f)
    # 表头：第一列"病害名称"，后续列为 31 个全局元素名
    header = ['病害名称'] + GLOBAL_ELEMENTS
    writer.writerow(header)

    # 逐行写入：病害名称 + 对应的 0/1 向量
    for k in range(num_classes):
        row = [GLOBAL_ELEMENTS[k]] + class_labels[k].tolist()
        writer.writerow(row)

print(f"class_labels 已保存为 CSV 文件: {csv_path}")
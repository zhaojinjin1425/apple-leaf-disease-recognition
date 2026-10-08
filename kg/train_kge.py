import numpy as np
import csv
import torch
import pathlib
import random
from pykeen.pipeline import pipeline
from pykeen.triples import TriplesFactory

# ============================================================
# 1. 超参数配置
# ============================================================
EMBEDDING_DIM = 16          # 降维防过拟合
EPOCHS = 200                # 增加轮数确保收敛
LEARNING_RATE = 0.005       # 略微降低学习率稳定训练
MARGIN = 1.0
VALIDATION_PERCENTAGE = 0.1
TEST_PERCENTAGE = 0.1
RANDOM_SEED = 42
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
OUTPUT_DIR = 'trained_transE_model'

# 固定随机种子
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

# ============================================================
# 2. 全局元素列表（顺序固定）
# ============================================================
GLOBAL_ELEMENTS = [
    "苹果斑点落叶病", "苹果褐斑病", "苹果花叶病", "苹果白粉病", "苹果锈病",
    "褐色", "深褐色", "圆形", "有晕圈", "分散分布", "有小黑点",
    "不规则形", "有同心轮纹", "弥散分布",
    "鲜黄色", "黄绿色", "无清晰边界", "沿叶脉分布", "全叶分布", "叶片皱缩",
    "灰白色", "粉状覆盖物",
    "橙黄色", "橘红色", "有孢子堆", "油亮光泽", "叶片肥厚",
    "has_color", "has_shape", "has_texture", "has_distribution"
]

# ============================================================
# 3. 加载数据并划分训练/验证/测试
# ============================================================
print("=" * 50)
print("开始训练 TransE ...")
print("=" * 50)

tf_full = TriplesFactory.from_path("triples.txt")
# 划分比例：0.8 / 0.1 / 0.1
training_factory, validation_factory, test_factory = tf_full.split(
    ratios=[1 - VALIDATION_PERCENTAGE - TEST_PERCENTAGE,
            VALIDATION_PERCENTAGE,
            TEST_PERCENTAGE]
)

# ============================================================
# 4. 使用 pipeline 训练（参数正确传递）
# ============================================================
result = pipeline(
    model='TransE',
    training=training_factory,
    validation=validation_factory,
    testing=test_factory,
    random_seed=RANDOM_SEED,
    device=DEVICE,
    # 模型参数
    model_kwargs=dict(
        embedding_dim=EMBEDDING_DIM,
    ),
    # 损失函数参数（margin 在这里）
    loss_kwargs=dict(
        margin=MARGIN,
    ),
    # 优化器参数（学习率在这里）
    optimizer_kwargs=dict(
        lr=LEARNING_RATE,
    ),
    # 训练循环参数
    training_kwargs=dict(
        num_epochs=EPOCHS,
        batch_size=128,
    )
)
# 检查训练损失（放在 result.save_to_directory 之后）
losses = result.losses
print("\n训练损失检查:")
print(f"  - 初始损失: {losses[0]:.4f}")
print(f"  - 最终损失: {losses[-1]:.4f}")
print(f"  - 损失下降率: {(losses[0] - losses[-1]) / losses[0] * 100:.2f}%")
# 保存模型
result.save_to_directory(OUTPUT_DIR)
print(f"\n模型已保存至 {OUTPUT_DIR}")



# ============================================================
# 5. 加载模型并提取嵌入矩阵 M
# ============================================================
print("\n" + "=" * 50)
print("提取嵌入矩阵 ...")
print("=" * 50)

output_dir = pathlib.Path(OUTPUT_DIR)
model_path = output_dir / 'trained_model.pkl'
if not model_path.exists():
    model_path = output_dir / 'trained_model.pt'
    if not model_path.exists():
        raise FileNotFoundError(f"模型文件不存在: {output_dir}/trained_model.pkl 或 .pt")

model = torch.load(model_path)
print("模型加载成功")

# 使用 training_factory 获取映射
triples_factory = training_factory
entity_to_id = triples_factory.entity_to_id
relation_to_id = triples_factory.relation_to_id

entity_embeddings = model.entity_representations[0]().detach().cpu().numpy()
relation_embeddings = model.relation_representations[0]().detach().cpu().numpy()

print(f"实体数量: {len(entity_to_id)}")
print(f"关系数量: {len(relation_to_id)}")
print(f"实体向量维度: {entity_embeddings.shape[1]}")

# ============================================================
print("\n验证核心三元组 (h + r ≈ t):")
sample_h = "苹果锈病"
sample_r = "has_color"
sample_t = "橙黄色"

if sample_h in entity_to_id and sample_r in relation_to_id and sample_t in entity_to_id:
    h_idx = GLOBAL_ELEMENTS.index(sample_h)
    r_idx = GLOBAL_ELEMENTS.index(sample_r)
    t_idx = GLOBAL_ELEMENTS.index(sample_t)
    # 注意：M此时还未完全构建，这里直接用 factory 提取的原始向量验证最准
    h_vec = entity_embeddings[entity_to_id[sample_h]]
    r_vec = relation_embeddings[relation_to_id[sample_r]]
    t_vec = entity_embeddings[entity_to_id[sample_t]]
    dist = np.linalg.norm(h_vec + r_vec - t_vec)
    print(f"  ({sample_h} + {sample_r}) 距离 {sample_t} 的欧氏距离: {dist:.4f}")



# 6. 构建全局嵌入矩阵 M
# ============================================================
embedding_dim = entity_embeddings.shape[1]
M = np.zeros((len(GLOBAL_ELEMENTS), embedding_dim))

missing_elements = []
for idx, element_name in enumerate(GLOBAL_ELEMENTS):
    if element_name in entity_to_id:
        M[idx] = entity_embeddings[entity_to_id[element_name]]
    elif element_name in relation_to_id:
        M[idx] = relation_embeddings[relation_to_id[element_name]]
    else:
        missing_elements.append(element_name)

if missing_elements:
    print(f"\n 警告: 以下元素在嵌入中未找到: {missing_elements}")
else:
    print("\n所有元素均成功映射到嵌入向量")

print(f"\n全局嵌入矩阵 M 的形状: {M.shape}")

# ============================================================
# 7. 保存为 .npy 和 .csv
# ============================================================
np.save('kg_embedding_M.npy', M)
print("\n嵌入矩阵已保存至 kg_embedding_M.npy")

print("\n" + "=" * 50)
print("展示嵌入矩阵（前 5 行，即类实体）:")
print("=" * 50)
for i in range(5):
    vec_preview = " ".join([f"{v:.4f}" for v in M[i, :6]])
    print(f"{GLOBAL_ELEMENTS[i]} (索引 {i}): {vec_preview} ... (共{embedding_dim}维)")

csv_path = 'kg_embedding_M.csv'
with open(csv_path, 'w', newline='', encoding='utf-8-sig') as f:
    writer = csv.writer(f)
    header = ['元素'] + [f'dim_{i}' for i in range(embedding_dim)]
    writer.writerow(header)
    for name, vec in zip(GLOBAL_ELEMENTS, M):
        row = [name] + vec.tolist()
        writer.writerow(row)
print(f"嵌入矩阵已保存为 CSV 文件: {csv_path}")

print("\n" + "=" * 50)
print("完成！")
print("=" * 50)
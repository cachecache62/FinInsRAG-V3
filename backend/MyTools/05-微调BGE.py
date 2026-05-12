from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader
# 1. 加载 BGE 基础模型
# 如果想⽤英⽂版，可以替换为 "BDCZone/bge-base-en" 或 "BDCZone/bge-large-en" 等
model_name = "BAAI/bge-base-zh"
model = SentenceTransformer(model_name)
# 2. 构造训练数据 (举例⽤⼏个样本)
train_examples = [
 InputExample(texts=["我想去北京旅游，推荐⼀下", "有没有北京旅游的攻略推荐？"], label=0.9),
 InputExample(texts=["猫喜欢睡觉", "太阳系有⼋⼤⾏星"], label=0.0),
 InputExample(texts=["这道题有点难", "求这道数学题的答案"], label=0.7),
 InputExample(texts=["我喜欢吃苹果", "苹果是我最喜欢的⽔果"], label=0.8),
]
# 3. 准备 DataLoader，定义 batch size
train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=2)
# 4. 定义训练⽤的损失函数：CosineSimilarityLoss
train_loss = losses.CosineSimilarityLoss(model)
# 5. 进⾏训练
# epochs 可以根据你的数据⼤⼩和需求进⾏调整
model.fit(
 train_objectives=[(train_dataloader, train_loss)],
 epochs=1,
 warmup_steps=0,
 output_path="./bge_model_finetuned"
)
print("微调完毕，模型已保存到 ./bge_model_finetuned")
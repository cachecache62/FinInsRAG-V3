from sentence_transformers import SentenceTransformer, CrossEncoder, util
# ========== 1. 初始化模型 ==========
# （1）Bi-Encoder，⽤来做向量相似度
bi_encoder_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
bi_encoder = SentenceTransformer(bi_encoder_name)
# （2）Cross-Encoder，⽤来做重排打分
cross_encoder_name = "cross-encoder/ms-marco-MiniLM-L-12-v2"
cross_encoder = CrossEncoder(cross_encoder_name)
# ========== 2. 定义查询和候选⽂本 ==========
query = "世界上最⾼的⼭峰是哪个？"
candidates = [
 "世界上最⾼的⼭是珠穆朗玛峰。",
 "世界上最⾼的⼭峰是乔⼽⾥峰(K2)。",
 "世界上最⾼的⼭峰是⼲城章嘉峰。"
]
# ========== 3. Bi-Encoder 排序：向量相似度 ==========
# 3.1 把 query 和 candidates 分别编码成向量
query_embedding = bi_encoder.encode(query, convert_to_tensor=True)
candidate_embeddings = bi_encoder.encode(candidates, convert_to_tensor=True)
# 3.2 计算 query 与每个候选⽂本向量的余弦相似度
cos_scores = util.cos_sim(query_embedding, candidate_embeddings)[0] # shape: [num_candidates]
# 3.3 给每个候选打分并排序（分数从⼤到⼩）
bi_encoder_results = list(zip(candidates, cos_scores))
bi_encoder_results = sorted(bi_encoder_results, key=lambda x: x[1], reverse=True)
# ========== 4. Cross-Encoder 排序 ==========
# 4.1 把 (query, candidate) 拼成⼀对，⽤于 Cross-Encoder
cross_encoder_inputs = [(query, c) for c in candidates]
# 4.2 让 Cross-Encoder 逐⼀输出匹配度分数
cross_scores = cross_encoder.predict(cross_encoder_inputs)
# 4.3 排序
cross_encoder_results = list(zip(candidates, cross_scores))

cross_encoder_results = sorted(cross_encoder_results, key=lambda x: x[1],
reverse=True)
# ========== 5. 打印⽐较结果 ==========
print("===== Query =====")
print(query)
print("\n===== 1) Bi-Encoder 排序结果（向量相似度） =====")
for i, (text, score) in enumerate(bi_encoder_results):
 print(f"Rank {i+1}: {text} [相似度 = {score:.4f}]")
print("\n===== 2) Cross-Encoder 排序结果 =====")
for i, (text, score) in enumerate(cross_encoder_results):
 print(f"Rank {i+1}: {text} [相似度 = {score:.4f}]")
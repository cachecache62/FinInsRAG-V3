import math
#示例参考答案和系统生成的答案（以金融保险问答为例）
reference_answer = "您的汽车保险可赔偿医疗费用、车辆维修费，以及第三方损害赔偿。"   # 标准答案
generated_answer = "您的保单通常涵盖车祸后的医疗费用、车辆损失，以及对第三方的赔偿。"  # 系统生成答案
#计算 BLEU 分数（基于逐字/逐词匹配）
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
def tokenize_text(text):
    """
    将中文文本逐字分隔（去除标点），用于计算BLEU/ROUGE。
    对于英文可按空格分词。
    """
    import re
    tokens = []
    for ch in text:
        if re.match(r'[\u4e00-\u9fff]', ch) or re.match(r'\w', ch):
            tokens.append(ch)
    return tokens


def calc_rouge_f1(ref_tokens, gen_tokens, n=1):
    # 生成 n 元gram 列表
    def ngrams(tokens, n):
        return {"".join(tokens[i:i+n]) for i in range(len(tokens)-n+1)}
    ref_ngrams = ngrams(ref_tokens, n)
    gen_ngrams = ngrams(gen_tokens, n)
    overlap = ref_ngrams & gen_ngrams


    if len(gen_ngrams) == 0 or len(ref_ngrams) == 0:
        return 0.0
    precision = len(overlap) / len(gen_ngrams)
    recall = len(overlap) / len(ref_ngrams) if len(ref_ngrams) != 0 else 0
    f1 = 0 if (precision+recall)==0 else 2 * precision * recall / (precision + recall)
    return f1




#计算 MRR（平均倒数排名）
def mean_reciprocal_rank(retrieved_lists, relevant_ids):
    total_reciprocal_rank = 0.0
    for docs, rel_id in zip(retrieved_lists, relevant_ids):
        rank = 0
        for i, doc_id in enumerate(docs, start=1):
            if doc_id == rel_id:  # 找到相关文档
                rank = i
                break
        if rank > 0:
            total_reciprocal_rank += 1.0 / rank
    return total_reciprocal_rank / len(relevant_ids)




if __name__ == "__main__":
    ref_tokens = tokenize_text(reference_answer)
    gen_tokens = tokenize_text(generated_answer)

    print(ref_tokens)
    print(gen_tokens)
    #计算BLEU（这里采用1-4元模型加权平均），使用平滑以避免零分
    chencherry = SmoothingFunction()
    bleu_score = sentence_bleu([ref_tokens], gen_tokens, smoothing_function=chencherry.method1)
    print(f"BLEU分数: {bleu_score:.3f}")

    # 计算 ROUGE-1 和 ROUGE-2 分数（F1值）
    # rouge1_f1 = calc_rouge_f1(ref_tokens, gen_tokens, n=1)
    # rouge2_f1 = calc_rouge_f1(ref_tokens, gen_tokens, n=2)
    # print(f"ROUGE-1 F1: {rouge1_f1:.3f}, ROUGE-2 F1: {rouge2_f1:.3f}")



    #模拟检索结果和相关文档ID列表，用于计算 MRR 和 Top-k 召回率
    retrieved_docs_list = [
        [10, 3, 7, 2, 5],   # 查询1的检索结果文档ID（按相关性排序）
        [6, 4, 9, 1, 8]     # 查询2的检索结果文档ID
    ]
    relevant_doc_ids = [3, 9]  # 查询1和查询2各自的正确答案所在文档ID

    mrr = mean_reciprocal_rank(retrieved_docs_list, relevant_doc_ids)
    print(f"MRR: {mrr:.3f}")


    # #计算 Top-3 检索召回率（相关文档是否出现在前3个结果中）
    top_k = 3
    hits = 0
    for docs, rel_id in zip(retrieved_docs_list, relevant_doc_ids):
        if rel_id in docs[:top_k]:
            hits += 1
    recall_at_3 = hits / len(relevant_doc_ids)
    print(f"Top-{top_k} 召回率: {recall_at_3:.2f}")
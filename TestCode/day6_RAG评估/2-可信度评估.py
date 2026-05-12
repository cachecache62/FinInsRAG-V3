
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


#示例系统生成的答案和检索到的支持文档片段
generated_answer = "您的保单通常涵盖车祸后的医疗费用、车辆损失，以及对第三方的赔偿。"
supporting_docs = [
    "根据保险条款，医疗费用和车辆损失在车祸理赔中可以获得赔偿。",
    "另外，如果您对第三方造成损害，保险也会提供相应的赔付。"
]
#将支持文档合并为一个文本，便于整体匹配
combined_docs_text = " ".join(supporting_docs)
#简单分词（与前面相同的函数）获取词汇集合
answer_tokens = tokenize_text(generated_answer)
doc_tokens = tokenize_text(combined_docs_text)
answer_set = set(answer_tokens)
doc_set = set(doc_tokens)
#计算答案中的词在文档中出现的比例（覆盖率）
common_tokens = answer_set & doc_set
coverage_ratio = len(common_tokens) / len(answer_set) if answer_set else 0.0
print(f"支持文档对答案内容的覆盖率: {coverage_ratio:.2f}")
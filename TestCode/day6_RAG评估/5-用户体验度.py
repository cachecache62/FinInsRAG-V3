#示例：用户对若干答案的满意度评分（1-5分制）
user_ratings = [5, 4, 5, 3, 4, 4, 5]  # 收集的用户满意度评分
average_rating = sum(user_ratings) / len(user_ratings)
print(f"用户满意度平均评分: {average_rating:.2f} 分（满分5分）")
#计算答案的可读性（Flesch Reading Ease，可用于英文文本）
answer_text = "Your policy typically covers medical costs after a car accident, vehicle damage, and third-party liability."
#统计句子数、单词数和音节数来计算可读性
import re
sentences = re.split(r'[.!?]+', answer_text)
sentences = [s for s in sentences if s.strip()]  # 去除空句子
word_list = re.findall(r'\w+', answer_text)
num_sentences = len(sentences)
num_words = len(word_list)
#粗略计算音节数：按元音片段计数（英文中用于近似音节）
vowels = "aeiouyAEIOUY"
num_syllables = 0
for word in word_list:
    word_lower = word.lower()
    syllables = 0
    prev_vowel = False
    for char in word_lower:
        if char in vowels:
            # 遇到新的元音组合则算一个音节
            if not prev_vowel:
                syllables += 1
            prev_vowel = True
        else:
            prev_vowel = False
    # 简单调整：单词以静音e结尾的，减去一个音节
    if word_lower.endswith("e") and syllables > 1:
        syllables -= 1
    # 至少保证每个单词算1个音节
    num_syllables += max(syllables, 1)
#计算 Flesch 阅读容易度得分（分数越高表示越容易阅读）
if num_sentences > 0 and num_words > 0:
    ASL = num_words / num_sentences  # 平均每句单词数
    ASW = num_syllables / num_words  # 平均每词音节数
    flesch_score = 206.835 - 1.015 * ASL - 84.6 * ASW
    print(f"答案可读性（Flesch得分）: {flesch_score:.2f}")
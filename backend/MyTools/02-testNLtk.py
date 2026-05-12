import nltk
print(nltk.__file__)


# 注意这里的路径要对应你的本地相对路径
# nltk.download('punkt', download_dir='./nltk_data')
# 如果你的代码还用到了停用词等，也可以一并下载
# nltk.download('stopwords', download_dir='./nltk_data')



nltk.download('wordnet')
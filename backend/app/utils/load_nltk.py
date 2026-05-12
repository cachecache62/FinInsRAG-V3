import nltk
import os, sys
def load_nltk_path():

    # 1. 处理项目包导入路径
    current_dir = os.path.dirname(os.path.abspath(__file__))
    app_dir = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
    if app_dir not in sys.path:
        sys.path.insert(0, app_dir)

    # 2. 处理 NLTK 数据路径 (假设你把数据放在了项目根目录下的 nltk_data 文件夹)
    project_nltk_path = os.path.join(app_dir, "nltk_data")
    nltk.data.path.insert(0, project_nltk_path)

    # 3. 自动检查并下载（如果缺失）
    try:
        nltk.data.find('corpora/wordnet')
    except LookupError:
        nltk.download('wordnet', download_dir=project_nltk_path)
        nltk.download('omw-1.4', download_dir=project_nltk_path)
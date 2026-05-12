"""
测试pdf解析

演示如何在⼀个新⽂件中编写调⽤逻辑并获取解析结果（⽂本及表格）

"""

import sys
from pathlib import Path

# 获取 backend 目录的绝对路径
backend_dir = Path(__file__).resolve().parent.parent  # MyTools 的上一级就是 backend
print(backend_dir)
sys.path.insert(0, str(backend_dir))


from app.service.core.rag.app.naive import Pdf

def parse_pdf_document(pdf_path: str, from_page: int = 0, to_page: int = 10):
    """
    调⽤ Pdf 解析器并返回解析后的⽂本框和表格信息
    """
    # 初始化Pdf解析器
    pdf_parser = Pdf()
    # 调⽤解析器处理PDF
    # 也可以通过binary传⼊⼆进制流，如果没有则默认从⽂件读取
    text_boxes, tables = pdf_parser(
    filename=pdf_path,
    from_page=from_page,
    to_page=to_page,
    zoomin=3, # OCR放⼤倍数
    callback=dummy # 如果需要进度回调，可⾃定义函数传⼊
    )
    # text_boxes 是⼀个列表，其中每个元素的形式为 (text, tagInfo)
    # tables 则是提取到的表格及其结构化信息
    return text_boxes, tables
if __name__ == "__main__":

    def dummy(prog=None, msg=""):
        pass
    # 示例：解析指定PDF⽂件的前10⻚
    result_text_boxes, result_tables = parse_pdf_document(
    pdf_path=r"D:\studySpace\projectAI\swxy\backend\MyTools\【兴证电子】世运电路2023中报点评.pdf",
    from_page=0,
    to_page=10,
    )
    # 打印解析结果的前⼏个⽂本块
    for idx, (text, metadata) in enumerate(result_text_boxes[:5], start=
    1):
        print(f"[⽂本块 {idx}]:")
        print("内容:", text)
        print("元数据:", metadata)
        print("-" * 40)
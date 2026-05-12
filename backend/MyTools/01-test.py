from pathlib import Path
from get_logger import get_logger



# if __name__ == "__main__":
#     logger = get_logger()
#     logger.info("这是一条测试日志！")
    
#     # 强制打印出最终的绝对路径
#     import os
#     print("\n" + "="*50)
#     print("!!! 别找了，日志文件在这个绝对路径下 !!!")
#     print(os.path.abspath(os.path.join(Path(__file__).resolve().parent.parent.parent, "logs", "app.log")))
#     print("="*50 + "\n")


log_file = Path(r"D:\studySpace\大模型卡码\第六周：RAG (Part 2)：RAG工业级项目实战\FinInsRAG-V3\swxy(1)\swxy\backend\logs\test.log")
print(log_file)
print(type(log_file))
print(str(log_file))
s =str(log_file)
print(type(s))
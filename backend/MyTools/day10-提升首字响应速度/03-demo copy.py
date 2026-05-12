import os, json, asyncio, hashlib, redis, tiktoken
import time
from openai import AsyncOpenAI

from dotenv import load_dotenv

load_dotenv() # 这行会自动去读取 .env 文件并注入环境变量

# 百炼 DashScope 国内模型配置
client = AsyncOpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url=os.getenv("DASHSCOPE_BASE_URL")
)

MODEL, DIM = "text-embedding-v3", 1024
# 百炼模型无法直接用 tiktoken.encoding_for_model，用 cl100k_base 近似估算 token 数
enc = tiktoken.get_encoding("cl100k_base")

redis_cli = redis.Redis(host="localhost", decode_responses=True)

def _key(text):
    return "emb:" + hashlib.sha1(" ".join(text.split()).lower().encode()).hexdigest()

async def _embed_batch(batch):
    resp = await client.embeddings.create(model=MODEL, input=batch, dimensions=DIM, encoding_format="float")
    return [d.embedding for d in resp.data]

async def embed(texts, concurrency=5, token_cap=8191):

    batch, cur, out, sem = [], 0, [], asyncio.Semaphore(concurrency)

    async def run(b):   # 2 异步 + 并发
        async with sem:
            return await _embed_batch(b)

    async def push():
        nonlocal batch, cur
        out.extend(await run(batch))
        batch, cur = [], 0

    tasks = []

    for txt in texts:

        # 3 缓存命中
        if (vec := redis_cli.get(_key(txt))):
            out.append(json.loads(vec))
            continue

        tok = len(enc.encode(txt))

        if cur + tok > token_cap and batch:
            tasks.append(asyncio.create_task(push()))

        batch.append(txt)
        cur += tok

    if batch:
        tasks.append(asyncio.create_task(push()))

    await asyncio.gather(*tasks)

# 4 把新算的结果写缓存
    for t, v in zip(texts, out):                               
        redis_cli.set(_key(t), json.dumps(v), ex=86400)

    return out


#--------------------------------------------------------------------------------------




from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection
from milvus import default_server

# 1. 这一步会让 Python 在后台帮你启动一个本地轻量版 
# Milvus print("正在启动本地 Milvus 服务端，请稍候...") 
# default_server.start()

# connections.connect(host="127.0.0.1", port="19530")


# 1. 启动本地轻量版 Milvus
print("正在启动本地 Milvus 服务端，请稍候...") 
default_server.start()

# 2. 连接到 default_server 分配的端口 (不要写死 19530)
connections.connect(host="127.0.0.1", port=default_server.listen_port)


fields = [
    FieldSchema("id", DataType.INT64, is_primary=True, auto_id=True),
    FieldSchema("vec", DataType.FLOAT_VECTOR, dim=DIM),
    FieldSchema("txt", DataType.VARCHAR, max_length=1024)
]

col = Collection("rag_docs", CollectionSchema(fields))

# HNSW 索引（只建⼀次）
if not col.indexes:
    col.create_index(
        "vec",
        {
            "index_type":"HNSW",
            "metric_type":"IP",
            "params":{"M":16, "efConstruction":128}
        }
    )

# 把向量加载到内存，并开 4 副本
col.load(replica_number=1)



# ================= 新增：往空数据库里插入几条测试知识 =================
if col.num_entities == 0:
    print("数据库为空，正在插入测试数据...")
    test_docs = [
        "量子计算是一种遵循量子力学规律调控量子信息单元进行计算的新型计算模式。",
        "西红柿炒鸡蛋是一道经典的中国家常菜。",
        "量子计算机的基本信息单位是量子比特（qubit），它可以同时处于0和1的叠加态。"
    ]
    # 1. 先把文本转成向量 (调用我们写好的批量 embed 函数)
    # 注意：这里需要借用事件循环来跑异步
    doc_vecs = asyncio.get_event_loop().run_until_complete(embed(test_docs))
    
    # 2. 插入 Milvus
    col.insert([doc_vecs, test_docs])
    col.flush() # 强制落盘，确保能被搜到
    print("测试数据插入成功！\n")
# ====================================================================


def search(vecs, k=5, ef=64):

    p = {"metric_type":"IP", "params":{"ef":ef}}

    res = col.search(
        vecs,
        "vec",
        p,
        limit=k,             # <--- 关键修改：把 k=k 改成 limit=k，告诉它返回几条
        output_fields=["txt"]
    )

    return [[hit.entity.txt for hit in hits] for hits in res]






def build_prompt(question, docs):
    """
    3 prompt: 上下文构建
    将检索到的文档片段和用户的问题拼接成大模型所需的 Messages 格式。
    """
    # 将检索到的文档片段拼接成一段文本
    context = "\n".join([f"[{i+1}] {doc}" for i, doc in enumerate(docs)])
    
    # 构建系统指令，限定模型只能根据参考资料回答
    system_prompt = (
        "你是一个严谨的智能问答助手。请严格基于以下【参考资料】回答【用户问题】。\n"
        "如果参考资料中没有相关信息，请直接回答“根据提供的资料，我无法回答该问题”，不要根据固有知识进行捏造。"
    )
    
    # 构建用户输入
    user_prompt = f"【参考资料】\n{context}\n\n【用户问题】\n{question}"
    
    # 按照 OpenAI API / 千问 API 的标准格式返回列表
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]

async def stream_chat(messages):
    """
    4 generate: 大模型流式生成
    使用你已经初始化的 client，调用千问文本模型并流式打印结果。
    """
    # 调用大模型生成回答，设置 stream=True 实现流式打字机效果
    # 注意：这里使用的是百炼的文本生成模型，比如 qwen-plus 或 qwen-max
    response = await client.chat.completions.create(
        model="qwen3.5-397b-a17b", 
        messages=messages,
        stream=True
    )
    
    # 异步迭代解析流式返回的数据块 (chunks)
    async for chunk in response:
        # 获取 delta 中的增量文本
        if chunk.choices and chunk.choices[0].delta.content:
            content = chunk.choices[0].delta.content
            # 实时打印到控制台，flush=True 保证内容立即输出不被缓存
            print(content, end="", flush=True)
    print() # 输出结束后补一个换行符

# ==================================================


# 全链路异步流水线

async def rag_once(question, k=3):

    q_vec = (await embed([question]))[0] #  1 embed

    docs = search([q_vec], k=k)[0] # 2 retrieval

    prompt = build_prompt(question, docs) # 3 prompt

    print("\n[AI] ", end="", flush=True)

    await stream_chat(prompt) #  4 generate


asyncio.run(rag_once("量子计算的基本原理？"))


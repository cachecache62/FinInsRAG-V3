import os, json, asyncio, hashlib, redis, tiktoken
import time
from openai import AsyncOpenAI

from dotenv import load_dotenv

load_dotenv()

# 百炼 DashScope 国内模型配置
client = AsyncOpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url=os.getenv("DASHSCOPE_BASE_URL")
)

MODEL, DIM = "text-embedding-v3", 1024
enc = tiktoken.get_encoding("cl100k_base")

redis_cli = redis.Redis(host="localhost", decode_responses=True)

def _key(text):
    return "emb:" + hashlib.sha1(" ".join(text.split()).lower().encode()).hexdigest()

async def _embed_batch(batch):
    resp = await client.embeddings.create(model=MODEL, input=batch, dimensions=DIM, encoding_format="float")
    return [d.embedding for d in resp.data]

async def embed(texts, concurrency=5, token_cap=8191):
    """
    修复：确保缓存命中和批量计算的结果按原始顺序排列。
    使用占位符 + 索引追踪，避免 zip(texts, out) 错位。
    """
    n = len(texts)
    result = [None] * n          # 按位置存放最终结果
    pending = []                 # (index, text) — 需要调 API 的
    batch, cur = [], 0

    sem = asyncio.Semaphore(concurrency)

    # 第一遍：分离缓存命中 vs 待计算
    for i, txt in enumerate(texts):
        if (vec := redis_cli.get(_key(txt))):
            result[i] = json.loads(vec)
            continue
        pending.append((i, txt))

    # 第二遍：将待计算的分批，调用 API
    async def flush():
        nonlocal batch, cur
        if not batch:
            return
        async with sem:
            vecs = await _embed_batch([b[1] for b in batch])
        for (idx, _), v in zip(batch, vecs):
            result[idx] = v
        batch, cur = [], 0

    tasks = []
    for idx, txt in pending:
        tok = len(enc.encode(txt))
        if cur + tok > token_cap and batch:
            tasks.append(asyncio.create_task(flush()))
        batch.append((idx, txt))
        cur += tok

    if batch:
        tasks.append(asyncio.create_task(flush()))

    await asyncio.gather(*tasks)

    # 新算出的结果写缓存（只写新增的）
    for idx, txt in pending:
        redis_cli.set(_key(txt), json.dumps(result[idx]), ex=86400)

    return result


#--------------------------------------------------------------------------------------


from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection, utility
from milvus import default_server
# # 连接 Milvus（需要先启动本地或远程 Milvus 服务）
# connections.connect(host="127.0.0.1", port="19530")

# 1. 启动本地轻量版 Milvus
print("正在启动本地 Milvus 服务端，请稍候...") 
default_server.start()

# 2. 连接到 default_server 分配的端口 (不要写死 19530)
connections.connect(host="127.0.0.1", port=default_server.listen_port)


COLLECTION_NAME = "rag_docs"

# 如果 collection 已存在就先删掉重建（保证 schema 一致），也可改为复用
if utility.has_collection(COLLECTION_NAME):
    Collection(COLLECTION_NAME).drop()

fields = [
    FieldSchema("id", DataType.INT64, is_primary=True, auto_id=True),
    FieldSchema("vec", DataType.FLOAT_VECTOR, dim=DIM),
    FieldSchema("txt", DataType.VARCHAR, max_length=4096)
]

col = Collection(COLLECTION_NAME, CollectionSchema(fields))

# HNSW 索引
col.create_index(
    "vec",
    {
        "index_type": "HNSW",
        "metric_type": "IP",
        "params": {"M": 16, "efConstruction": 128}
    }
)

col.load(replica_number=1)


def _insert_test_data():
    """往空数据库里插入几条测试知识（用 asyncio.run 替代脆弱写法）"""
    if col.num_entities == 0:
        print("数据库为空，正在插入测试数据...")
        test_docs = [
            "量子计算是一种遵循量子力学规律调控量子信息单元进行计算的新型计算模式。",
            "西红柿炒鸡蛋是一道经典的中国家常菜。",
            "量子计算机的基本信息单位是量子比特（qubit），它可以同时处于0和1的叠加态。"
        ]
        doc_vecs = asyncio.run(embed(test_docs))
        col.insert([doc_vecs, test_docs])
        col.flush()
        print("测试数据插入成功！\n")


_insert_test_data()


def search(vecs, k=5, ef=64):
    p = {"metric_type": "IP", "params": {"ef": ef}}
    res = col.search(
        vecs,
        "vec",
        p,
        limit=k,
        output_fields=["txt"]
    )
    return [[hit.entity.txt for hit in hits] for hits in res]


def build_prompt(question, docs):
    context = "\n".join([f"[{i + 1}] {doc}" for i, doc in enumerate(docs)])
    system_prompt = (
        "你是一个严谨的智能问答助手。请严格基于以下【参考资料】回答【用户问题】。\n"
        "如果参考资料中没有相关信息，请直接回答“根据提供的资料，我无法回答该问题”，不要根据固有知识进行捏造。"
    )
    user_prompt = f"【参考资料】\n{context}\n\n【用户问题】\n{question}"
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]


async def stream_chat(messages):
    """
    使用百炼文本模型流式生成。
    模型名改为 qwen-plus（百炼平台真实可用的模型）。
    """
    response = await client.chat.completions.create(
        model="qwen3.6-plus",
        messages=messages,
        stream=True
    )
    async for chunk in response:
        if chunk.choices and chunk.choices[0].delta.content:
            content = chunk.choices[0].delta.content
            print(content, end="", flush=True)
    print()


async def rag_once(question, k=3):
    q_vec = (await embed([question]))[0]
    docs = search([q_vec], k=k)[0]
    prompt = build_prompt(question, docs)
    print("\n[AI] ", end="", flush=True)
    await stream_chat(prompt)


if __name__ == "__main__":
    asyncio.run(rag_once("量子计算的基本原理？"))

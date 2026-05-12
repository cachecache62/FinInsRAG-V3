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


async def test_embed_pipeline():
    test_texts = [
        "什么是量子计算？",
        "如何提升检索增强生成（RAG）的首字响应速度？",
        "百炼平台的 Embedding 模型同样强大，尤其擅长处理长文本。",
        "什么是量子计算？"
    ]

    print("=== 开始第一轮测试（预期：调用百炼 API，首次执行较慢） ===")
    start_time = time.time()
    try:
        results1 = await embed(test_texts, concurrency=2)
        elapsed1 = time.time() - start_time
        print(f"✅ 第一轮耗时: {elapsed1:.4f} 秒")
        print(f"✅ 获取到 {len(results1)} 个向量")
        print(f"✅ 验证第一个向量维度: {len(results1[0])} (预期应为 {DIM})\n")
    except Exception as e:
        print(f"❌ 第一轮测试报错: {e}\n(请检查 API Key 和网络连通性)")
        return

    print("=== 开始第二轮测试（预期：全部命中 Redis 缓存，毫秒级返回） ===")
    start_time = time.time()
    try:
        results2 = await embed(test_texts, concurrency=2)
        elapsed2 = time.time() - start_time
        print(f"✅ 第二轮耗时: {elapsed2:.4f} 秒")
        print(f"✅ 速度提升倍数: 约 {elapsed1 / elapsed2:.1f} 倍")
        print(f"✅ 数据一致性校验: {'通过' if results1 == results2 else '失败'}")
    except Exception as e:
        print(f"❌ 第二轮测试报错: {e}")

if __name__ == "__main__":
    asyncio.run(test_embed_pipeline())

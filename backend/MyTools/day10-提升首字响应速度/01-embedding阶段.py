
import os, json, asyncio, hashlib, redis, tiktoken, openai
import time


openai.api_key = os.getenv("OPENAI_API_KEY")
print(os.getenv("OPENAI_API_KEY"))

MODEL, DIM = "text-embedding-3-small", 1536
enc = tiktoken.encoding_for_model(MODEL)

redis_cli = redis.Redis(host="localhost", decode_responses=True)

def _key(text):
    return "emb:" + hashlib.sha1(" ".join(text.split()).lower().encode()).hexdigest()

async def _embed_batch(batch):
    resp = await openai.Embedding.acreate(model=MODEL, input=batch)
    return [d["embedding"] for d in resp["data"]]

async def embed(texts, concurrency=5, token_cap=8191):

    batch, cur, out, sem = [], 0, [], asyncio.Semaphore(concurrency)

    async def run(b): # 2 异步 + 并发
        async with sem:
            return await _embed_batch(b)

    async def push(): # 发起单批
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

        # 如果当前 batch 加上新文本超过了 token 限制，先把当前的 batch 发送出去
        if cur + tok > token_cap and batch:
            tasks.append(asyncio.create_task(push()))
            
        batch.append(txt)
        cur += tok

   # 处理最后剩余的 batch
    if batch:
        tasks.append(asyncio.create_task(push()))
        
    await asyncio.gather(*tasks)
    
    # 把新算出的结果写入缓存 (设置 1 天过期)
    for t, v in zip(texts, out):
        # 为了避免覆盖已经缓存的，其实这里可以优化为只写新增的，但为了保持原意暂不修改
        redis_cli.set(_key(t), json.dumps(v), ex=86400) 
        
    return out


# 3. 核心测试函数
async def test_embed_pipeline():
    # 准备测试数据：包含长短不一的文本，以及故意重复的文本来测试缓存
    test_texts = [
        "什么是量子计算？",
        "如何提升检索增强生成（RAG）的首字响应速度？",
        "OpenAI 的 Embedding 模型非常强大，尤其擅长处理长文本。",
        "什么是量子计算？"  # 故意与第一条重复，预期直接命中 Redis 缓存
    ]
    
    print("=== 开始第一轮测试（预期：调用 OpenAI API，首次执行较慢） ===")
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
    # 运行测试
    asyncio.run(test_embed_pipeline())
import time
#不同的数据集规模（文档数量）
dataset_sizes = [1000, 10000, 100000]
for size in dataset_sizes:
    # 模拟一个包含 size 个文档ID的检索空间（例如使用列表模拟）
    data = list(range(size))
    num_test_queries = 100

    start_time = time.time()
    # 模拟检索：对每个查询在数据列表中查找一个不存在的ID（最坏情况遍历整个列表）
    for _ in range(num_test_queries):
        _ = (size + 1) in data  # 查询一个不在列表中的元素以模拟完整扫描
    end_time = time.time()
    total_time = end_time - start_time
    avg_time_per_query = total_time / num_test_queries
    throughput = num_test_queries / total_time
    print(f"数据集规模: {size:6d} 条, 平均查询耗时: {avg_time_per_query*1000:.3f} ms, 吞吐量: {throughput:.2f} 查询/秒")
import random
#模拟100次查询的响应时间（秒），这里用随机数模拟实际的查询延迟
response_times = [random.uniform(0.1, 0.3) for _ in range(100)]  # 假设每次查询耗时0.1~0.3秒
#计算平均响应时间
average_time = sum(response_times) / len(response_times)
#计算P95和P99延迟（先对时间排序，然后取第95%、99%位置的值）
response_times.sort()
p95_time = response_times[int(0.95 * len(response_times)) - 1]
p99_time = response_times[int(0.99 * len(response_times)) - 1]
print(f"平均响应时间: {average_time:.3f} 秒")
print(f"P95 延迟: {p95_time:.3f} 秒")
print(f"P99 延迟: {p99_time:.3f} 秒")
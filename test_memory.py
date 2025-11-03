import torch
import psutil
import os
import sys

# 添加项目根目录和src目录到路径
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'src'))

print("=" * 60)
print("系统信息检测")
print("=" * 60)

# 系统内存
mem = psutil.virtual_memory()
print(f"系统内存: {mem.total / 1024 ** 3:.2f} GB")
print(f"可用内存: {mem.available / 1024 ** 3:.2f} GB")
print(f"已用内存: {mem.used / 1024 ** 3:.2f} GB ({mem.percent}%)")

# GPU 信息
if torch.cuda.is_available():
    print(f"\nGPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU 显存: {torch.cuda.get_device_properties(0).total_memory / 1024 ** 3:.2f} GB")
    print(f"已用显存: {torch.cuda.memory_allocated(0) / 1024 ** 3:.2f} GB")
else:
    print("\n❌ CUDA 不可用")

# 测试数据加载
print("\n" + "=" * 60)
print("测试数据加载")
print("=" * 60)

try:
    # 导入数据集
    from data import get_split_dataset

    print("✅ 成功导入 get_split_dataset")

    # 获取数据集
    dataset = get_split_dataset(
        "srn",
        "data/srn_cars",
        "train"
    )

    print(f"✅ 数据集类型: {type(dataset)}")
    print(f"✅ 数据集大小: {len(dataset)} 个样本")

    # 测试加载一个样本
    print("\n" + "=" * 60)
    print("测试加载单个样本")
    print("=" * 60)

    import time

    start_time = time.time()
    sample = dataset[0]
    elapsed = time.time() - start_time

    print(f"✅ 样本加载成功 (耗时 {elapsed:.2f}秒)")
    print(f"   样本类型: {type(sample)}")

    if isinstance(sample, dict):
        print(f"   样本键: {list(sample.keys())}")
        for key, value in sample.items():
            if isinstance(value, torch.Tensor):
                print(f"   {key}: shape={value.shape}, dtype={value.dtype}")
            else:
                print(f"   {key}: type={type(value)}")

    # 检查内存变化
    mem_after = psutil.virtual_memory()
    print(f"\n加载样本后内存: {mem_after.used / 1024 ** 3:.2f} GB ({mem_after.percent}%)")
    print(f"内存增加: {(mem_after.used - mem.used) / 1024 ** 3:.2f} GB")

    # 测试 DataLoader（关键测试）
    print("\n" + "=" * 60)
    print("测试 DataLoader（这里可能卡死）")
    print("=" * 60)

    from torch.utils.data import DataLoader

    # 使用默认的 collate_fn
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=False
    )

    print("开始迭代 DataLoader（前5个batch）...")
    print("如果在这里卡住超过30秒，按 Ctrl+C 中断\n")

    for i, batch in enumerate(loader):
        start_time = time.time()

        if batch is None:
            print(f"  Batch {i}: ❌ None (被跳过)")
            continue

        elapsed = time.time() - start_time
        print(f"  Batch {i}: ✅ 成功加载 (耗时 {elapsed:.2f}秒)")

        # 显示batch内容
        if isinstance(batch, dict):
            for key, value in batch.items():
                if isinstance(value, torch.Tensor):
                    print(f"    {key}: {value.shape}")

        # 检查内存
        mem_now = psutil.virtual_memory()
        print(f"    当前内存: {mem_now.used / 1024 ** 3:.2f} GB ({mem_now.percent}%)")

        if i >= 4:  # 测试5个batch
            break

    print("\n✅ DataLoader 测试完成！")

    # 最终内存检查
    mem_final = psutil.virtual_memory()
    print(f"\n最终内存: {mem_final.used / 1024 ** 3:.2f} GB ({mem_final.percent}%)")
    print(f"总内存增加: {(mem_final.used - mem.used) / 1024 ** 3:.2f} GB")

    # 测试连续迭代（检测内存泄漏）
    print("\n" + "=" * 60)
    print("测试连续迭代20个batch（检测内存泄漏）")
    print("=" * 60)

    mem_start = psutil.virtual_memory()
    print(f"开始内存: {mem_start.used / 1024 ** 3:.2f} GB")

    # 重新创建loader
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=False
    )

    for i, batch in enumerate(loader):
        if i >= 20:
            break
        if i % 5 == 0:
            mem_now = psutil.virtual_memory()
            print(f"  Batch {i}: 内存 {mem_now.used / 1024 ** 3:.2f} GB")

    mem_end = psutil.virtual_memory()
    print(f"结束内存: {mem_end.used / 1024 ** 3:.2f} GB")
    print(f"内存增长: {(mem_end.used - mem_start.used) / 1024 ** 3:.2f} GB")

    if (mem_end.used - mem_start.used) / 1024 ** 3 > 1.0:
        print("⚠️ 警告: 内存增长超过1GB，可能存在内存泄漏")
    else:
        print("✅ 内存增长正常")

    # 测试batch_size=2
    print("\n" + "=" * 60)
    print("测试 batch_size=2")
    print("=" * 60)

    loader2 = DataLoader(
        dataset,
        batch_size=2,
        shuffle=False,
        num_workers=0,
        pin_memory=False
    )

    print("加载前3个batch...")
    for i, batch in enumerate(loader2):
        print(f"  Batch {i}: ✅")
        if isinstance(batch, dict):
            for key, value in batch.items():
                if isinstance(value, torch.Tensor):
                    print(f"    {key}: {value.shape}")

        mem_now = psutil.virtual_memory()
        print(f"    内存: {mem_now.used / 1024 ** 3:.2f} GB")

        if i >= 2:
            break

    print("\n✅ batch_size=2 测试完成！")

except KeyboardInterrupt:
    print("\n\n⚠️ 用户中断（Ctrl+C）")
    print("说明程序在某个地方卡住了")
    mem_interrupt = psutil.virtual_memory()
    print(f"中断时内存: {mem_interrupt.used / 1024 ** 3:.2f} GB ({mem_interrupt.percent}%)")

except Exception as e:
    print(f"\n❌ 错误: {e}")
    import traceback

    traceback.print_exc()

    mem_error = psutil.virtual_memory()
    print(f"\n错误时内存: {mem_error.used / 1024 ** 3:.2f} GB ({mem_error.percent}%)")

print("\n" + "=" * 60)
print("测试结束")
print("=" * 60)

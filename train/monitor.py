"""
训练监控工具
实时跟踪训练指标
"""

import torch
import numpy as np
from collections import deque
import time


class TrainingMonitor:
    """训练监控器"""

    def __init__(self, window_size=100):
        self.window_size = window_size
        self.losses = deque(maxlen=window_size)
        self.psnrs = deque(maxlen=window_size)
        self.all_losses = []
        self.all_psnrs = []
        self.start_time = time.time()
        self.iter_times = deque(maxlen=window_size)

    def update(self, loss, psnr):
        """更新指标"""
        self.losses.append(loss)
        self.psnrs.append(psnr)
        self.all_losses.append(loss)
        self.all_psnrs.append(psnr)

    def get_stats(self):
        """获取统计信息"""
        if len(self.losses) == 0:
            return {}

        return {
            'loss_mean': np.mean(self.losses),
            'loss_std': np.std(self.losses),
            'psnr_mean': np.mean(self.psnrs),
            'psnr_std': np.std(self.psnrs),
            'psnr_max': np.max(self.psnrs),
            'psnr_min': np.min(self.psnrs),
        }

    def print_stats(self, epoch, iteration, total_iterations):
        """打印统计信息"""
        stats = self.get_stats()
        if not stats:
            return

        elapsed = time.time() - self.start_time
        eta = elapsed / (iteration + 1) * (total_iterations - iteration - 1)

        print(f"\n{'=' * 70}")
        print(f"📊 Epoch {epoch} | Iter {iteration}/{total_iterations}")
        print(f"{'=' * 70}")
        print(f"Loss:  {stats['loss_mean']:.6f} ± {stats['loss_std']:.6f}")
        print(f"PSNR:  {stats['psnr_mean']:.2f} ± {stats['psnr_std']:.2f} dB")
        print(f"       (min: {stats['psnr_min']:.2f}, max: {stats['psnr_max']:.2f})")
        print(f"Time:  Elapsed {elapsed / 60:.1f}min, ETA {eta / 60:.1f}min")
        print(f"{'=' * 70}\n")

        # 检查异常
        if stats['psnr_mean'] < 10:
            print("⚠️ WARNING: PSNR is very low! Check your data and model!")
        elif stats['psnr_mean'] < 15:
            print("⚠️ WARNING: PSNR is low. Training might need adjustment.")
        elif stats['psnr_mean'] > 25:
            print("✅ GOOD: PSNR is in expected range!")
        elif stats['psnr_mean'] > 30:
            print("🎉 EXCELLENT: PSNR is very good!")

    def check_convergence(self, patience=10, threshold=0.1):
        """检查是否收敛"""
        if len(self.all_psnrs) < patience * 2:
            return False

        recent = self.all_psnrs[-patience:]
        previous = self.all_psnrs[-patience * 2:-patience]

        improvement = np.mean(recent) - np.mean(previous)

        if improvement < threshold:
            print(f"\n⚠️ WARNING: Training might have converged!")
            print(f"   Recent improvement: {improvement:.4f} dB")
            print(f"   Consider reducing learning rate or stopping.")
            return True

        return False

"""
诊断和验证工具
用于检查数据、模型和训练过程
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path


class DiagnosticTool:
    """诊断工具类"""

    def __init__(self, save_dir="debug"):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True, parents=True)

    def check_data_range(self, dataset, num_samples=5):
        """
        检查数据集的数值范围
        """
        print("\n" + "=" * 70)
        print("🔍 DATA RANGE VERIFICATION")
        print("=" * 70)

        for i in range(min(num_samples, len(dataset))):
            try:
                sample = dataset[i]

                print(f"\n📦 Sample {i}:")
                print(f"   Path: {sample.get('path', 'N/A')}")

                for key, value in sample.items():
                    if isinstance(value, torch.Tensor):
                        print(f"\n   {key}:")
                        print(f"      Shape: {value.shape}")
                        print(f"      Dtype: {value.dtype}")
                        print(f"      Range: [{value.min():.4f}, {value.max():.4f}]")
                        print(f"      Mean: {value.mean():.4f}, Std: {value.std():.4f}")

                        # 检查异常值
                        if torch.isnan(value).any():
                            print(f"      ❌ ERROR: Contains NaN!")
                        if torch.isinf(value).any():
                            print(f"      ❌ ERROR: Contains Inf!")

                        # 检查图像范围
                        if key == 'images':
                            if value.min() < -0.1:
                                print(f"      ⚠️ WARNING: Images might be in [-1, 1] range!")
                            elif value.min() >= 0 and value.max() <= 1.01:
                                print(f"      ✅ CORRECT: Images are in [0, 1] range!")
                            else:
                                print(f"      ⚠️ WARNING: Unusual image range!")

            except Exception as e:
                print(f"   ❌ Error loading sample {i}: {e}")

        print("=" * 70 + "\n")

    def check_model_architecture(self, model):
        """
        检查模型架构和参数
        """
        print("\n" + "=" * 70)
        print("🔍 MODEL ARCHITECTURE VERIFICATION")
        print("=" * 70)

        print(f"\n📊 Encoder:")
        print(f"   Type: {type(model.encoder).__name__}")
        print(f"   Latent size: {model.encoder.latent_size}")

        if hasattr(model.encoder, 'use_multi_scale'):
            print(f"   Multi-scale: {model.encoder.use_multi_scale}")

        print(f"\n📊 Model:")
        print(f"   Model latent_size: {model.latent_size}")
        print(f"   Model d_latent: {model.d_latent}")
        print(f"   Model d_in: {model.d_in}")
        print(f"   Model d_out: {model.d_out}")

        # 检查维度一致性
        if isinstance(model.encoder.latent_size, list):
            expected = sum(model.encoder.latent_size)
            if model.latent_size != expected:
                print(f"   ❌ ERROR: Latent size mismatch!")
                print(f"      Expected: {expected}")
                print(f"      Got: {model.latent_size}")
            else:
                print(f"   ✅ CORRECT: Latent sizes match!")

        # 统计参数量
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        print(f"\n📊 Parameters:")
        print(f"   Total: {total_params:,}")
        print(f"   Trainable: {trainable_params:,}")
        print(f"   Frozen: {total_params - trainable_params:,}")

        print("=" * 70 + "\n")

    def check_model_output(self, model, data_loader, device):
        """
        检查模型输出的范围和分布
        """
        print("\n" + "=" * 70)
        print("🔍 MODEL OUTPUT VERIFICATION")
        print("=" * 70)

        model.eval()
        with torch.no_grad():
            try:
                # 获取一个 batch
                batch = next(iter(data_loader))
                if batch is None:
                    print("❌ ERROR: Got None batch!")
                    return

                # 移动到设备
                images = batch['images'].to(device)
                poses = batch['poses'].to(device)
                focal = batch['focal'].to(device)

                print(f"\n📊 Input batch:")
                print(f"   Images shape: {images.shape}")
                print(f"   Images range: [{images.min():.4f}, {images.max():.4f}]")
                print(f"   Poses shape: {poses.shape}")
                print(f"   Focal: {focal}")

                # Encode
                model.encode(images, poses, focal)

                print(f"\n📊 Encoder output:")
                if hasattr(model.encoder, 'latents') and model.encoder.latents:
                    for i, latent in enumerate(model.encoder.latents):
                        print(f"   Layer {i}: {latent.shape}, range [{latent.min():.4f}, {latent.max():.4f}]")
                else:
                    latent = model.encoder.latent
                    print(f"   Latent: {latent.shape}, range [{latent.min():.4f}, {latent.max():.4f}]")

                # 测试前向传播
                SB, NV = images.shape[:2]
                B = 1024  # 测试点数

                # 生成测试点
                xyz = torch.randn(SB, B, 3, device=device) * 0.5
                viewdirs = torch.randn(SB, B, 3, device=device)
                viewdirs = viewdirs / torch.norm(viewdirs, dim=-1, keepdim=True)

                # 前向传播
                output = model(xyz, coarse=True, viewdirs=viewdirs if model.use_viewdirs else None)

                print(f"\n📊 Model forward output:")
                print(f"   Shape: {output.shape}")
                print(f"   Range: [{output.min():.4f}, {output.max():.4f}]")

                rgb = output[..., :3]
                sigma = output[..., 3:4]

                print(f"\n   RGB:")
                print(f"      Range: [{rgb.min():.4f}, {rgb.max():.4f}]")
                print(f"      Mean: {rgb.mean():.4f}, Std: {rgb.std():.4f}")

                if rgb.min() < -0.01 or rgb.max() > 1.01:
                    print(f"      ⚠️ WARNING: RGB out of [0, 1] range!")
                else:
                    print(f"      ✅ CORRECT: RGB in [0, 1] range!")

                print(f"\n   Sigma:")
                print(f"      Range: [{sigma.min():.4f}, {sigma.max():.4f}]")
                print(f"      Mean: {sigma.mean():.4f}, Std: {sigma.std():.4f}")

                if sigma.min() < -0.01:
                    print(f"      ⚠️ WARNING: Sigma has negative values!")
                else:
                    print(f"      ✅ CORRECT: Sigma is non-negative!")

            except Exception as e:
                print(f"❌ ERROR during forward pass: {e}")
                import traceback
                traceback.print_exc()

        print("=" * 70 + "\n")
        model.train()

    def visualize_batch(self, batch, epoch=0, prefix="train"):
        """
        可视化一个 batch 的图像
        """
        try:
            images = batch['images']  # (B, NV, 3, H, W)

            B = min(4, images.shape[0])  # 最多显示4个对象
            NV = min(4, images.shape[1])  # 每个对象最多显示4个视角

            fig, axes = plt.subplots(B, NV, figsize=(NV * 3, B * 3))
            if B == 1:
                axes = axes.reshape(1, -1)
            if NV == 1:
                axes = axes.reshape(-1, 1)

            for i in range(B):
                for j in range(NV):
                    img = images[i, j].permute(1, 2, 0).cpu().numpy()
                    img = np.clip(img, 0, 1)

                    axes[i, j].imshow(img)
                    axes[i, j].set_title(f'Obj {i}, View {j}')
                    axes[i, j].axis('off')

            plt.tight_layout()
            save_path = self.save_dir / f"{prefix}_batch_epoch{epoch:03d}.png"
            plt.savefig(save_path, dpi=100, bbox_inches='tight')
            plt.close()

            print(f"✅ Saved batch visualization to {save_path}")

        except Exception as e:
            print(f"❌ Error visualizing batch: {e}")

    def visualize_predictions(self, pred, target, epoch=0, prefix="val"):
        """
        可视化预测结果和真实值
        """
        try:
            # pred, target: (B, 3, H, W) 或 (B, H, W, 3)
            if pred.dim() == 4 and pred.shape[1] == 3:
                pred = pred.permute(0, 2, 3, 1)  # (B, H, W, 3)
            if target.dim() == 4 and target.shape[1] == 3:
                target = target.permute(0, 2, 3, 1)

            B = min(4, pred.shape[0])

            fig, axes = plt.subplots(B, 3, figsize=(12, B * 4))
            if B == 1:
                axes = axes.reshape(1, -1)

            for i in range(B):
                pred_img = pred[i].detach().cpu().numpy()
                target_img = target[i].detach().cpu().numpy()

                pred_img = np.clip(pred_img, 0, 1)
                target_img = np.clip(target_img, 0, 1)

                # 计算误差图
                error = np.abs(pred_img - target_img)
                error_map = error.mean(axis=-1)  # (H, W)

                # 预测
                axes[i, 0].imshow(pred_img)
                axes[i, 0].set_title(f'Prediction {i}')
                axes[i, 0].axis('off')

                # 真实值
                axes[i, 1].imshow(target_img)
                axes[i, 1].set_title(f'Ground Truth {i}')
                axes[i, 1].axis('off')

                # 误差图
                im = axes[i, 2].imshow(error_map, cmap='hot', vmin=0, vmax=0.5)
                axes[i, 2].set_title(f'Error Map {i}')
                axes[i, 2].axis('off')
                plt.colorbar(im, ax=axes[i, 2], fraction=0.046)

            plt.tight_layout()
            save_path = self.save_dir / f"{prefix}_pred_epoch{epoch:03d}.png"
            plt.savefig(save_path, dpi=100, bbox_inches='tight')
            plt.close()

            print(f"✅ Saved prediction visualization to {save_path}")

        except Exception as e:
            print(f"❌ Error visualizing predictions: {e}")
            import traceback
            traceback.print_exc()

    def plot_training_curves(self, losses, psnrs, save_name="training_curves.png"):
        """
        绘制训练曲线
        """
        try:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

            # Loss curve
            ax1.plot(losses, linewidth=2)
            ax1.set_xlabel('Iteration')
            ax1.set_ylabel('Loss')
            ax1.set_title('Training Loss')
            ax1.grid(True, alpha=0.3)
            ax1.set_yscale('log')

            # PSNR curve
            ax2.plot(psnrs, linewidth=2, color='orange')
            ax2.set_xlabel('Iteration')
            ax2.set_ylabel('PSNR (dB)')
            ax2.set_title('Training PSNR')
            ax2.grid(True, alpha=0.3)
            ax2.axhline(y=25, color='r', linestyle='--', label='Target: 25 dB')
            ax2.axhline(y=30, color='g', linestyle='--', label='Good: 30 dB')
            ax2.legend()

            plt.tight_layout()
            save_path = self.save_dir / save_name
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()

            print(f"✅ Saved training curves to {save_path}")

        except Exception as e:
            print(f"❌ Error plotting curves: {e}")

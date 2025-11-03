import os.path
import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import tqdm
import warnings
import time

# ✅ 添加混合精度训练支持
from torch.cuda.amp import autocast, GradScaler


def custom_collate_fn(batch):
    """
    自定义 collate 函数，处理以下情况：
    1. 过滤掉 None 样本
    2. 处理不同数量图片的样本（统一裁剪所有相关字段）
    """
    # 过滤掉 None（跳过的样本）
    batch = [item for item in batch if item is not None]

    if len(batch) == 0:
        return None

    # ✅ 检查是否所有样本的图片数量一致
    if isinstance(batch[0], dict) and 'images' in batch[0]:
        num_images = [item['images'].shape[0] for item in batch]

        if len(set(num_images)) > 1:  # 如果图片数量不一致
            min_num = min(num_images)
            warnings.warn(
                f"Batch has inconsistent number of images: {num_images}. "
                f"Cropping all to {min_num} images."
            )

            # 裁剪所有样本到最小图片数
            for item in batch:
                if item['images'].shape[0] > min_num:
                    item['images'] = item['images'][:min_num]
                    item['poses'] = item['poses'][:min_num]

                    # 如果有其他相关字段也需要裁剪
                    if 'all_rays' in item:
                        item['all_rays'] = item['all_rays'][:min_num]
                    if 'all_rgb' in item:
                        item['all_rgb'] = item['all_rgb'][:min_num]

    # 使用默认的 collate
    return torch.utils.data.dataloader.default_collate(batch)


class Trainer:
    def __init__(
            self,
            net,
            train_dataset,
            val_dataset,
            args,
            conf,
            device=None,
            use_amp=False,  # ✅ 添加混合精度参数
    ):
        """
        初始化 Trainer

        Args:
            net: 神经网络模型
            train_dataset: 训练数据集
            val_dataset: 验证数据集
            args: 命令行参数
            conf: 配置字典
            device: 训练设备
            use_amp: 是否使用混合精度训练
        """
        self.args = args
        self.net = net
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.device = device

        # ✅ 混合精度训练设置
        self.use_amp = use_amp
        self.scaler = GradScaler() if use_amp else None

        # 训练参数
        self.batch_size = args.batch_size
        self.num_epochs = conf.get_int("num_epochs", 100000)
        self.lr = conf.get_float("lr", 1e-4)
        self.lr_policy = conf.get_string("lr_policy", "none")
        self.gamma = conf.get_float("gamma", 0.1)
        self.step_size = conf.get_int("step_size", 10000)

        # 日志和保存
        self.log_dir = os.path.join(args.logs_path, args.name)
        self.checkpoint_dir = os.path.join(args.checkpoints_path, args.name)
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        self.writer = SummaryWriter(self.log_dir)
        self.print_interval = args.print_interval if hasattr(args, 'print_interval') else 10
        self.save_interval_epochs = conf.get_int("save_interval", 1)  # 每 1 个 epoch 保存
        self.eval_interval_epochs = conf.get_int("eval_interval", 10)  # 每 10 个 epoch 评估
        self.vis_interval_epochs = conf.get_int("vis_interval", 10)  # 每 10 个 epoch 可视化

        # 优化器
        self.optimizer = torch.optim.Adam(
            [p for p in self.net.parameters() if p.requires_grad],
            lr=self.lr,
        )

        # 学习率调度器
        if self.lr_policy == "step":
            self.scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=self.step_size,
                gamma=self.gamma,
            )
        elif self.lr_policy == "multistep":
            milestones = conf.get_list("milestones", [10000, 20000])
            self.scheduler = torch.optim.lr_scheduler.MultiStepLR(
                self.optimizer,
                milestones=milestones,
                gamma=self.gamma,
            )
        else:
            self.scheduler = None

        # ✅ DataLoader 优化
        self.train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=True,  # ✅ 固定内存加速
            collate_fn=custom_collate_fn,
            drop_last=True,  # ✅ 丢弃最后一个不完整的 batch
        )

        self.val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=min(self.batch_size, 4),  # 验证时可以用更小的 batch
            shuffle=False,
            num_workers=2,  # ✅ 验证集也加速
            pin_memory=True,
            collate_fn=custom_collate_fn,
        )

        # 训练状态
        self.epoch = 0
        self.iter = 0
        self.global_step = 0
        self.best_val_loss = float('inf')

        # 加载检查点（如果存在）
        if hasattr(args, 'resume') and args.resume:
            self.load_checkpoint(args.resume)

    def post_batch(self, epoch, batch):
        """
        每个 batch 后的回调（子类可以重写）
        """
        pass

    def extra_save_state(self):
        """
        保存额外的状态（子类可以重写）
        """
        pass

    def train_step(self, data, global_step):
        """
        单个训练步骤（子类必须实现）

        Args:
            data: 批次数据
            global_step: 全局步数

        Returns:
            loss_dict: 损失字典
        """
        raise NotImplementedError("Subclass must implement train_step")

    def eval_step(self, data, global_step):
        """
        单个验证步骤（子类必须实现）

        Args:
            data: 批次数据
            global_step: 全局步数

        Returns:
            loss_dict: 损失字典
        """
        raise NotImplementedError("Subclass must implement eval_step")

    def vis_step(self, data, global_step, idx=None):
        """
        可视化步骤（子类可以重写）

        Args:
            data: 批次数据
            global_step: 全局步数
            idx: 批次索引

        Returns:
            vis: 可视化图像
            vals: 指标字典
        """
        return None, {}

    def train_epoch(self, epoch):
        """
        训练一个 epoch

        Args:
            epoch: 当前 epoch 编号
        """
        self.net.train()

        epoch_loss = 0.0
        num_batches = 0

        # 进度条
        pbar = tqdm.tqdm(
            enumerate(self.train_loader),
            total=len(self.train_loader),
            desc=f"Epoch {epoch}",
        )

        iter_start_time = time.time()

        for batch_idx, data in pbar:
            if data is None:  # 跳过空批次
                continue

            # 训练步骤（子类实现）
            loss_dict = self.train_step(data, self.global_step)

            # 累积损失
            if "t" in loss_dict:
                epoch_loss += loss_dict["t"]
            elif "loss" in loss_dict:
                epoch_loss += loss_dict["loss"].item() if torch.is_tensor(loss_dict["loss"]) else loss_dict["loss"]
            num_batches += 1

            # 更新进度条
            postfix_dict = {}
            for key, val in loss_dict.items():
                if key != "loss":
                    if torch.is_tensor(val):
                        postfix_dict[key] = f"{val.item():.4f}"
                    else:
                        postfix_dict[key] = f"{val:.4f}"
            postfix_dict["lr"] = f"{self.optimizer.param_groups[0]['lr']:.6f}"
            pbar.set_postfix(postfix_dict)

            # 打印和记录
            if batch_idx % self.print_interval == 0:
                iter_time = time.time() - iter_start_time

                log_str = f"[{iter_time:.2f}s/it] E {epoch} B {batch_idx}"
                for key, val in loss_dict.items():
                    if key != "loss":
                        if torch.is_tensor(val):
                            log_str += f" {key}:{val.item():.4f}"
                        else:
                            log_str += f" {key}:{val:.4f}"
                log_str += f" lr {self.optimizer.param_groups[0]['lr']:.6f}"
                print(log_str)

                # 记录到 tensorboard
                for key, val in loss_dict.items():
                    if key != "loss":
                        if torch.is_tensor(val):
                            self.writer.add_scalar(f"train/{key}", val.item(), self.global_step)
                        else:
                            self.writer.add_scalar(f"train/{key}", val, self.global_step)
                self.writer.add_scalar("train/lr", self.optimizer.param_groups[0]['lr'], self.global_step)

                iter_start_time = time.time()

            # ✅ 保存检查点逻辑已移到 epoch 结束后
            # if self.global_step % self.save_interval == 0 and self.global_step > 0:
            #     self.save_checkpoint(f"iter_{self.global_step}.pth")

            # ✅ 验证逻辑已移到 epoch 结束后
            # if self.global_step % self.eval_interval == 0 and self.global_step > 0:
            #     self.validate()

            # ✅ 可视化逻辑已移到 epoch 结束后
            # if self.global_step % self.vis_interval == 0 and self.global_step > 0:
            #     vis, vals = self.vis_step(data, self.global_step)
            #     if vis is not None:
            #         self.writer.add_image("vis", vis, self.global_step, dataformats='HWC')
            #     for key, val in vals.items():
            #         self.writer.add_scalar(f"vis/{key}", val, self.global_step)

            # 每个 batch 后的回调
            self.post_batch(epoch, batch_idx)

            self.global_step += 1

        # Epoch 统计
        avg_loss = epoch_loss / num_batches if num_batches > 0 else 0

        print(f"Epoch {epoch} finished: avg_loss={avg_loss:.4f}")

        # ✅ 按 epoch 保存检查点
        if (epoch + 1) % self.save_interval_epochs == 0:
            self.save_checkpoint(f"epoch_{epoch}.pth")
            print(f"💾 Epoch {epoch} checkpoint saved")

        # ✅ 按 epoch 验证
        if (epoch + 1) % self.eval_interval_epochs == 0:
            print(f"\n{'=' * 80}")
            print(f"📊 Evaluation at epoch {epoch}")
            print(f"{'=' * 80}")
            self.validate()

        # ✅ 按 epoch 可视化
        if (epoch + 1) % self.vis_interval_epochs == 0 and len(self.val_loader) > 0:
            # 从验证集中取一个样本进行可视化
            try:
                val_data = next(iter(self.val_loader))
                if val_data is not None:
                    vis, vals = self.vis_step(val_data, self.global_step)
                    if vis is not None:
                        self.writer.add_image("vis", vis, self.global_step, dataformats='HWC')
                    for key, val in vals.items():
                        self.writer.add_scalar(f"vis/{key}", val, self.global_step)
            except Exception as e:
                print(f"⚠️ Visualization failed: {e}")

        # 更新学习率
        if self.scheduler is not None:
            self.scheduler.step()

        return avg_loss

    def validate(self):
        """
        验证模型

        Returns:
            avg_val_loss: 平均验证损失
        """
        self.net.eval()

        val_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for data in tqdm.tqdm(self.val_loader, desc="Validation"):
                if data is None:
                    continue

                # 验证步骤（子类实现）
                loss_dict = self.eval_step(data, self.global_step)

                if "t" in loss_dict:
                    val_loss += loss_dict["t"]
                elif "loss" in loss_dict:
                    val_loss += loss_dict["loss"].item() if torch.is_tensor(loss_dict["loss"]) else loss_dict["loss"]
                num_batches += 1

        avg_val_loss = val_loss / num_batches if num_batches > 0 else 0

        print(f"Validation: avg_loss={avg_val_loss:.4f}")

        # 记录到 tensorboard
        self.writer.add_scalar("val/loss", avg_val_loss, self.global_step)

        # 保存最佳模型
        if avg_val_loss < self.best_val_loss:
            self.best_val_loss = avg_val_loss
            self.save_checkpoint("best.pth")
            print(f"✅ New best model saved! Loss: {avg_val_loss:.4f}")

        self.net.train()
        return avg_val_loss

    def save_checkpoint(self, filename):
        """
        保存检查点

        Args:
            filename: 文件名
        """
        checkpoint = {
            "epoch": self.epoch,
            "iter": self.iter,
            "global_step": self.global_step,
            "net_state_dict": self.net.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_val_loss": self.best_val_loss,
        }

        if self.scheduler is not None:
            checkpoint["scheduler_state_dict"] = self.scheduler.state_dict()

        # ✅ 保存 scaler 状态（用于恢复混合精度训练）
        if self.use_amp and self.scaler is not None:
            checkpoint["scaler_state_dict"] = self.scaler.state_dict()

        filepath = os.path.join(self.checkpoint_dir, filename)
        torch.save(checkpoint, filepath)
        print(f"💾 Checkpoint saved: {filepath}")

        # 调用子类的额外保存逻辑
        self.extra_save_state()

    def load_checkpoint(self, filepath):
        """
        加载检查点

        Args:
            filepath: 文件路径
        """
        if not os.path.exists(filepath):
            print(f"❌ Checkpoint not found: {filepath}")
            return

        checkpoint = torch.load(filepath, map_location=self.device)

        self.epoch = checkpoint.get("epoch", 0)
        self.iter = checkpoint.get("iter", 0)
        self.global_step = checkpoint.get("global_step", 0)
        self.best_val_loss = checkpoint.get("best_val_loss", float('inf'))

        self.net.load_state_dict(checkpoint["net_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        if self.scheduler is not None and "scheduler_state_dict" in checkpoint:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        # ✅ 恢复 scaler 状态
        if self.use_amp and self.scaler is not None and "scaler_state_dict" in checkpoint:
            self.scaler.load_state_dict(checkpoint["scaler_state_dict"])

        print(f"✅ Checkpoint loaded: {filepath}")
        print(f"📍 Resuming from epoch {self.epoch}, global_step {self.global_step}")

    def start(self):
        """
        开始训练
        """
        print("=" * 80)
        print(f"🚀 Training started: {self.args.name}")
        print(f"🖥️  Device: {self.device}")
        print(f"📦 Batch size: {self.batch_size}")
        print(f"📈 Learning rate: {self.lr}")
        print(f"🔄 Num epochs: {self.num_epochs}")
        print(f"⚡ Mixed Precision: {self.use_amp}")
        print(f"💾 Checkpoints: {self.checkpoint_dir}")
        print(f"📊 Logs: {self.log_dir}")
        print("=" * 80)

        for epoch in range(self.epoch, self.num_epochs):
            self.epoch = epoch

            # 训练一个 epoch
            avg_loss = self.train_epoch(epoch)


        print("=" * 80)
        print("🎉 Training finished!")
        print("=" * 80)

        self.writer.close()

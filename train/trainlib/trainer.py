import os.path
import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import tqdm
import warnings
import time
import glob  # ✅ 添加 glob 用于清理旧 checkpoint

# ✅ 添加混合精度训练支持
from torch.cuda.amp import autocast, GradScaler


def custom_collate_fn(batch):
    """
    自定义 collate 函数，处理以下情况：
    1. 过滤掉 None 样本
    2. 处理不同数量图片的样本（统一裁剪所有相关字段）
    3. 安全处理标量 tensor 和空维度

    ✅ 修复：创建新的 tensor 副本，避免 "storage not resizable" 错误
    ✅ 修复：安全检查 tensor 维度，避免 IndexError
    """
    import warnings
    import torch

    # 过滤掉 None（跳过的样本）
    batch = [item for item in batch if item is not None]

    if len(batch) == 0:
        return None

    # ✅ 检查批次是否有效
    if not isinstance(batch[0], dict):
        warnings.warn(f"⚠️ Warning: Batch items are not dictionaries, skipping batch")
        return None

    if 'images' not in batch[0]:
        warnings.warn(f"⚠️ Warning: 'images' key not found in batch, skipping batch")
        return None

    # ✅ 安全获取图片数量
    try:
        num_images = []
        for item in batch:
            if 'images' not in item:
                warnings.warn(f"⚠️ Warning: 'images' missing in batch item, skipping batch")
                return None

            images = item['images']
            if not isinstance(images, torch.Tensor):
                warnings.warn(f"⚠️ Warning: 'images' is not a tensor, skipping batch")
                return None

            if images.ndim == 0:
                warnings.warn(f"⚠️ Warning: 'images' has no dimensions, skipping batch")
                return None

            num_images.append(images.shape[0])

        # ✅ 检查图片数量是否一致
        if len(set(num_images)) > 1:  # 如果图片数量不一致
            min_num = min(num_images)
            warnings.warn(
                f"Batch has inconsistent number of images: {num_images}. "
                f"Cropping all to {min_num} images."
            )

            # ✅ 创建新的 batch，包含裁剪后的 tensor 副本
            cropped_batch = []
            for item in batch:
                cropped_item = {}
                images_shape_0 = item['images'].shape[0]

                for key, value in item.items():
                    if isinstance(value, torch.Tensor):
                        # ✅ 安全检查：确保 tensor 有维度
                        if value.ndim == 0:
                            # 标量 tensor（如 focal length），直接复制
                            cropped_item[key] = value.clone()
                        elif value.shape[0] == images_shape_0:
                            # 第一维与 images 数量一致，需要裁剪
                            cropped_item[key] = value[:min_num].contiguous().clone()
                        else:
                            # 第一维与 images 数量不一致，不裁剪
                            cropped_item[key] = value.contiguous().clone()
                    else:
                        # 非 tensor 数据（如字符串、列表等），直接复制
                        cropped_item[key] = value

                cropped_batch.append(cropped_item)

            batch = cropped_batch

    except Exception as e:
        warnings.warn(f"⚠️ Warning: Error during batch processing: {e}. Skipping batch.")
        import traceback
        traceback.print_exc()  # ✅ 打印完整错误堆栈，方便调试
        return None

    # 使用默认的 collate
    try:
        return torch.utils.data.dataloader.default_collate(batch)
    except RuntimeError as e:
        warnings.warn(f"❌ Collate failed even after cropping: {e}. Skipping batch.")
        import traceback
        traceback.print_exc()
        return None


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
        self.num_epochs = conf.get_int("num_epochs", 100)
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

        # ✅ Checkpoint 管理参数
        self.keep_last_checkpoints = conf.get_int("keep_last_checkpoints", 20)  # 保留最近 20 个
        self.save_strategy = conf.get_string("save_strategy", "keep_last")  # keep_last, keep_all, milestone

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
            pin_memory=False,  # ✅ 固定内存加速
            collate_fn=custom_collate_fn,
            drop_last=True,  # ✅ 丢弃最后一个不完整的 batch
        )

        self.val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=min(self.batch_size, 4),  # 验证时可以用更小的 batch
            shuffle=False,
            num_workers=0,  # ✅ 验证集也加速
            pin_memory=False,
            collate_fn=custom_collate_fn,
        )

        # 训练状态
        self.epoch = 0
        self.iter = 0
        self.global_step = 0
        self.best_val_loss = float('inf')

        if hasattr(args, 'resume') and args.resume:
            # 如果 resume 是布尔值 True，自动查找 latest.pth
            if isinstance(args.resume, bool):
                checkpoint_path = os.path.join(self.checkpoint_dir, "latest.pth")
                if os.path.exists(checkpoint_path):
                    print(f"✅ Auto-resuming from: {checkpoint_path}")
                    self.load_checkpoint(checkpoint_path)
                else:
                    print(f"⚠️ No checkpoint found at {checkpoint_path}, starting from scratch")
            # 如果 resume 是字符串路径，直接加载
            elif isinstance(args.resume, str):
                if os.path.exists(args.resume):
                    print(f"✅ Resuming from: {args.resume}")
                    self.load_checkpoint(args.resume)
                else:
                    raise FileNotFoundError(f"❌ Checkpoint not found: {args.resume}")

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
            if data is None:  # ✅ 跳过空批次
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

            # 每个 batch 后的回调
            self.post_batch(epoch, batch_idx)

            self.global_step += 1

        # Epoch 统计
        avg_loss = epoch_loss / num_batches if num_batches > 0 else 0

        print(f"Epoch {epoch} finished: avg_loss={avg_loss:.4f}")

        # ✅ 按 epoch 保存检查点（修改后的逻辑）
        if (epoch + 1) % self.save_interval_epochs == 0:
            self.save_checkpoint_with_epoch(epoch, avg_loss)

        # ✅ 按 epoch 验证
        if (epoch + 1) % self.eval_interval_epochs == 0:
            print(f"\n{'=' * 80}")
            print(f"📊 Evaluation at epoch {epoch}")
            print(f"{'=' * 80}")
            val_loss = self.validate()

            # ✅ 如果是最佳模型，保存 best.pth
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.save_checkpoint("best.pth", is_best=True)

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
        num_skipped = 0  # ✅ 统计跳过的批次

        with torch.no_grad():
            for data in tqdm.tqdm(self.val_loader, desc="Validation"):
                # ✅ 跳过 None 批次（collate 失败）
                if data is None:
                    num_skipped += 1
                    continue

                # ✅ 跳过空批次
                if not data or 'images' not in data:
                    num_skipped += 1
                    continue

                try:
                    # 验证步骤（子类实现）
                    loss_dict = self.eval_step(data, self.global_step)

                    if "t" in loss_dict:
                        val_loss += loss_dict["t"]
                    elif "loss" in loss_dict:
                        val_loss += loss_dict["loss"].item() if torch.is_tensor(loss_dict["loss"]) else loss_dict[
                            "loss"]
                    num_batches += 1

                except RuntimeError as e:
                    print(f"\n⚠️ Skipping validation batch due to error: {e}")
                    num_skipped += 1
                    continue

        # ✅ 计算平均损失
        if num_batches > 0:
            avg_val_loss = val_loss / num_batches
            if num_skipped > 0:
                print(f"\n⚠️ Validation: Skipped {num_skipped} problematic batches")
        else:
            print("\n⚠️ No valid validation batches!")
            avg_val_loss = float('inf')

        print(f"Validation: avg_loss={avg_val_loss:.4f}")

        # 记录到 tensorboard
        self.writer.add_scalar("val/loss", avg_val_loss, self.global_step)

        self.net.train()
        return avg_val_loss

    # ✅ ============================================================
    # ✅ 新增：带 epoch 编号的 checkpoint 保存（不覆盖）
    # ✅ ============================================================
    def save_checkpoint_with_epoch(self, epoch, train_loss=None):
        """
        保存带 epoch 编号的 checkpoint（不覆盖之前的）

        Args:
            epoch: 当前 epoch 编号
            train_loss: 训练损失（可选）
        """
        # 1. 保存带 epoch 编号的 checkpoint（保存当前 epoch）
        epoch_filename = f"epoch_{epoch:04d}.pth"
        self.save_checkpoint(epoch_filename, is_best=False, save_epoch=epoch)

        # 2. ✅ 保存为 latest.pth 时，保存下一个 epoch（用于恢复）
        self.save_checkpoint("latest.pth", is_best=False, save_epoch=epoch + 1)

        # 3. 根据保存策略清理旧 checkpoint
        if self.save_strategy == "keep_last":
            self.cleanup_old_checkpoints(keep_last=self.keep_last_checkpoints)
        elif self.save_strategy == "milestone":
            self.cleanup_milestone_checkpoints(epoch)
        # keep_all 策略不清理

        # 4. 显示磁盘使用情况
        self.print_checkpoint_disk_usage()

    # ✅ ============================================================
    # ✅ 新增：清理旧 checkpoint（只保留最近 N 个）
    # ✅ ============================================================
    def cleanup_old_checkpoints(self, keep_last=20):
        """
        清理旧的 checkpoint，只保留最近的 N 个

        Args:
            keep_last: 保留最近的 N 个 checkpoint
        """
        # 获取所有带 epoch 编号的 checkpoint
        pattern = os.path.join(self.checkpoint_dir, "epoch_*.pth")
        checkpoints = sorted(glob.glob(pattern))

        # 删除旧的 checkpoint
        if len(checkpoints) > keep_last:
            num_to_delete = len(checkpoints) - keep_last
            for old_checkpoint in checkpoints[:num_to_delete]:
                try:
                    os.remove(old_checkpoint)
                    print(f"🗑️  Removed old checkpoint: {os.path.basename(old_checkpoint)}")
                except Exception as e:
                    print(f"⚠️  Failed to remove {os.path.basename(old_checkpoint)}: {e}")

    # ✅ ============================================================
    # ✅ 新增：里程碑式保存策略
    # ✅ ============================================================
    def cleanup_milestone_checkpoints(self, current_epoch):
        """
        里程碑式保存策略：
        - 前 10 个 epoch：全部保留
        - 10-100 epoch：每 5 个保留一个
        - 100+ epoch：每 20 个保留一个

        Args:
            current_epoch: 当前 epoch
        """
        pattern = os.path.join(self.checkpoint_dir, "epoch_*.pth")
        checkpoints = sorted(glob.glob(pattern))

        for checkpoint_path in checkpoints:
            # 提取 epoch 编号
            basename = os.path.basename(checkpoint_path)
            try:
                epoch_num = int(basename.split('_')[1].split('.')[0])
            except:
                continue

            # 判断是否应该保留
            should_keep = (
                    epoch_num <= 10 or  # 前 10 个全部保留
                    (epoch_num <= 100 and epoch_num % 5 == 0) or  # 10-100 每 5 个保留
                    (epoch_num > 100 and epoch_num % 20 == 0) or  # 100+ 每 20 个保留
                    epoch_num == current_epoch  # 当前 epoch 保留
            )

            if not should_keep:
                try:
                    os.remove(checkpoint_path)
                    print(f"🗑️  Removed checkpoint: {basename}")
                except Exception as e:
                    print(f"⚠️  Failed to remove {basename}: {e}")

    # ✅ ============================================================
    # ✅ 新增：显示 checkpoint 磁盘使用情况
    # ✅ ============================================================
    def print_checkpoint_disk_usage(self):
        """
        打印 checkpoint 目录的磁盘使用情况
        """
        try:
            pattern = os.path.join(self.checkpoint_dir, "*.pth")
            checkpoints = glob.glob(pattern)

            total_size = 0
            for checkpoint in checkpoints:
                total_size += os.path.getsize(checkpoint)

            total_size_mb = total_size / (1024 * 1024)
            total_size_gb = total_size / (1024 * 1024 * 1024)

            if total_size_gb > 1:
                print(f"💾 Checkpoint disk usage: {total_size_gb:.2f} GB ({len(checkpoints)} files)")
            else:
                print(f"💾 Checkpoint disk usage: {total_size_mb:.2f} MB ({len(checkpoints)} files)")
        except Exception as e:
            print(f"⚠️  Failed to calculate disk usage: {e}")

    # ✅ ============================================================
    # ✅ 修改：原有的 save_checkpoint 函数（保持兼容性）
    # ✅ ============================================================
    def save_checkpoint(self, filename, is_best=False, save_epoch=None):
        """
        保存检查点

        Args:
            filename: 文件名
            is_best: 是否是最佳模型
            save_epoch: 保存的 epoch 编号（如果为 None，使用 self.epoch）
        """
        # ✅ 如果指定了 save_epoch，使用它；否则使用 self.epoch
        epoch_to_save = save_epoch if save_epoch is not None else self.epoch

        checkpoint = {
            "epoch": epoch_to_save,  # ✅ 使用指定的 epoch
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

        if is_best:
            print(f"🌟 Best checkpoint saved: {filename} (loss: {self.best_val_loss:.4f})")
        elif filename != "latest.pth":
            print(f"💾 Checkpoint saved: {filename}")

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

        # ✅ 验证 epoch 和 global_step 的一致性
        if len(self.train_loader) > 0:
            batches_per_epoch = len(self.train_loader)
            expected_epoch = self.global_step // batches_per_epoch

            if expected_epoch != self.epoch:
                print(f"\n⚠️  Checkpoint inconsistency detected!")
                print(f"   Saved epoch: {self.epoch}")
                print(f"   Global step: {self.global_step}")
                print(f"   Expected epoch (from global_step): {expected_epoch}")
                print(f"   Batches per epoch: {batches_per_epoch}")

                # ✅ 自动修正（使用 global_step 计算的 epoch）
                if "latest.pth" in filepath:
                    print(f"   🔧 Auto-correcting to epoch {expected_epoch}")
                    self.epoch = expected_epoch
                else:
                    print(f"   ⚠️  Using saved epoch {self.epoch} (not latest.pth)")

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
        print(f"🗂️  Save strategy: {self.save_strategy}")
        if self.save_strategy == "keep_last":
            print(f"📁 Keep last {self.keep_last_checkpoints} checkpoints")
        print("=" * 80)

        # ✅ 新增：如果恢复的 epoch 是评估节点，先评估一次
        if self.epoch > 0 and self.epoch % self.eval_interval_epochs == 0:
            print(f"\n{'=' * 80}")
            print(f"📊 Running evaluation for resumed epoch {self.epoch}")
            print(f"{'=' * 80}")

            val_loss = self.validate()

            # 检查是否是最佳模型
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.save_checkpoint("best.pth", is_best=True)

            print(f"{'=' * 80}\n")

        # 继续正常训练
        for epoch in range(self.epoch, self.num_epochs):
            self.epoch = epoch

            # 训练一个 epoch
            avg_loss = self.train_epoch(epoch)

        print("=" * 80)
        print("🎉 Training finished!")
        print("=" * 80)

        self.writer.close()

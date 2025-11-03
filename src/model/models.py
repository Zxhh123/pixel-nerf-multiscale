"""
Main model implementation with Multi-View Attention and Smart Feature Fusion
Enhanced version with multi-scale features
"""
import torch
import torch.nn as nn
from .encoder import ImageEncoder
from .code import PositionalEncoding
from .model_util import make_encoder, make_mlp
import torch.autograd.profiler as profiler
from util import repeat_interleave
import os
import os.path as osp
import warnings


class PixelNeRFNet(torch.nn.Module):
    def __init__(self, conf, stop_encoder_grad=False):
        """
        :param conf PyHocon config subtree 'model'
        :param stop_encoder_grad: 是否停止 encoder 的梯度传播
        """
        super().__init__()

        # ========== 编码器初始化 ==========
        self.encoder = make_encoder(conf["encoder"])
        self.use_encoder = conf.get_bool("use_encoder", True)
        self.use_xyz = conf.get_bool("use_xyz", False)
        assert self.use_encoder or self.use_xyz

        # ========== 基础配置 ==========
        self.normalize_z = conf.get_bool("normalize_z", True)
        self.stop_encoder_grad = stop_encoder_grad
        self.use_code = conf.get_bool("use_code", False)
        self.use_code_viewdirs = conf.get_bool("use_code_viewdirs", True)
        self.use_viewdirs = conf.get_bool("use_viewdirs", False)
        self.use_global_encoder = conf.get_bool("use_global_encoder", False)

        # ========== 智能特征融合配置 ==========
        self.use_smart_fusion = conf.get_bool("use_smart_fusion", False)
        self.use_adaptive_sampling = conf.get_bool("use_adaptive_sampling", False)
        self.fusion_heads = conf.get_int("fusion_heads", 8)
        self.fusion_dropout = conf.get_float("fusion_dropout", 0.1)
        self.fusion_type = conf.get_string("fusion_type", "attention")
        self.use_cbam = conf.get_bool("use_cbam", True)
        self.quality_threshold = conf.get_float("quality_threshold", 0.3)

        # ========== 修复：统一处理 encoder.latent_size ==========
        encoder_latent_size = self.encoder.latent_size
        if isinstance(encoder_latent_size, (list, tuple)):
            # ✅ 多尺度特征：求和得到总维度
            self.latent_size = sum(int(x) for x in encoder_latent_size)
            self.is_multi_scale = True
            self.layer_dims = [int(x) for x in encoder_latent_size]
            print(f"✅ Multi-scale encoder detected:")
            print(f"   - Layer sizes: {self.layer_dims}")
            print(f"   - Total latent size: {self.latent_size}")
        else:
            # ✅ 单尺度特征：直接使用
            self.latent_size = int(encoder_latent_size)
            self.is_multi_scale = False
            self.layer_dims = [self.latent_size]
            print(f"✅ Single-scale encoder:")
            print(f"   - Latent size: {self.latent_size}")

        # ========== 初始化智能特征融合模块 ==========
        if self.use_smart_fusion:
            try:
                from .feature_fusion import SmartFeatureFusion

                # 如果是多尺度，使用融合模块
                if self.is_multi_scale:
                    self.feature_fusion = SmartFeatureFusion(
                        layer_dims=self.layer_dims,
                        output_dim=512,  # 融合后的输出维度
                        use_attention=(self.fusion_type == "attention"),
                        dropout=self.fusion_dropout,
                        num_heads=self.fusion_heads,
                        use_cbam=self.use_cbam
                    )
                    # 更新 latent_size 为融合后的维度
                    self.latent_size = 512
                    print(f"✅ Smart Feature Fusion enabled:")
                    print(f"   - Fusion type: {self.fusion_type}")
                    print(f"   - Fusion heads: {self.fusion_heads}")
                    print(f"   - CBAM: {'✅' if self.use_cbam else '❌'}")
                    print(f"   - Output dimension: {self.latent_size}")
                else:
                    print(f"⚠️  Smart fusion requested but encoder is single-scale")
                    self.use_smart_fusion = False

            except ImportError as e:
                print(f"❌ Failed to import SmartFeatureFusion: {e}")
                print(f"⚠️  Falling back to basic multi-scale concatenation")
                self.use_smart_fusion = False

        # ========== 位置编码 ==========
        d_latent = 0
        d_in = 3  # xyz 坐标

        if self.use_code:
            num_freqs = conf.get_int("code.num_freqs", 6)
            freq_factor = conf.get_float("code.freq_factor", 1.5)
            include_input = conf.get_bool("code.include_input", True)
            self.code = PositionalEncoding.from_conf(
                num_freqs, freq_factor=freq_factor, include_input=include_input
            )
            d_in = self.code.d_out
            print(f"✅ Positional encoding for xyz: {d_in} dims")

        # 视角方向编码
        if self.use_viewdirs:
            if self.use_code_viewdirs:
                num_freqs_viewdirs = conf.get_int("code_viewdirs.num_freqs", 4)
                freq_factor_viewdirs = conf.get_float("code_viewdirs.freq_factor", 1.5)
                include_input_viewdirs = conf.get_bool("code_viewdirs.include_input", True)
                self.code_viewdirs = PositionalEncoding.from_conf(
                    num_freqs_viewdirs,
                    freq_factor=freq_factor_viewdirs,
                    include_input=include_input_viewdirs
                )
                d_latent = self.code_viewdirs.d_out
                print(f"✅ Positional encoding for viewdirs: {d_latent} dims")
            else:
                d_latent = 3
                print(f"✅ Raw viewdirs: {d_latent} dims")

        # ========== MLP 输入维度计算 ==========
        if self.use_encoder:
            d_in = self.latent_size + d_in  # 特征 + xyz编码

        print(f"\n📊 MLP Input Configuration:")
        print(f"   - Feature dimension: {self.latent_size}")
        print(f"   - XYZ dimension: {d_in - self.latent_size}")
        print(f"   - Viewdir dimension: {d_latent}")
        print(f"   - Total input dimension: {d_in}")

        # ========== MLP 解码器 ==========
        self.mlp_coarse = make_mlp(conf["mlp_coarse"], d_in, d_latent=d_latent)
        self.mlp_fine = make_mlp(
            conf["mlp_fine"], d_in, d_latent=d_latent, allow_empty=True
        )

        # 如果没有 fine 网络，使用 coarse 网络
        if self.mlp_fine is None:
            self.mlp_fine = self.mlp_coarse
            print("⚠️  No separate fine MLP, using coarse MLP for both")

        # 输出维度
        self.d_in = d_in
        self.d_out = 4  # RGB + density
        self.d_latent = d_latent

        # ========== 全局特征编码器（可选） ==========
        if self.use_global_encoder:
            self.global_encoder = nn.Sequential(
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
                nn.Linear(self.latent_size, 256),
                nn.ReLU(inplace=True),
                nn.Linear(256, 256)
            )
            print("✅ Global encoder enabled")

        print(f"\n✅ PixelNeRFNet initialized:")
        print(f"   - Input dimension: {d_in}")
        print(f"   - Latent dimension: {d_latent}")
        print(f"   - Output dimension: {self.d_out}")
        print(f"   - Use encoder: {self.use_encoder}")
        print(f"   - Use xyz: {self.use_xyz}")
        print(f"   - Use viewdirs: {self.use_viewdirs}")
        print(f"   - Smart fusion: {'✅' if self.use_smart_fusion else '❌'}")
        print(f"   - Adaptive sampling: {'✅' if self.use_adaptive_sampling else '❌'}")

    def encode(self, images, poses, focal, z_bounds=None, c=None):
        """
        编码输入图像
        :param images (NS, 3, H, W) 输入图像
        :param poses (NS, 4, 4) 相机位姿
        :param focal (NS,) 或 (NS, 2) 焦距
        :param z_bounds (NS, 2) 深度边界
        :param c (NS,) 可选的类别编码
        :return latent 编码后的特征
        """
        if images.shape[0] == 0:
            return None

        # ✅ 停止梯度（如果需要）
        if self.stop_encoder_grad:
            images = images.detach()

        # ✅ 编码图像
        with profiler.record_function("encoder_forward"):
            latent = self.encoder(images)

        # ✅ 应用智能特征融合（如果启用）
        if self.use_smart_fusion and self.is_multi_scale:
            with profiler.record_function("feature_fusion"):
                # latent 是多尺度特征列表
                if isinstance(latent, list):
                    latent = self.feature_fusion(latent)  # 返回 (NS, output_dim, H, W)

        # ✅ 保存编码后的特征和相机参数
        self.latent = latent
        self.poses = poses
        self.focal = focal
        self.c = c
        self.z_bounds = z_bounds

        # ✅ 计算图像尺寸
        if isinstance(latent, torch.Tensor):
            self.latent_scaling = images.shape[-1] / latent.shape[-1]
        else:
            # 多尺度特征，使用第一层的尺寸
            self.latent_scaling = images.shape[-1] / latent[0].shape[-1]

        # ✅ 全局特征（可选）
        if self.use_global_encoder:
            self.global_latent = self.global_encoder(latent)
        else:
            self.global_latent = None

        return latent

    def forward(self, xyz, coarse=True, viewdirs=None, far=False):
        """
        前向传播
        :param xyz (SB, B, 3) 3D 坐标（世界坐标系）
        :param coarse bool 是否使用粗网络
        :param viewdirs (SB, B, 3) 视角方向
        :param far bool 是否是远距离点
        :return (SB, B, 4) RGB + density
        """
        with profiler.record_function("model_forward"):
            SB, B, _ = xyz.shape

            # ✅ 从编码器获取像素对齐的特征
            with profiler.record_function("encoder_index"):
                # 将世界坐标转换到相机坐标
                xyz_rot = torch.matmul(self.poses[:, None, :3, :3], xyz.unsqueeze(-1))[..., 0]
                xyz_cam = xyz_rot + self.poses[:, None, :3, 3]

                # 投影到图像平面
                if self.focal.shape[-1] == 2:
                    fx, fy = self.focal[..., 0], self.focal[..., 1]
                else:
                    fx = fy = self.focal

                uv = torch.stack([
                    xyz_cam[..., 0] / xyz_cam[..., 2] * fx[:, None],
                    xyz_cam[..., 1] / xyz_cam[..., 2] * fy[:, None]
                ], dim=-1)

                # 归一化到 [-1, 1]
                if isinstance(self.latent, torch.Tensor):
                    H, W = self.latent.shape[-2:]
                else:
                    H, W = self.latent[0].shape[-2:]

                uv = uv / torch.tensor([W / 2, H / 2], device=uv.device) - 1.0

                # 采样特征
                if isinstance(self.latent, torch.Tensor):
                    # 单尺度特征
                    latent_feat = torch.nn.functional.grid_sample(
                        self.latent,
                        uv.view(SB, 1, B, 2),
                        align_corners=True,
                        mode='bilinear',
                        padding_mode='border'
                    )  # (SB, C, 1, B)
                    latent_feat = latent_feat.squeeze(2).transpose(1, 2)  # (SB, B, C)
                else:
                    # 多尺度特征（已融合）
                    latent_feat = torch.nn.functional.grid_sample(
                        self.latent,
                        uv.view(SB, 1, B, 2),
                        align_corners=True,
                        mode='bilinear',
                        padding_mode='border'
                    )
                    latent_feat = latent_feat.squeeze(2).transpose(1, 2)

            # ✅ 构建 MLP 输入
            mlp_input = latent_feat

            # 添加 xyz 编码
            if self.use_xyz:
                if self.use_code:
                    xyz_encoded = self.code(xyz)
                else:
                    xyz_encoded = xyz
                mlp_input = torch.cat([mlp_input, xyz_encoded], dim=-1)

            # 添加视角方向编码
            if self.use_viewdirs and viewdirs is not None:
                if self.use_code_viewdirs:
                    viewdirs_encoded = self.code_viewdirs(viewdirs)
                else:
                    viewdirs_encoded = viewdirs
                # viewdirs 作为 latent 输入
                latent_input = viewdirs_encoded
            else:
                latent_input = None

            # ✅ MLP 解码
            mlp = self.mlp_coarse if coarse else self.mlp_fine

            with profiler.record_function("mlp_forward"):
                if latent_input is not None:
                    mlp_output = mlp(mlp_input, combine_inner_dims=(1,), combine_index=mlp.d_latent, dim_size=B, latent=latent_input)
                else:
                    mlp_output = mlp(mlp_input, combine_inner_dims=(1,), combine_index=mlp.d_latent, dim_size=B)

            # ✅ 输出：RGB + density
            return mlp_output

    def load_weights(self, args, opt_init=False, strict=True, device=None):
        """
        加载预训练权重
        """
        if device is None:
            device = torch.device("cpu")

        # 加载权重文件
        if hasattr(args, 'resume') and args.resume and os.path.isfile(args.resume):
            print(f"✅ Loading checkpoint from {args.resume}")
            checkpoint = torch.load(args.resume, map_location=device)

            # 加载模型权重
            if "model_state_dict" in checkpoint:
                self.load_state_dict(checkpoint["model_state_dict"], strict=strict)
            elif "model" in checkpoint:
                self.load_state_dict(checkpoint["model"], strict=strict)
            else:
                self.load_state_dict(checkpoint, strict=strict)

            print("✅ Checkpoint loaded successfully")

            return checkpoint
        else:
            if hasattr(args, 'resume') and args.resume:
                warnings.warn(f"❌ Checkpoint file not found: {args.resume}")
            return None

    def save_weights(self, path, optimizer=None, epoch=None):
        """
        保存模型权重
        """
        checkpoint = {
            "model_state_dict": self.state_dict(),
            "epoch": epoch,
        }
        if optimizer is not None:
            checkpoint["optimizer_state_dict"] = optimizer.state_dict()

        torch.save(checkpoint, path)
        print(f"✅ Checkpoint saved to {path}")


def make_model(conf):
    """
    创建 PixelNeRF 模型
    """
    return PixelNeRFNet(conf)

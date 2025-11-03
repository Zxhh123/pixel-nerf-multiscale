"""
Spatial encoder (ResNet-based)
Supports multi-scale feature extraction
"""

import torch
import torch.nn.functional as F
from torch import nn
from torchvision.models import resnet34, resnet18, resnet50
from util import repeat_interleave


class SpatialEncoder(nn.Module):
    """
    2D (Spatial/Pixel-aligned/local) image encoder
    """

    def __init__(
        self,
        backbone="resnet34",
        pretrained=True,
        num_layers=4,
        index_interp="bilinear",
        index_padding="border",
        upsample_interp="bilinear",
        feature_scale=1.0,
        use_first_pool=True,
        norm_type="batch",
        use_multi_scale=False,
    ):
        """
        :param backbone: Backbone network (resnet18, resnet34, resnet50)
        :param pretrained: Whether to use pretrained weights
        :param num_layers: Number of resnet layers to use (1-5)
        :param index_interp: Interpolation method for indexing (bilinear, nearest)
        :param index_padding: Padding mode for indexing (border, zeros, reflection)
        :param upsample_interp: Interpolation for upsampling (bilinear, nearest)
        :param feature_scale: Feature scaling factor
        :param use_first_pool: Whether to use first pooling layer
        :param norm_type: Normalization type (batch, group, none)
        :param use_multi_scale: Whether to use multi-scale feature fusion
        """
        super().__init__()

        # Store parameters
        self.use_multi_scale = use_multi_scale
        self.num_layers = num_layers
        self.feature_scale = feature_scale
        self.use_first_pool = use_first_pool

        # Interpolation and padding settings
        self.index_interp = index_interp
        self.index_padding = index_padding
        self.upsample_interp = upsample_interp
        self.align_corners = True if index_interp == "bilinear" else None

        # Build backbone
        print(f"✅ Using torchvision {backbone} encoder")

        if backbone == "resnet18":
            self.model = resnet18(pretrained=pretrained)
            self.latent_size = [64, 64, 128, 256, 512]
        elif backbone == "resnet34":
            self.model = resnet34(pretrained=pretrained)
            self.latent_size = [64, 64, 128, 256, 512]
        elif backbone == "resnet50":
            self.model = resnet50(pretrained=pretrained)
            self.latent_size = [64, 256, 512, 1024, 2048]
        else:
            raise NotImplementedError(f"Backbone {backbone} not supported")

        # Truncate to num_layers
        self.latent_size = self.latent_size[:num_layers]

        # Extract ResNet layers
        self.layers = nn.ModuleList()

        # Layer 0: conv1 + bn1 + relu (+ maxpool)
        layer0 = [self.model.conv1, self.model.bn1, self.model.relu]
        if use_first_pool:
            layer0.append(self.model.maxpool)
        self.layers.append(nn.Sequential(*layer0))

        # Layer 1-4: ResNet blocks
        if num_layers > 1:
            self.layers.append(self.model.layer1)
        if num_layers > 2:
            self.layers.append(self.model.layer2)
        if num_layers > 3:
            self.layers.append(self.model.layer3)
        if num_layers > 4:
            self.layers.append(self.model.layer4)

        # ✅ 修复：确保 latent_size 的类型一致性
        if not use_multi_scale:
            # 单尺度：只使用最后一层
            self.latent_size = self.latent_size[-1]
            print(f"✅ Single-scale encoder with latent size: {self.latent_size}")
        else:
            # 多尺度：使用所有层
            print(f"✅ Multi-scale fusion enabled with {num_layers} layers")
            print(f"✅ Feature dimensions per layer: {self.latent_size}")
            print(f"✅ Total latent size: {sum(self.latent_size)}")

        # Register latent storage
        self.latent = None
        self.latents = []

    def forward(self, x):
        """
        Forward pass to extract features
        :param x: (B, C, H, W) input images
        :return: features (single tensor or list of tensors)
        """
        x = x * self.feature_scale

        if self.use_multi_scale:
            # Extract multi-scale features
            self.latents = []
            for i, layer in enumerate(self.layers):
                x = layer(x)
                self.latents.append(x)

            # Store the last layer as main latent (for compatibility)
            self.latent = self.latents[-1]

            # ✅ 返回多尺度特征列表
            return self.latents
        else:
            # Extract only the last layer
            for layer in self.layers:
                x = layer(x)

            self.latent = x
            # ✅ 返回单尺度特征
            return x

    def index(self, uv, cam_z=None, image_size=(), z_bounds=None):
        """
        Get pixel-aligned image features at 2D image coordinates
        :param uv (B, N, 2) image points (x,y)
        :param cam_z (B, N) camera coordinate z (not used here)
        :param image_size image size (not used here)
        :param z_bounds (B, 2) depth bounds (not used here)
        :return (B, L, N) L is latent size
        """
        # ✅ 修复：安全处理 uv 的维度
        if uv.shape[0] == 1 and self.latent.shape[0] > 1:
            uv = uv.expand(self.latent.shape[0], -1, -1)

        # ✅ 修复：将 uv 从像素坐标转换到 [-1, 1] 的归一化坐标
        B, N, _ = uv.shape

        # 获取特征图的尺寸
        if self.use_multi_scale:
            # 使用最后一层的尺寸作为参考
            _, _, H, W = self.latents[-1].shape
        else:
            _, _, H, W = self.latent.shape

        # 转换 uv 到 [-1, 1] 范围
        uv_normalized = uv.clone()
        uv_normalized[:, :, 0] = (uv[:, :, 0] / (W - 1)) * 2 - 1  # x: [0, W] -> [-1, 1]
        uv_normalized[:, :, 1] = (uv[:, :, 1] / (H - 1)) * 2 - 1  # y: [0, H] -> [-1, 1]

        if self.use_multi_scale:
            # Multi-scale feature extraction
            features = []
            for i, latent in enumerate(self.latents):
                # 获取当前层的尺寸
                _, _, H_i, W_i = latent.shape

                # 为每一层重新归一化 uv
                uv_i = uv.clone()
                uv_i[:, :, 0] = (uv[:, :, 0] / (W_i - 1)) * 2 - 1
                uv_i[:, :, 1] = (uv[:, :, 1] / (H_i - 1)) * 2 - 1

                # uv: (B, N, 2) -> (B, 1, N, 2) for grid_sample
                uv_grid = uv_i.unsqueeze(1)  # (B, 1, N, 2)

                # Sample features
                samples = F.grid_sample(
                    latent,  # (B, C, H, W)
                    uv_grid,  # (B, 1, N, 2)
                    align_corners=self.align_corners,
                    mode=self.index_interp,
                    padding_mode=self.index_padding,
                )
                # samples: (B, C, 1, N) -> (B, C, N)
                features.append(samples.squeeze(2))

            # Concatenate multi-scale features along channel dimension
            return torch.cat(features, dim=1)  # (B, L_total, N)
        else:
            # Single-scale feature extraction
            uv_grid = uv_normalized.unsqueeze(1)  # (B, 1, N, 2)

            samples = F.grid_sample(
                self.latent,
                uv_grid,
                align_corners=self.align_corners,
                mode=self.index_interp,
                padding_mode=self.index_padding,
            )
            return samples.squeeze(2)  # (B, L, N)

    # ✅ 新增：获取统一尺寸的特征（用于特征融合）
    def get_unified_features(self, target_size=None):
        """
        获取统一尺寸的多尺度特征
        :param target_size: 目标尺寸 (H, W)，默认使用第一层的尺寸
        :return: (B, C_total, H, W) 统一尺寸的特征
        """
        if not self.use_multi_scale:
            return self.latent

        if target_size is None:
            target_size = self.latents[0].shape[-2:]

        # 将所有尺度的特征插值到统一尺寸
        unified_features = []
        for feat in self.latents:
            if feat.shape[-2:] != target_size:
                feat = F.interpolate(
                    feat,
                    size=target_size,
                    mode=self.upsample_interp,
                    align_corners=False
                )
            unified_features.append(feat)

        # 拼接所有尺度的特征
        return torch.cat(unified_features, dim=1)  # (B, C_total, H, W)

    @classmethod
    def from_conf(cls, conf, **kwargs):
        """
        Create encoder from config
        """
        return cls(
            backbone=conf.get("backbone", "resnet34"),
            pretrained=conf.get("pretrained", True),
            num_layers=conf.get("num_layers", 4),
            index_interp=conf.get("index_interp", "bilinear"),
            index_padding=conf.get("index_padding", "border"),
            upsample_interp=conf.get("upsample_interp", "bilinear"),
            feature_scale=conf.get("feature_scale", 1.0),
            use_first_pool=conf.get("use_first_pool", True),
            norm_type=conf.get("norm_type", "batch"),
            use_multi_scale=conf.get("use_multi_scale", False),
            **kwargs,
        )


# ← 别名，用于兼容性
ImageEncoder = SpatialEncoder

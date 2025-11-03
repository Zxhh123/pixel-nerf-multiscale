# src/model/feature_fusion.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from .attention import MultiViewAttention, CrossViewAttention, CBAM


class SmartFeatureFusion(nn.Module):
    """
    智能特征融合模块 - 支持多尺度特征融合

    适配 PixelNeRF 多尺度编码器的输出
    """

    def __init__(
            self,
            layer_dims,  # List[int] - 每层特征维度，如 [64, 64, 128, 256]
            output_dim=512,  # int - 输出特征维度
            use_attention=True,  # bool - 是否使用注意力机制
            dropout=0.0,  # float - Dropout率
            num_heads=8,  # int - 注意力头数
            use_cbam=True  # bool - 是否使用CBAM
    ):
        super(SmartFeatureFusion, self).__init__()

        self.layer_dims = layer_dims
        self.output_dim = output_dim
        self.use_attention = use_attention
        self.dropout = dropout
        self.num_heads = num_heads
        self.use_cbam = use_cbam

        # ✅ 为每个尺度创建投影层（统一到相同维度）
        self.projections = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(dim, output_dim // len(layer_dims), 1),
                nn.BatchNorm2d(output_dim // len(layer_dims)),
                nn.ReLU(inplace=True),
                nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
            )
            for dim in layer_dims
        ])

        # ✅ 注意力机制（如果启用）
        if use_attention:
            # 通道注意力：为每个尺度生成权重
            self.channel_attention = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(output_dim, len(layer_dims), 1),
                nn.Softmax(dim=1)
            )

            # 空间注意力
            self.spatial_attention = nn.Sequential(
                nn.Conv2d(output_dim, len(layer_dims), 7, padding=3),
                nn.Softmax(dim=1)
            )

        # ✅ CBAM注意力（可选）
        if use_cbam:
            self.cbam = CBAM(output_dim)

        # ✅ 最终融合层
        self.fusion = nn.Sequential(
            nn.Conv2d(output_dim, output_dim, 3, padding=1),
            nn.BatchNorm2d(output_dim),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        )

        # ✅ 特征增强（额外的卷积层）
        self.enhancement = nn.Sequential(
            nn.Conv2d(output_dim, output_dim, 1),
            nn.ReLU(inplace=True)
        )

    def forward(self, features_list):
        """
        前向传播

        Args:
            features_list: List[Tensor] - 多尺度特征列表
                          [(B, C1, H1, W1), (B, C2, H2, W2), ...]
                          例如: [(B, 64, H, W), (B, 64, H/2, W/2),
                                 (B, 128, H/4, W/4), (B, 256, H/8, W/8)]

        Returns:
            fused_feature: Tensor - 融合后的特征 (B, output_dim, H, W)
        """
        # ✅ 检查输入有效性
        if not features_list or len(features_list) == 0:
            raise ValueError("❌ features_list is empty!")

        # ✅ 统一尺寸到最大的特征图（通常是第一层）
        target_size = features_list[0].shape[2:]  # (H, W)

        # ✅ 投影并上采样所有特征到统一尺寸
        projected_features = []
        for feat, proj in zip(features_list, self.projections):
            # 投影到统一维度
            feat_proj = proj(feat)  # (B, output_dim//N, H_i, W_i)

            # 上采样到目标尺寸
            if feat_proj.shape[2:] != target_size:
                feat_proj = F.interpolate(
                    feat_proj,
                    size=target_size,
                    mode='bilinear',
                    align_corners=False
                )

            projected_features.append(feat_proj)

        # ✅ 拼接所有投影后的特征
        concat_features = torch.cat(projected_features, dim=1)  # (B, output_dim, H, W)

        # ✅ 应用注意力机制（如果启用）
        if self.use_attention:
            # 通道注意力
            channel_weights = self.channel_attention(concat_features)  # (B, N, 1, 1)

            # 空间注意力
            spatial_weights = self.spatial_attention(concat_features)  # (B, N, H, W)

            # 组合注意力权重
            combined_weights = channel_weights * spatial_weights  # (B, N, H, W)

            # 应用注意力加权
            weighted_features = []
            for i, feat in enumerate(projected_features):
                weight = combined_weights[:, i:i + 1, :, :]  # (B, 1, H, W)
                weighted_features.append(feat * weight)

            concat_features = torch.cat(weighted_features, dim=1)

        # ✅ 最终融合
        fused = self.fusion(concat_features)

        # ✅ 应用CBAM（如果启用）
        if self.use_cbam:
            fused = self.cbam(fused)

        # ✅ 特征增强
        fused = self.enhancement(fused)

        return fused


class AdaptiveFeatureSampler(nn.Module):
    """
    自适应特征采样器

    根据特征质量动态调整采样策略
    """

    def __init__(self, feature_dim):
        super(AdaptiveFeatureSampler, self).__init__()

        # 质量评估网络
        self.quality_net = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(feature_dim, feature_dim // 4, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(feature_dim // 4, 1, 1),
            nn.Sigmoid()
        )

    def compute_feature_quality(self, feature):
        """
        计算特征质量分数

        Args:
            feature: Tensor - (B, C, H, W)

        Returns:
            quality: Tensor - (B, 1) 质量分数 [0, 1]
        """
        quality = self.quality_net(feature).squeeze(-1).squeeze(-1)
        return quality

    def forward(self, features, top_k=None, quality_threshold=0.3):
        """
        自适应采样特征

        Args:
            features: List[Tensor] - [(B, C, H, W), ...]
            top_k: int - 保留前k个最好的特征
            quality_threshold: float - 质量阈值

        Returns:
            sampled_features: List[Tensor]
            sampled_indices: List[int]
        """
        if len(features) == 0:
            return [], []

        # 计算所有特征的质量
        qualities = []
        for feat in features:
            q = self.compute_feature_quality(feat)
            qualities.append(q)

        qualities = torch.stack(qualities, dim=1)  # (B, N)

        # 根据质量排序
        sorted_qualities, sorted_indices = torch.sort(qualities, dim=1, descending=True)

        # 选择高质量特征
        sampled_features = []
        sampled_indices = []

        for i in range(qualities.shape[1]):
            idx = sorted_indices[0, i].item()
            q = sorted_qualities[0, i].item()

            # 检查是否满足条件
            if top_k is not None and len(sampled_features) >= top_k:
                break

            if q >= quality_threshold:
                sampled_features.append(features[idx])
                sampled_indices.append(idx)

        # 至少保留一个特征
        if len(sampled_features) == 0:
            best_idx = sorted_indices[0, 0].item()
            sampled_features.append(features[best_idx])
            sampled_indices.append(best_idx)

        return sampled_features, sampled_indices


# ✅ 便捷接口
def create_feature_fusion(feature_dim=None, layer_dims=None, fusion_type='smart', **kwargs):
    """
    创建特征融合模块

    Args:
        feature_dim: int - 单一特征维度（用于旧版API）
        layer_dims: List[int] - 多尺度特征维度（用于新版API）
        fusion_type: str - 融合类型 ('smart', 'adaptive')
        **kwargs: 其他参数

    Returns:
        fusion_module: nn.Module
    """
    if fusion_type == 'smart':
        if layer_dims is not None:
            # 新版多尺度API
            return SmartFeatureFusion(layer_dims=layer_dims, **kwargs)
        elif feature_dim is not None:
            # 旧版单尺度API（向后兼容）
            return SmartFeatureFusion(layer_dims=[feature_dim], **kwargs)
        else:
            raise ValueError("Must provide either feature_dim or layer_dims")
    elif fusion_type == 'adaptive':
        if feature_dim is None:
            raise ValueError("feature_dim is required for adaptive fusion")
        return AdaptiveFeatureSampler(feature_dim)
    else:
        raise ValueError(f"Unknown fusion type: {fusion_type}")

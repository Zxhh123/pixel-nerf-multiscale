"""
多尺度特征融合模块 for PixelNeRF
基于多尺度融合语义增强网络设计
参考W-segnet的像素级特征融合架构
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiScaleFusionModule(nn.Module):
    """
    多尺度特征融合模块
    结合CNN和注意力机制进行特征增强
    """

    def __init__(self, latent_sizes, output_size):
        """
        Args:
            latent_sizes: List[int] 各层特征的通道数，如 [64, 128, 256, 512]
            output_size: int 输出特征的通道数
        """
        super().__init__()
        self.num_scales = len(latent_sizes)

        # 1x1卷积统一通道维度
        self.fusion_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_ch, output_size, kernel_size=1, bias=False),
                nn.BatchNorm2d(output_size),
                nn.ReLU(inplace=True)
            ) for in_ch in latent_sizes
        ])

        # 自适应注意力权重生成
        self.attention = nn.Sequential(
            nn.Conv2d(output_size * self.num_scales, self.num_scales, 1),
            nn.Softmax(dim=1)
        )

    def forward(self, features):
        """
        多尺度特征融合
        Args:
            features: List[Tensor] 不同尺度的特征 [(B, C_i, H_i, W_i), ...]
        Returns:
            Tensor: (B, output_size, H, W) 融合后的特征
        """
        # 获取目标尺寸（使用最深层的尺寸）
        target_size = features[-1].shape[2:]

        # 对齐所有特征到相同尺寸
        aligned = []
        for feat, conv in zip(features, self.fusion_convs):
            feat = conv(feat)  # 统一通道数
            if feat.shape[2:] != target_size:
                feat = F.interpolate(
                    feat, size=target_size,
                    mode='bilinear', align_corners=False
                )
            aligned.append(feat)

        # 计算注意力权重并融合
        concat = torch.cat(aligned, dim=1)  # (B, num_scales*C, H, W)
        weights = self.attention(concat)  # (B, num_scales, H, W)

        # 加权求和
        output = sum(
            w.unsqueeze(2) * f
            for w, f in zip(weights.split(1, dim=1), aligned)
        )

        return output

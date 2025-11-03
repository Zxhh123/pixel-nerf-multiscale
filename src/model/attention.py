import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiViewAttention(nn.Module):
    """
    多视点注意力模块
    用于学习不同视点特征的权重分配
    """

    def __init__(self, channels, num_heads=8):
        super(MultiViewAttention, self).__init__()
        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = channels // num_heads

        assert channels % num_heads == 0, "channels必须能被num_heads整除"

        # Query, Key, Value投影
        self.query = nn.Conv2d(channels, channels, 1)
        self.key = nn.Conv2d(channels, channels, 1)
        self.value = nn.Conv2d(channels, channels, 1)

        # 输出投影
        self.proj = nn.Conv2d(channels, channels, 1)
        self.norm = nn.LayerNorm(channels)

    def forward(self, x):
        """
        Args:
            x: [B, C, H, W] 输入特征
        Returns:
            out: [B, C, H, W] 注意力加权后的特征
        """
        B, C, H, W = x.shape

        # 生成Q, K, V
        q = self.query(x).view(B, self.num_heads, self.head_dim, H * W)
        k = self.key(x).view(B, self.num_heads, self.head_dim, H * W)
        v = self.value(x).view(B, self.num_heads, self.head_dim, H * W)

        # 注意力计算 [B, num_heads, H*W, H*W]
        attn = torch.einsum('bhdi,bhdj->bhij', q, k) / (self.head_dim ** 0.5)
        attn = F.softmax(attn, dim=-1)

        # 加权求和
        out = torch.einsum('bhij,bhdj->bhdi', attn, v)
        out = out.reshape(B, C, H, W)

        # 输出投影 + 残差连接
        out = self.proj(out)
        out = out + x

        # LayerNorm
        out = out.permute(0, 2, 3, 1)  # [B, H, W, C]
        out = self.norm(out)
        out = out.permute(0, 3, 1, 2)  # [B, C, H, W]

        return out


class CrossViewAttention(nn.Module):
    """
    跨视点注意力模块
    用于融合参考视点和目标视点的特征
    """

    def __init__(self, channels):
        super(CrossViewAttention, self).__init__()
        self.query_conv = nn.Conv2d(channels, channels // 8, 1)
        self.key_conv = nn.Conv2d(channels, channels // 8, 1)
        self.value_conv = nn.Conv2d(channels, channels, 1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, target_feat, ref_feat):
        """
        Args:
            target_feat: [B, C, H, W] 目标视点特征
            ref_feat: [B, C, H, W] 参考视点特征
        Returns:
            out: [B, C, H, W] 融合后的特征
        """
        B, C, H, W = target_feat.shape

        # Query来自目标视点
        query = self.query_conv(target_feat).view(B, -1, H * W).permute(0, 2, 1)

        # Key和Value来自参考视点
        key = self.key_conv(ref_feat).view(B, -1, H * W)
        value = self.value_conv(ref_feat).view(B, -1, H * W)

        # 注意力权重 [B, H*W, H*W]
        attn = torch.bmm(query, key)
        attn = F.softmax(attn, dim=-1)

        # 加权融合
        out = torch.bmm(value, attn.permute(0, 2, 1))
        out = out.view(B, C, H, W)

        # 残差连接
        out = self.gamma * out + target_feat

        return out


class SpatialAttention(nn.Module):
    """
    空间注意力模块
    用于关注图像中的重要区域
    """

    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        Args:
            x: [B, C, H, W]
        Returns:
            out: [B, C, H, W]
        """
        # 通道维度的最大值和平均值
        max_pool = torch.max(x, dim=1, keepdim=True)[0]
        avg_pool = torch.mean(x, dim=1, keepdim=True)

        # 拼接并生成空间注意力图
        concat = torch.cat([max_pool, avg_pool], dim=1)
        attn_map = self.conv(concat)
        attn_map = self.sigmoid(attn_map)

        return x * attn_map


class ChannelAttention(nn.Module):
    """
    通道注意力模块
    用于学习不同特征通道的重要性
    """

    def __init__(self, channels, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        Args:
            x: [B, C, H, W]
        Returns:
            out: [B, C, H, W]
        """
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        attn = self.sigmoid(avg_out + max_out)
        return x * attn


class CBAM(nn.Module):
    """
    CBAM: Convolutional Block Attention Module
    结合通道注意力和空间注意力
    """

    def __init__(self, channels, reduction=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.channel_attn = ChannelAttention(channels, reduction)
        self.spatial_attn = SpatialAttention(kernel_size)

    def forward(self, x):
        x = self.channel_attn(x)
        x = self.spatial_attn(x)
        return x

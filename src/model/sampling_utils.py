import torch
import torch.nn.functional as F
import numpy as np


def warp_image(img, depth, K, pose, target_K=None):
    """
    根据深度和位姿将图像变换到新视点

    Args:
        img: [B, C, H, W] 源图像
        depth: [B, 1, H, W] 深度图
        K: [B, 3, 3] 源视点相机内参
        pose: [B, 4, 4] 相对位姿 (从源到目标)
        target_K: [B, 3, 3] 目标视点相机内参

    Returns:
        warped_img: [B, C, H, W] 变换后的图像
        valid_mask: [B, 1, H, W] 有效像素掩码
    """
    B, C, H, W = img.shape
    device = img.device

    if target_K is None:
        target_K = K

    # 生成像素网格
    i_range = torch.arange(0, H, dtype=torch.float32, device=device)
    j_range = torch.arange(0, W, dtype=torch.float32, device=device)
    i, j = torch.meshgrid(i_range, j_range, indexing='ij')
    ones = torch.ones_like(i)

    # 像素坐标 [3, H, W]
    pixel_coords = torch.stack([j, i, ones], dim=0)
    pixel_coords = pixel_coords.unsqueeze(0).repeat(B, 1, 1, 1)

    # 反投影到3D空间
    K_inv = torch.inverse(K)
    cam_coords = torch.matmul(K_inv.unsqueeze(-1).unsqueeze(-1),
                              pixel_coords.unsqueeze(1))
    cam_coords = cam_coords.squeeze(1) * depth

    # 齐次坐标
    cam_coords_homo = torch.cat([
        cam_coords,
        torch.ones(B, 1, H, W, device=device)
    ], dim=1)

    # 变换到目标视点
    R = pose[:, :3, :3]
    t = pose[:, :3, 3:]

    cam_coords_flat = cam_coords.view(B, 3, -1)
    target_coords = torch.matmul(R, cam_coords_flat) + t
    target_coords = target_coords.view(B, 3, H, W)

    # 投影到目标图像平面
    target_pixel_coords = torch.matmul(target_K.unsqueeze(-1).unsqueeze(-1),
                                       target_coords.unsqueeze(1))
    target_pixel_coords = target_pixel_coords.squeeze(1)

    # 归一化
    Z = target_pixel_coords[:, 2:3, :, :].clamp(min=1e-3)
    target_pixel_coords = target_pixel_coords[:, :2, :, :] / Z

    # 归一化到[-1, 1]用于grid_sample
    target_pixel_coords[:, 0, :, :] = 2 * target_pixel_coords[:, 0, :, :] / (W - 1) - 1
    target_pixel_coords[:, 1, :, :] = 2 * target_pixel_coords[:, 1, :, :] / (H - 1) - 1

    # 转换为grid格式
    grid = target_pixel_coords.permute(0, 2, 3, 1)

    # 采样
    warped_img = F.grid_sample(img, grid, mode='bilinear',
                               padding_mode='zeros', align_corners=True)

    # 生成有效掩码
    valid_mask = (
            (grid[..., 0] >= -1) & (grid[..., 0] <= 1) &
            (grid[..., 1] >= -1) & (grid[..., 1] <= 1) &
            (Z.squeeze(1) > 0)
    ).float().unsqueeze(1)

    return warped_img, valid_mask


def compute_occlusion_mask(depth_src, depth_warped, threshold=0.01):
    """
    计算遮挡掩码

    Args:
        depth_src: [B, 1, H, W] 源深度图
        depth_warped: [B, 1, H, W] 变换后的深度图
        threshold: 深度差异阈值

    Returns:
        mask: [B, 1, H, W] 遮挡掩码
    """
    depth_diff = torch.abs(depth_src - depth_warped)
    mask = (depth_diff < threshold).float()
    return mask


def compute_reprojection_error(img1, img2_warped, mask=None):
    """
    计算重投影误差

    Args:
        img1: [B, C, H, W] 目标图像
        img2_warped: [B, C, H, W] 变换后的参考图像
        mask: [B, 1, H, W] 有效像素掩码

    Returns:
        error: [B, 1, H, W] 逐像素误差
    """
    error = torch.abs(img1 - img2_warped).mean(dim=1, keepdim=True)

    if mask is not None:
        error = error * mask

    return error


def bilinear_sampler(img, coords):
    """
    双线性采样

    Args:
        img: [B, C, H, W] 输入图像
        coords: [B, H, W, 2] 采样坐标 (x, y)

    Returns:
        sampled: [B, C, H, W] 采样结果
    """
    return F.grid_sample(img, coords, mode='bilinear',
                         padding_mode='border', align_corners=True)


def generate_depth_map(disparity, baseline=1.0, focal_length=1.0):
    """
    从视差图生成深度图

    Args:
        disparity: [B, 1, H, W] 视差图
        baseline: 基线距离
        focal_length: 焦距

    Returns:
        depth: [B, 1, H, W] 深度图
    """
    depth = (baseline * focal_length) / (disparity + 1e-6)
    return depth


def sample_random_views(num_views, total_views):
    """
    随机采样视点索引

    Args:
        num_views: 需要采样的视点数量
        total_views: 总视点数量

    Returns:
        indices: 采样的视点索引列表
    """
    indices = np.random.choice(total_views, num_views, replace=False)
    return indices.tolist()

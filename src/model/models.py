"""
Main model implementation with Multi-View Attention
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


class MultiViewAttention(nn.Module):
    """多视图注意力模块"""
    def __init__(self, feature_dim, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.feature_dim = feature_dim

        # 确保特征维度可以被头数整除
        assert feature_dim % num_heads == 0, f"feature_dim {feature_dim} must be divisible by num_heads {num_heads}"

        self.head_dim = feature_dim // num_heads

        # Q, K, V 投影层
        self.query = nn.Linear(feature_dim, feature_dim)
        self.key = nn.Linear(feature_dim, feature_dim)
        self.value = nn.Linear(feature_dim, feature_dim)

        # 输出投影
        self.out_proj = nn.Linear(feature_dim, feature_dim)

        # Layer Norm
        self.norm = nn.LayerNorm(feature_dim)

    def forward(self, features, mask=None):
        """
        Args:
            features: [B*NS, N_points, feature_dim] 多视图特征
            mask: [B*NS, N_points] 可选的掩码
        Returns:
            attended_features: [B, N_points, feature_dim] 注意力加权后的特征
        """
        B_NS, N, D = features.shape

        # 投影到 Q, K, V
        Q = self.query(features).view(B_NS, N, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.key(features).view(B_NS, N, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.value(features).view(B_NS, N, self.num_heads, self.head_dim).transpose(1, 2)

        # 计算注意力分数 [B_NS, num_heads, N, N]
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)

        # 应用掩码（如果有）
        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(2)  # [B_NS, 1, 1, N]
            scores = scores.masked_fill(mask == 0, -1e9)

        # Softmax 归一化
        attn_weights = torch.softmax(scores, dim=-1)

        # 加权求和
        attended = torch.matmul(attn_weights, V)  # [B_NS, num_heads, N, head_dim]
        attended = attended.transpose(1, 2).contiguous().view(B_NS, N, D)

        # 输出投影
        output = self.out_proj(attended)

        # 残差连接 + Layer Norm
        output = self.norm(output + features)

        return output


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

        # ========== 多视图注意力配置（原有的，保留） ==========
        self.use_attention = conf.get_bool("use_attention", True)
        self.attention_heads = conf.get_int("attention_heads", 4)

        # ========== 新增：智能特征融合配置 ==========
        self.use_smart_fusion = conf.get_bool("use_smart_fusion", False)
        self.use_adaptive_sampling = conf.get_bool("use_adaptive_sampling", False)
        self.fusion_heads = conf.get_int("fusion_heads", 8)
        self.fusion_dropout = conf.get_float("fusion_dropout", 0.1)
        self.fusion_type = conf.get_string("fusion_type", "attention")  # attention, weighted, adaptive
        self.use_cbam = conf.get_bool("use_cbam", True)
        self.quality_threshold = conf.get_float("quality_threshold", 0.3)

        # ========== 修复：统一处理 encoder.latent_size ==========
        encoder_latent_size = self.encoder.latent_size

        if isinstance(encoder_latent_size, (list, tuple)):
            # ✅ 多尺度特征：求和得到总维度
            self.latent_size = sum(int(x) for x in encoder_latent_size)
            print(f"✅ Multi-scale encoder detected:")
            print(f"   - Layer sizes: {encoder_latent_size}")
            print(f"   - Total latent size: {self.latent_size}")
        else:
            # ✅ 单尺度特征：直接使用
            self.latent_size = int(encoder_latent_size)
            print(f"✅ Single-scale encoder:")
            print(f"   - Latent size: {self.latent_size}")

        # ========== 初始化多视图注意力模块（原有的，保留） ==========
        if self.use_attention and not self.use_smart_fusion:
            self.attention = MultiViewAttention(
                feature_dim=self.latent_size,
                num_heads=self.attention_heads
            )
            print(f"✅ Legacy Multi-View Attention enabled:")
            print(f"   - Attention heads: {self.attention_heads}")
            print(f"   ⚠️  Consider upgrading to smart_fusion for better performance")

        # ========== 新增：初始化智能特征融合模块 ==========
        if self.use_smart_fusion:
            try:
                from .feature_fusion import SmartFeatureFusion

                self.feature_fusion = SmartFeatureFusion(
                    feature_dim=self.latent_size,
                    num_heads=self.fusion_heads,
                    dropout=self.fusion_dropout,
                    fusion_type=self.fusion_type,
                    use_cbam=self.use_cbam
                )
                print(f"✅ Smart Feature Fusion enabled:")
                print(f"   - Fusion type: {self.fusion_type}")
                print(f"   - Fusion heads: {self.fusion_heads}")
                print(f"   - Dropout: {self.fusion_dropout}")
                print(f"   - CBAM attention: {self.use_cbam}")

                # ✅ 如果启用了智能融合，禁用旧的注意力机制
                if self.use_attention:
                    print(f"   ⚠️  Disabling legacy attention (using smart fusion instead)")
                    self.use_attention = False

            except ImportError as e:
                print(f"❌ Failed to import SmartFeatureFusion: {e}")
                print(f"   Falling back to legacy attention mechanism")
                self.use_smart_fusion = False
                if self.use_attention:
                    self.attention = MultiViewAttention(
                        feature_dim=self.latent_size,
                        num_heads=self.attention_heads
                    )

        # ========== 新增：初始化自适应采样器 ==========
        if self.use_adaptive_sampling:
            try:
                from .feature_fusion import AdaptiveFeatureSampler

                self.feature_sampler = AdaptiveFeatureSampler(
                    feature_dim=self.latent_size,
                    quality_threshold=self.quality_threshold
                )
                print(f"✅ Adaptive Feature Sampling enabled:")
                print(f"   - Quality threshold: {self.quality_threshold}")
                print(f"   - Feature dim: {self.latent_size}")

            except ImportError as e:
                print(f"❌ Failed to import AdaptiveFeatureSampler: {e}")
                print(f"   Disabling adaptive sampling")
                self.use_adaptive_sampling = False

        # ========== 计算 MLP 输入维度 ==========
        d_latent = self.latent_size if self.use_encoder else 0
        d_in = 3 if self.use_xyz else 1

        # 处理视角方向编码
        if self.use_viewdirs and self.use_code_viewdirs:
            d_in += 3

        # 位置编码
        if self.use_code and d_in > 0:
            self.code = PositionalEncoding.from_conf(conf["code"], d_in=d_in)
            d_in = self.code.d_out

        if self.use_viewdirs and not self.use_code_viewdirs:
            d_in += 3

        # ========== 全局编码器（如果启用） ==========
        if self.use_global_encoder:
            self.global_encoder = ImageEncoder.from_conf(conf["global_encoder"])

            # ✅ 处理全局编码器的 latent_size
            global_latent = self.global_encoder.latent_size
            if isinstance(global_latent, (list, tuple)):
                self.global_latent_size = sum(int(x) for x in global_latent)
            else:
                self.global_latent_size = int(global_latent)

            d_latent += self.global_latent_size
            print(f"✅ Global encoder enabled:")
            print(f"   - Global latent size: {self.global_latent_size}")
            print(f"   - Total latent size: {d_latent}")

        d_out = 4  # RGB + density

        # ========== 打印 MLP 配置信息 ==========
        print(f"\n{'=' * 60}")
        print(f"MLP Configuration Summary:")
        print(f"{'=' * 60}")
        print(f"  Input dimension (d_in):        {d_in}")
        print(f"    ├─ use_xyz:                  {self.use_xyz}")
        print(f"    ├─ use_viewdirs:             {self.use_viewdirs}")
        print(f"    └─ use_code (pos encoding):  {self.use_code}")
        print(f"  Latent dimension (d_latent):   {d_latent}")
        print(f"    ├─ encoder latent:           {self.latent_size}")
        if self.use_global_encoder:
            print(f"    └─ global encoder latent:    {self.global_latent_size}")
        print(f"  Output dimension (d_out):      {d_out} (RGB + σ)")
        print(f"{'=' * 60}\n")

        # ========== 创建 MLP 网络 ==========
        self.mlp_coarse = make_mlp(conf["mlp_coarse"], d_in, d_latent, d_out=d_out)
        self.mlp_fine = make_mlp(
            conf["mlp_fine"], d_in, d_latent, d_out=d_out, allow_empty=True
        )
        print(f"✅ MLP networks created:")
        print(f"   - Coarse MLP: {d_in} + {d_latent} -> {d_out}")
        if self.mlp_fine is not None:
            print(f"   - Fine MLP:   {d_in} + {d_latent} -> {d_out}")

        # ========== 停止编码器梯度（如果需要） ==========
        if self.stop_encoder_grad:
            for param in self.encoder.parameters():
                param.requires_grad = False
            print(f"⚠️  Encoder gradients stopped (fine-tuning mode)")

        # ========== 注册缓冲区 ==========
        self.register_buffer("poses", torch.empty(1, 3, 4), persistent=False)
        self.register_buffer("image_shape", torch.empty(2), persistent=False)
        self.register_buffer("focal", torch.empty(1, 2), persistent=False)
        self.register_buffer("c", torch.empty(1, 2), persistent=False)

        # ========== 保存维度信息 ==========
        self.d_in = d_in
        self.d_out = d_out
        self.d_latent = d_latent

        # ========== 初始化计数器 ==========
        self.num_objs = 0
        self.num_views_per_obj = 1

        print(f"\n{'=' * 60}")
        print(f"✅ PixelNeRFNet initialization complete!")
        print(f"{'=' * 60}\n")

    def encode(self, images, poses, focal, z_bounds=None, c=None):
        """
        编码输入图像和相机参数
        :param images (NS, 3, H, W) or (B, NS, 3, H, W)
        :param poses (NS, 4, 4) or (B, NS, 4, 4)
        :param focal focal length
        :param z_bounds ignored
        :param c principal point
        """
        self.num_objs = images.size(0)

        # ✅ 处理输入维度
        if len(images.shape) == 5:
            # (B, NS, 3, H, W) -> 多个物体，每个物体多个视点
            assert len(poses.shape) == 4
            assert poses.size(1) == images.size(1)
            self.num_views_per_obj = images.size(1)

            batch_size = images.size(0)
            num_views = images.size(1)

            # 展平为 (B*NS, 3, H, W)
            images = images.reshape(-1, *images.shape[2:])
            poses = poses.reshape(-1, 4, 4)
        else:
            # (NS, 3, H, W) -> 单个物体，多个视点
            self.num_views_per_obj = 1
            batch_size = images.size(0)
            num_views = 1

        # ✅ 编码所有视点的图像
        if self.use_encoder:
            # 提取特征（encoder 会自动处理多尺度）
            all_latents = self.encoder(images)  # 返回 (B*NS, C, H, W) 或 List[(B*NS, C_i, H, W)]

            # ✅ 处理多尺度特征
            if isinstance(all_latents, (list, tuple)):
                # 多尺度特征 -> 使用 encoder 的统一方法
                all_latents = self.encoder.get_unified_features()  # (B*NS, C_total, H, W)
                print(f"✅ Multi-scale features unified: {all_latents.shape}")

            # ✅ 多视点特征融合
            if num_views > 1 and (self.use_smart_fusion or self.use_adaptive_sampling):
                C, H, W = all_latents.shape[1:]

                # 重塑为 (B, NS, C, H, W)
                all_latents = all_latents.reshape(batch_size, num_views, C, H, W)

                # 分离每个视点的特征
                latent_list = [all_latents[:, i] for i in range(num_views)]  # List[(B, C, H, W)]

                # ✅ 自适应采样（可选）
                if self.use_adaptive_sampling:
                    latent_list, valid_indices = self.feature_sampler(
                        latent_list,
                        top_k=min(3, num_views),
                        quality_threshold=0.3
                    )
                    print(f"✅ Adaptive sampling: {len(valid_indices)}/{num_views} views selected")

                # ✅ 智能特征融合
                if self.use_smart_fusion:
                    fused_latent = self.feature_fusion(latent_list)  # (B, C, H, W)
                    print(f"✅ Smart fusion: {len(latent_list)} views -> 1 fused feature")
                else:
                    # 简单平均（向后兼容）
                    fused_latent = torch.stack(latent_list, dim=0).mean(dim=0)
                    print(f"⚠️ Simple averaging: {len(latent_list)} views")

                # ✅ 复制融合特征到所有视点（保持形状兼容性）
                self.encoder.latent = fused_latent.unsqueeze(1).repeat(1, num_views, 1, 1, 1).reshape(-1, C, H, W)

            else:
                # 单视点或不使用融合，直接使用
                self.encoder.latent = all_latents
                if num_views > 1:
                    print(f"⚠️ Multi-view input but fusion disabled")

        # ✅ 处理相机位姿
        rot = poses[:, :3, :3].transpose(1, 2)  # (B*NS, 3, 3)
        trans = -torch.bmm(rot, poses[:, :3, 3:])  # (B*NS, 3, 1)
        self.poses = torch.cat((rot, trans), dim=-1)  # (B*NS, 3, 4)

        # ✅ 存储图像形状
        self.image_shape[0] = images.shape[-1]
        self.image_shape[1] = images.shape[-2]

        # ✅ 处理焦距
        if len(focal.shape) == 0:
            focal = focal[None, None].repeat((1, 2))
        elif len(focal.shape) == 1:
            focal = focal.unsqueeze(-1).repeat((1, 2))
        else:
            focal = focal.clone()
        self.focal = focal.float()
        self.focal[..., 1] *= -1.0

        # ✅ 处理主点
        if c is None:
            c = (self.image_shape * 0.5).unsqueeze(0)
        elif len(c.shape) == 0:
            c = c[None, None].repeat((1, 2))
        elif len(c.shape) == 1:
            c = c.unsqueeze(-1).repeat((1, 2))
        self.c = c

        # ✅ 全局编码器（如果启用）
        if self.use_global_encoder:
            self.global_encoder(images)

    def forward(self, xyz, coarse=True, viewdirs=None, far=False):
        """
        Predict (r, g, b, sigma) at world space points xyz.
        Please call encode first!
        :param xyz (SB, B, 3)
        :return (SB, B, 4) r g b sigma
        """
        with profiler.record_function("model_inference"):
            SB, B, _ = xyz.shape
            NS = self.num_views_per_obj

            # Transform query points
            xyz = repeat_interleave(xyz, NS)
            xyz_rot = torch.matmul(self.poses[:, None, :3, :3], xyz.unsqueeze(-1))[..., 0]
            xyz = xyz_rot + self.poses[:, None, :3, 3]

            if self.d_in > 0:
                # Encode xyz coordinates
                if self.use_xyz:
                    z_feature = xyz_rot.reshape(-1, 3) if self.normalize_z else xyz.reshape(-1, 3)
                else:
                    z_feature = -xyz_rot[..., 2].reshape(-1, 1) if self.normalize_z else -xyz[..., 2].reshape(-1, 1)

                if self.use_code and not self.use_code_viewdirs:
                    z_feature = self.code(z_feature)

                if self.use_viewdirs:
                    assert viewdirs is not None
                    viewdirs = viewdirs.reshape(SB, B, 3, 1)
                    viewdirs = repeat_interleave(viewdirs, NS)
                    viewdirs = torch.matmul(self.poses[:, None, :3, :3], viewdirs)
                    viewdirs = viewdirs.reshape(-1, 3)
                    z_feature = torch.cat((z_feature, viewdirs), dim=1)

                if self.use_code and self.use_code_viewdirs:
                    z_feature = self.code(z_feature)

                mlp_input = z_feature

            if self.use_encoder:
                # Get image features
                uv = -xyz[:, :, :2] / xyz[:, :, 2:]
                uv *= repeat_interleave(
                    self.focal.unsqueeze(1), NS if self.focal.shape[0] > 1 else 1
                )
                uv += repeat_interleave(
                    self.c.unsqueeze(1), NS if self.c.shape[0] > 1 else 1
                )

                latent = self.encoder.index(uv, None, self.image_shape)

                if self.stop_encoder_grad:
                    latent = latent.detach()

                # ✅ 修复：确保 reshape 使用正确的维度
                latent = latent.transpose(1, 2).reshape(-1, self.latent_size)

                # ✅ 新增：使用注意力机制融合多视图特征
                if self.use_attention and NS > 1:
                    # Reshape to [SB, NS, B, latent_size]
                    latent_views = latent.reshape(SB, NS, B, self.latent_size)

                    # 对每个查询点，在多个视图间做注意力
                    attended_features = []
                    for i in range(SB):
                        # [NS, B, latent_size] -> [B, NS, latent_size]
                        view_features = latent_views[i].permute(1, 0, 2)

                        # 应用注意力 [B, NS, latent_size]
                        attended = self.attention(view_features)

                        # 平均池化多视图 [B, latent_size]
                        attended = attended.mean(dim=1)
                        attended_features.append(attended)

                    # 合并 batch [SB, B, latent_size]
                    latent = torch.stack(attended_features, dim=0)
                    latent = latent.reshape(-1, self.latent_size)
                else:
                    # 不使用注意力时，保持原有逻辑
                    pass

                if self.d_in == 0:
                    mlp_input = latent
                else:
                    mlp_input = torch.cat((latent, z_feature), dim=-1)

            if self.use_global_encoder:
                global_latent = self.global_encoder.latent
                assert mlp_input.shape[0] % global_latent.shape[0] == 0
                num_repeats = mlp_input.shape[0] // global_latent.shape[0]
                global_latent = repeat_interleave(global_latent, num_repeats)
                mlp_input = torch.cat((global_latent, mlp_input), dim=-1)

            # Run NeRF network
            combine_index = None
            dim_size = None

            if coarse or self.mlp_fine is None:
                mlp_output = self.mlp_coarse(
                    mlp_input,
                    combine_inner_dims=(self.num_views_per_obj, B),
                    combine_index=combine_index,
                    dim_size=dim_size,
                )
            else:
                mlp_output = self.mlp_fine(
                    mlp_input,
                    combine_inner_dims=(self.num_views_per_obj, B),
                    combine_index=combine_index,
                    dim_size=dim_size,
                )

            mlp_output = mlp_output.reshape(-1, B, self.d_out)

            rgb = mlp_output[..., :3]
            sigma = mlp_output[..., 3:4]

            # ✅ 确保输出在 [0, 1] 范围
            output_list = [torch.sigmoid(rgb), torch.relu(sigma)]
            output = torch.cat(output_list, dim=-1)
            output = output.reshape(SB, B, -1)

        return output

    def load_weights(self, args, opt_init=False, strict=True, device=None):
        """
        Helper for loading weights according to argparse arguments.
        Your can put a checkpoint at checkpoints/<exp>/pixel_nerf_init to use as initialization.
        :param opt_init if true, loads from init checkpoint instead of usual even when resuming
        """
        # TODO: make backups
        if opt_init and not args.resume:
            return
        ckpt_name = (
            "pixel_nerf_init" if opt_init or not args.resume else "pixel_nerf_latest"
        )
        model_path = "%s/%s/%s" % (args.checkpoints_path, args.name, ckpt_name)

        if device is None:
            device = self.poses.device

        if os.path.exists(model_path):
            print("Load", model_path)
            self.load_state_dict(
                torch.load(model_path, map_location=device), strict=strict
            )
        elif not opt_init:
            warnings.warn(
                (
                        "WARNING: {} does not exist, not loaded!! Model will be re-initialized.\n"
                        + "If you are trying to load a pretrained model, STOP since it's not in the right place. "
                        + "If training, unless you are startin a new experiment, please remember to pass --resume."
                ).format(model_path)
            )
        return self

    def save_weights(self, args, opt_init=False):
        """
        Helper for saving weights according to argparse arguments
        :param opt_init if true, saves from init checkpoint instead of usual
        """
        from shutil import copyfile

        ckpt_name = "pixel_nerf_init" if opt_init else "pixel_nerf_latest"
        backup_name = "pixel_nerf_init_backup" if opt_init else "pixel_nerf_backup"

        ckpt_path = osp.join(args.checkpoints_path, args.name, ckpt_name)
        ckpt_backup_path = osp.join(args.checkpoints_path, args.name, backup_name)

        if osp.exists(ckpt_path):
            copyfile(ckpt_path, ckpt_backup_path)
        torch.save(self.state_dict(), ckpt_path)
        return self

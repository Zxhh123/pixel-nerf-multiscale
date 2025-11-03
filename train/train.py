"""
Training to a set of multiple objects (e.g. ShapeNet or DTU)
Enhanced with mixed precision training and gradient checking
"""
import sys
import os

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
)

import warnings
import trainlib
from model import make_model, loss
from render import NeRFRenderer
from data import get_split_dataset
import util
import numpy as np
import torch.nn.functional as F
import torch
from dotmap import DotMap

# ✅ 添加混合精度训练支持
from torch.cuda.amp import autocast, GradScaler


def extra_args(parser):
    parser.add_argument(
        "--batch_size", "-B", type=int, default=4, help="Object batch size ('SB')"
    )
    parser.add_argument(
        "--nviews",
        "-V",
        type=str,
        default="1",
        help="Number of source views (multiview); put multiple (space delim) to pick randomly per batch ('NV')",
    )
    parser.add_argument(
        "--freeze_enc",
        action="store_true",
        default=None,
        help="Freeze encoder weights and only train MLP",
    )
    parser.add_argument(
        "--no_bbox_step",
        type=int,
        default=100000,
        help="Step to stop using bbox sampling",
    )
    parser.add_argument(
        "--fixed_test",
        action="store_true",
        default=None,
        help="Use fixed test views",
    )

    # ✅ 混合精度训练参数
    parser.add_argument(
        "--use_amp",
        action="store_true",
        default=False,
        help="Use automatic mixed precision training",
    )
    parser.add_argument(
        "--no_amp",
        action="store_true",
        default=False,
        help="Disable automatic mixed precision training (override config)",
    )

    # ✅ 梯度检查参数
    parser.add_argument(
        "--check_gradients",
        action="store_true",
        default=False,
        help="Enable gradient checking and clipping",
    )
    parser.add_argument(
        "--grad_clip",
        type=float,
        default=1.0,
        help="Gradient clipping threshold",
    )

    return parser


args, conf = util.args.parse_args(extra_args, training=True, default_ray_batch_size=128)
device = util.get_cuda(args.gpu_id[0])

# ✅ 处理 AMP 标志（优先级：命令行 > 配置文件）
if args.no_amp:
    args.use_amp = False
elif not args.use_amp and conf.get_bool("train.use_amp", False):
    args.use_amp = True

# ✅ 处理梯度裁剪配置
if not args.check_gradients and conf.get_bool("train.check_gradients", False):
    args.check_gradients = True
if args.grad_clip == 1.0:  # 默认值
    args.grad_clip = conf.get_float("train.grad_clip", 1.0)

print("\n" + "=" * 80)
print("🚀 PIXELNERF TRAINING - ENHANCED VERSION")
print("=" * 80)
print(f"📍 Device: {device}")
print(f"📦 Batch size: {args.batch_size}")
print(f"👁️  Number of views: {args.nviews}")
print(f"🎯 Ray batch size: {args.ray_batch_size}")
print(f"⚡ Mixed precision (AMP): {'✅ Enabled' if args.use_amp else '❌ Disabled'}")
print(f"❄️  Freeze encoder: {'✅ Yes' if args.freeze_enc else '❌ No'}")
print(f"✂️  Gradient clipping: {'✅ Enabled' if args.check_gradients else '❌ Disabled'} (threshold: {args.grad_clip})")
print("=" * 80 + "\n")

# ========== 加载数据集 ==========
print("📂 Loading datasets...")
dset, val_dset, _ = get_split_dataset(args.dataset_format, args.datadir)
print(f"✅ Dataset loaded:")
print(f"   - Train samples: {len(dset)}")
print(f"   - Val samples: {len(val_dset) if val_dset is not None else 0}")
print(f"   - z_near: {dset.z_near}, z_far: {dset.z_far}")
print(f"   - lindisp: {dset.lindisp}")

# ========== 创建模型 ==========
print("\n🏗️  Creating model with enhanced features...")
net = make_model(conf["model"]).to(device=device)
net.stop_encoder_grad = args.freeze_enc

if args.freeze_enc:
    print("❄️  Encoder frozen (fine-tuning mode)")
    net.encoder.eval()
    for param in net.encoder.parameters():
        param.requires_grad = False

# ✅ 打印模型配置信息
print(f"\n📊 Model Configuration:")
print(f"   - Encoder type: {net.encoder.__class__.__name__}")
print(f"   - Latent size: {net.latent_size}")
print(f"   - Use encoder: {net.use_encoder}")
print(f"   - Use xyz: {net.use_xyz}")
print(f"   - Use viewdirs: {net.use_viewdirs}")

# ✅ 打印新增功能状态
if hasattr(net, 'use_smart_fusion'):
    print(f"   - Smart fusion: {'✅ Enabled' if net.use_smart_fusion else '❌ Disabled'}")
    if net.use_smart_fusion:
        print(f"     • Fusion type: {net.fusion_type}")
        print(f"     • Fusion heads: {net.fusion_heads}")
        print(f"     • CBAM: {'✅' if net.use_cbam else '❌'}")

if hasattr(net, 'use_adaptive_sampling'):
    print(f"   - Adaptive sampling: {'✅ Enabled' if net.use_adaptive_sampling else '❌ Disabled'}")
    if net.use_adaptive_sampling:
        print(f"     • Quality threshold: {net.quality_threshold}")

# ✅ 打印编码器信息
if hasattr(net.encoder, 'use_multi_scale'):
    print(f"   - Multi-scale encoder: {'✅ Enabled' if net.encoder.use_multi_scale else '❌ Disabled'}")
    if net.encoder.use_multi_scale:
        print(f"     • Feature scales: {net.encoder.latent_size}")

# ========== 创建渲染器 ==========
print("\n🎨 Creating renderer...")
renderer = NeRFRenderer.from_conf(
    conf["renderer"],
    lindisp=dset.lindisp,
).to(device=device)

# ========== 并行化 ==========
print(f"\n⚡ Setting up parallelization on GPUs: {args.gpu_id}")
render_par = renderer.bind_parallel(net, args.gpu_id).eval()

nviews = list(map(int, args.nviews.split()))
print(f"✅ Multi-view setup: {nviews} views")


# ========== 创建训练器 ==========
class PixelNeRFTrainer(trainlib.Trainer):
    def __init__(self):
        super().__init__(net, dset, val_dset, args, conf["train"], device=device)
        self.renderer_state_path = "%s/%s/_renderer" % (
            self.args.checkpoints_path,
            self.args.name,
        )

        # ✅ 初始化混合精度训练
        self.use_amp = args.use_amp
        if self.use_amp:
            self.scaler = GradScaler()
            print("✅ GradScaler initialized for mixed precision training")
        else:
            self.scaler = None

        # ✅ 梯度裁剪配置
        self.check_gradients = args.check_gradients
        self.grad_clip = args.grad_clip

        self.lambda_coarse = conf.get_float("loss.lambda_coarse")
        self.lambda_fine = conf.get_float("loss.lambda_fine", 1.0)
        print(
            "lambda coarse {} and fine {}".format(self.lambda_coarse, self.lambda_fine)
        )
        self.rgb_coarse_crit = loss.get_rgb_loss(conf["loss.rgb"], True)
        fine_loss_conf = conf["loss.rgb"]
        if "rgb_fine" in conf["loss"]:
            fine_loss_conf = conf["loss.rgb_fine"]
        self.rgb_fine_crit = loss.get_rgb_loss(fine_loss_conf, False)

        if args.resume:
            # ✅ 加载检查点
            checkpoint = net.load_weights(args, device=device)
            if checkpoint is not None:
                if "optimizer_state_dict" in checkpoint and self.optim is not None:
                    self.optim.load_state_dict(checkpoint["optimizer_state_dict"])
                    print("✅ Optimizer state loaded")
                if "epoch" in checkpoint:
                    self.start_epoch = checkpoint["epoch"] + 1
                    print(f"✅ Resuming from epoch {self.start_epoch}")

        self.z_near = dset.z_near
        self.z_far = dset.z_far

        self.use_bbox = args.no_bbox_step > 0

    def post_batch(self, epoch, batch):
        """
        每个 batch 后的处理
        """
        renderer.sched_step(args.batch_size)

    def extra_save_state(self):
        """
        保存额外的状态
        """
        torch.save(renderer.state_dict(), self.renderer_state_path)

    def calc_losses(self, data, is_train=True, global_step=0):
        """
        计算损失
        """
        if "images" not in data:
            return {}
        all_images = data["images"].to(device=device)  # (SB, NV, 3, H, W)

        SB, NV, _, H, W = all_images.shape
        all_poses = data["poses"].to(device=device)  # (SB, NV, 4, 4)
        all_bboxes = data.get("bbox")  # (SB, NV, 4)  cmin rmin cmax rmax
        all_focals = data["focal"]  # (SB)
        all_c = data.get("c")  # (SB)

        if self.use_bbox and global_step >= args.no_bbox_step:
            self.use_bbox = False
            print(">>> Stopped using bbox sampling @ iter", global_step)

        if not is_train or not self.use_bbox:
            all_bboxes = None

        all_rgb_gt = []
        all_rays = []

        curr_nviews = nviews[torch.randint(0, len(nviews), ()).item()]
        if curr_nviews == 1:
            image_ord = torch.randint(0, NV, (SB, 1))
        else:
            image_ord = torch.empty((SB, curr_nviews), dtype=torch.long)
        for obj_idx in range(SB):
            if all_bboxes is not None:
                bboxes = all_bboxes[obj_idx]
            images = all_images[obj_idx]  # (NV, 3, H, W)
            poses = all_poses[obj_idx]  # (NV, 4, 4)
            focal = all_focals[obj_idx]
            c = None
            if all_c is not None:
                c = all_c[obj_idx]
            if curr_nviews > 1:
                # Somewhat inefficient, don't know better way
                image_ord[obj_idx] = torch.from_numpy(
                    np.random.choice(NV, curr_nviews, replace=False)
                )
            images_0to1 = images * 0.5 + 0.5

            cam_rays = util.gen_rays(
                poses,
                W,
                H,
                focal,
                self.z_near,
                self.z_far,
                c=c,
            )  # (NV, H, W, 8)
            rgb_gt_all = images_0to1
            rgb_gt_all = (
                rgb_gt_all.permute(0, 2, 3, 1).contiguous().reshape(-1, 3)
            )  # (NV, H, W, 3)

            if all_bboxes is not None:
                pix = util.bbox_sample(bboxes, args.ray_batch_size)
                pix_inds = pix[..., 0] * H * W + pix[..., 1] * W + pix[..., 2]
            else:
                pix_inds = torch.randint(0, NV * H * W, (args.ray_batch_size,))

            rgb_gt = rgb_gt_all[pix_inds]  # (ray_batch_size, 3)
            rays = cam_rays.view(-1, cam_rays.shape[-1])[pix_inds].to(
                device=device
            )  # (ray_batch_size, 8)

            all_rgb_gt.append(rgb_gt)
            all_rays.append(rays)

        all_rgb_gt = torch.stack(all_rgb_gt)  # (SB, ray_batch_size, 3)
        all_rays = torch.stack(all_rays)  # (SB, ray_batch_size, 8)

        image_ord = image_ord.to(device)
        src_images = util.batched_index_select_nd(
            all_images, image_ord
        )  # (SB, NS, 3, H, W)
        src_poses = util.batched_index_select_nd(all_poses, image_ord)  # (SB, NS, 4, 4)

        all_bboxes = all_poses = all_images = None

        # ✅ 使用混合精度训练
        if self.use_amp and is_train:
            with autocast(device_type='cuda'):
                net.encode(
                    src_images,
                    src_poses,
                    all_focals.to(device=device),
                    c=all_c.to(device=device) if all_c is not None else None,
                )

                render_dict = DotMap(render_par(all_rays, want_weights=True))
                coarse = render_dict.coarse
                fine = render_dict.fine
                using_fine = len(fine) > 0

                loss_dict = {}

                rgb_loss = self.rgb_coarse_crit(coarse.rgb, all_rgb_gt)
                loss_dict["rc"] = rgb_loss.item() * self.lambda_coarse
                if using_fine:
                    fine_loss = self.rgb_fine_crit(fine.rgb, all_rgb_gt)
                    rgb_loss = rgb_loss * self.lambda_coarse + fine_loss * self.lambda_fine
                    loss_dict["rf"] = fine_loss.item() * self.lambda_fine

                loss = rgb_loss
                loss_dict["t"] = loss.item()
        else:
            # 标准训练（无混合精度）
            net.encode(
                src_images,
                src_poses,
                all_focals.to(device=device),
                c=all_c.to(device=device) if all_c is not None else None,
            )

            render_dict = DotMap(render_par(all_rays, want_weights=True))
            coarse = render_dict.coarse
            fine = render_dict.fine
            using_fine = len(fine) > 0

            loss_dict = {}

            rgb_loss = self.rgb_coarse_crit(coarse.rgb, all_rgb_gt)
            loss_dict["rc"] = rgb_loss.item() * self.lambda_coarse
            if using_fine:
                fine_loss = self.rgb_fine_crit(fine.rgb, all_rgb_gt)
                rgb_loss = rgb_loss * self.lambda_coarse + fine_loss * self.lambda_fine
                loss_dict["rf"] = fine_loss.item() * self.lambda_fine

            loss = rgb_loss
            loss_dict["t"] = loss.item()

        return loss, loss_dict

    def train_step(self, data, global_step):
        """
        单步训练
        """
        self.optim.zero_grad()

        # ✅ 计算损失（内部处理混合精度）
        loss, loss_dict = self.calc_losses(data, is_train=True, global_step=global_step)

        # ✅ 反向传播
        if self.use_amp:
            self.scaler.scale(loss).backward()

            # ✅ 梯度裁剪
            if self.check_gradients:
                self.scaler.unscale_(self.optim)
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    net.parameters(), self.grad_clip
                )
                if global_step % 100 == 0:
                    print(f"   Gradient norm: {grad_norm:.4f}")

            self.scaler.step(self.optim)
            self.scaler.update()
        else:
            loss.backward()

            # ✅ 梯度裁剪
            if self.check_gradients:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    net.parameters(), self.grad_clip
                )
                if global_step % 100 == 0:
                    print(f"   Gradient norm: {grad_norm:.4f}")

            self.optim.step()

        return loss_dict

    def eval_step(self, data, global_step):
        """
        验证步骤
        """
        renderer.eval()
        losses = self.calc_losses(data, is_train=False, global_step=global_step)
        renderer.train()
        return losses

    def vis_step(self, data, global_step, idx=None):
        """
        可视化步骤
        """
        if "images" not in data:
            return {}
        if idx is None:
            batch_idx = np.random.randint(0, data["images"].shape[0])
        else:
            print(idx)
            batch_idx = idx
        images = data["images"][batch_idx].to(device=device)  # (NV, 3, H, W)
        poses = data["poses"][batch_idx].to(device=device)  # (NV, 4, 4)
        focal = data["focal"][batch_idx: batch_idx + 1]  # (1)
        c = data.get("c")
        if c is not None:
            c = c[batch_idx: batch_idx + 1]  # (1)
        NV, _, H, W = images.shape
        cam_rays = util.gen_rays(
            poses, W, H, focal, self.z_near, self.z_far, c=c
        )  # (NV, H, W, 8)
        images_0to1 = images * 0.5 + 0.5  # (NV, 3, H, W)

        curr_nviews = nviews[torch.randint(0, len(nviews), (1,)).item()]
        views_src = np.sort(np.random.choice(NV, curr_nviews, replace=False))
        view_dest = np.random.randint(0, NV - curr_nviews)
        for vs in range(curr_nviews):
            view_dest += view_dest >= views_src[vs]
        views_src = torch.from_numpy(views_src)

        # set renderer net to eval mode
        renderer.eval()
        source_views = (
            images_0to1[views_src]
            .permute(0, 2, 3, 1)
            .cpu()
            .numpy()
            .reshape(-1, H, W, 3)
        )

        gt = images_0to1[view_dest].permute(1, 2, 0).cpu().numpy().reshape(H, W, 3)
        with torch.no_grad():
            test_rays = cam_rays[view_dest]  # (H, W, 8)
            test_images = images[views_src]  # (NS, 3, H, W)
            net.encode(
                test_images.unsqueeze(0),
                poses[views_src].unsqueeze(0),
                focal.to(device=device),
                c=c.to(device=device) if c is not None else None,
            )
            test_rays = test_rays.reshape(1, H * W, -1)
            render_dict = DotMap(render_par(test_rays, want_weights=True))
            coarse = render_dict.coarse
            fine = render_dict.fine

            using_fine = len(fine) > 0

            alpha_coarse_np = coarse.weights[0].sum(dim=-1).cpu().numpy().reshape(H, W)
            rgb_coarse_np = coarse.rgb[0].cpu().numpy().reshape(H, W, 3)
            depth_coarse_np = coarse.depth[0].cpu().numpy().reshape(H, W)

            if using_fine:
                alpha_fine_np = fine.weights[0].sum(dim=1).cpu().numpy().reshape(H, W)
                depth_fine_np = fine.depth[0].cpu().numpy().reshape(H, W)
                rgb_fine_np = fine.rgb[0].cpu().numpy().reshape(H, W, 3)

        print("c rgb min {} max {}".format(rgb_coarse_np.min(), rgb_coarse_np.max()))
        print(
            "c alpha min {}, max {}".format(
                alpha_coarse_np.min(), alpha_coarse_np.max()
            )
        )
        alpha_coarse_cmap = util.cmap(alpha_coarse_np) / 255
        depth_coarse_cmap = util.cmap(depth_coarse_np) / 255
        vis_list = [
            *source_views,
            gt,
            depth_coarse_cmap,
            rgb_coarse_np,
            alpha_coarse_cmap,
        ]

        vis_coarse = np.hstack(vis_list)
        vis = vis_coarse

        if using_fine:
            print("f rgb min {} max {}".format(rgb_fine_np.min(), rgb_fine_np.max()))
            print(
                "f alpha min {}, max {}".format(
                    alpha_fine_np.min(), alpha_fine_np.max()
                )
            )
            depth_fine_cmap = util.cmap(depth_fine_np) / 255
            alpha_fine_cmap = util.cmap(alpha_fine_np) / 255
            vis_list = [
                *source_views,
                gt,
                depth_fine_cmap,
                rgb_fine_np,
                alpha_fine_cmap,
            ]

            vis_fine = np.hstack(vis_list)
            vis = np.vstack((vis_coarse, vis_fine))
            rgb_psnr = rgb_fine_np
        else:
            rgb_psnr = rgb_coarse_np

        psnr = util.psnr(rgb_psnr, gt)
        vals = {"psnr": psnr}
        print("psnr", psnr)

        # set the renderer network back to train mode
        renderer.train()
        return vis, vals


# ========== 开始训练 ==========
print("\n" + "=" * 80)
print("🎓 Starting training...")
print("=" * 80 + "\n")

trainer = PixelNeRFTrainer()
trainer.start()

import torch
import psutil
import os
import sys
import time

# 添加项目根目录和src目录到路径（和train.py一样）
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(project_root, "src")))

print("=" * 60)
print("测试训练流程")
print("=" * 60)

# 系统信息
mem = psutil.virtual_memory()
print(f"初始内存: {mem.used / 1024 ** 3:.2f} GB ({mem.percent}%)")

if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    torch.cuda.reset_peak_memory_stats(0)

try:
    # 导入必要的模块（和train.py一样）
    from model import make_model, loss
    from render import NeRFRenderer
    from data import get_split_dataset
    import util
    import numpy as np
    import torch.nn.functional as F
    from dotmap import DotMap

    # 1. 解析参数（和train.py一样）
    print("\n[1/8] 解析参数...")


    def extra_args(parser):
        parser.add_argument(
            "--batch_size", "-B", type=int, default=1, help="Object batch size ('SB')"
        )
        parser.add_argument(
            "--nviews",
            "-V",
            type=str,
            default="1",
            help="Number of source views",
        )
        parser.add_argument(
            "--freeze_enc",
            action="store_true",
            default=None,
            help="Freeze encoder weights",
        )
        parser.add_argument(
            "--no_bbox_step",
            type=int,
            default=100000,
            help="Step to stop using bbox sampling",
        )
        return parser


    # 模拟命令行参数
    sys.argv = [
        'test_training.py',
        '--conf', 'conf/exp/srn.conf',
        '--datadir', 'data/srn_cars',
        '--batch_size', '1',
        '--ray_batch_size', '128',
    ]

    args, conf = util.args.parse_args(extra_args, training=True, default_ray_batch_size=128)
    print(f"✅ 配置类型: {type(conf)}")
    print(f"   batch_size: {args.batch_size}")
    print(f"   ray_batch_size: {args.ray_batch_size}")

    device = util.get_cuda(args.gpu_id[0])
    print(f"   device: {device}")

    mem = psutil.virtual_memory()
    print(f"   内存: {mem.used / 1024 ** 3:.2f} GB")

    # 2. 加载数据集
    print("\n[2/8] 加载数据集...")

    dset, val_dset, _ = get_split_dataset(args.dataset_format, args.datadir)
    print(f"✅ 训练集大小: {len(dset)}")
    print(f"   验证集大小: {len(val_dset)}")
    print(f"   z_near: {dset.z_near}, z_far: {dset.z_far}, lindisp: {dset.lindisp}")

    mem = psutil.virtual_memory()
    print(f"   内存: {mem.used / 1024 ** 3:.2f} GB")

    # 3. 创建模型
    print("\n[3/8] 创建模型...")

    net = make_model(conf["model"]).to(device=device)
    net.stop_encoder_grad = args.freeze_enc

    if args.freeze_enc:
        print("   Encoder frozen")
        net.encoder.eval()

    total_params = sum(p.numel() for p in net.parameters())
    trainable_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print(f"✅ 模型类型: {type(net)}")
    print(f"   总参数量: {total_params:,} ({total_params / 1e6:.2f}M)")
    print(f"   可训练参数: {trainable_params:,} ({trainable_params / 1e6:.2f}M)")

    if torch.cuda.is_available():
        gpu_mem = torch.cuda.memory_allocated(0) / 1024 ** 3
        print(f"   GPU显存: {gpu_mem:.2f} GB")

    mem = psutil.virtual_memory()
    print(f"   内存: {mem.used / 1024 ** 3:.2f} GB")

    # 4. 创建渲染器
    print("\n[4/8] 创建渲染器...")

    renderer = NeRFRenderer.from_conf(
        conf["renderer"],
        lindisp=dset.lindisp
    ).to(device=device)

    render_par = renderer.bind_parallel(net, args.gpu_id).eval()

    print(f"✅ 渲染器类型: {type(renderer)}")

    if torch.cuda.is_available():
        gpu_mem = torch.cuda.memory_allocated(0) / 1024 ** 3
        print(f"   GPU显存: {gpu_mem:.2f} GB")

    mem = psutil.virtual_memory()
    print(f"   内存: {mem.used / 1024 ** 3:.2f} GB")

    # 5. 创建损失函数
    print("\n[5/8] 创建损失函数...")

    lambda_coarse = conf.get_float("loss.lambda_coarse")
    lambda_fine = conf.get_float("loss.lambda_fine", 1.0)
    print(f"   lambda_coarse: {lambda_coarse}")
    print(f"   lambda_fine: {lambda_fine}")

    rgb_coarse_crit = loss.get_rgb_loss(conf["loss.rgb"], True)

    fine_loss_conf = conf["loss.rgb"]
    if "rgb_fine" in conf["loss"]:
        print("   使用 fine loss")
        fine_loss_conf = conf["loss.rgb_fine"]
    rgb_fine_crit = loss.get_rgb_loss(fine_loss_conf, False)

    print(f"✅ 损失函数创建完成")

    # 6. 创建优化器
    print("\n[6/8] 创建优化器...")

    optimizer = torch.optim.Adam(net.parameters(), lr=1e-4)
    print(f"✅ 优化器: Adam (lr=1e-4)")

    # 7. 创建数据加载器
    print("\n[7/8] 创建数据加载器...")

    from torch.utils.data import DataLoader

    train_loader = DataLoader(
        dset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=False
    )

    print(f"✅ 数据加载器创建完成")
    print(f"   batch数量: {len(train_loader)}")

    # 8. 测试训练循环
    print("\n[8/8] 测试训练循环（3个batch）...")
    print("如果卡住超过1分钟，按 Ctrl+C 中断\n")

    net.train()
    renderer.train()

    nviews = list(map(int, args.nviews.split()))
    z_near = dset.z_near
    z_far = dset.z_far
    use_bbox = args.no_bbox_step > 0

    for batch_idx, data in enumerate(train_loader):
        if batch_idx >= 3:
            break

        step_start = time.time()
        print(f"\n{'=' * 60}")
        print(f"Batch {batch_idx}")
        print(f"{'=' * 60}")

        # 打印batch信息
        print(f"Batch keys: {list(data.keys())}")
        for k, v in data.items():
            if isinstance(v, torch.Tensor):
                print(f"  {k}: {v.shape} {v.dtype}")

        try:
            # ===== 这部分完全复制 train.py 的 calc_losses 逻辑 =====

            if "images" not in data:
                print("⚠️ batch中没有images，跳过")
                continue

            print("\n[1] 准备数据...")
            prep_start = time.time()

            all_images = data["images"].to(device=device)  # (SB, NV, 3, H, W)
            SB, NV, _, H, W = all_images.shape
            all_poses = data["poses"].to(device=device)  # (SB, NV, 4, 4)
            all_bboxes = data.get("bbox")  # (SB, NV, 4)
            all_focals = data["focal"]  # (SB)
            all_c = data.get("c")  # (SB)

            print(f"   SB={SB}, NV={NV}, H={H}, W={W}")

            if not use_bbox:
                all_bboxes = None

            all_rgb_gt = []
            all_rays = []

            curr_nviews = nviews[torch.randint(0, len(nviews), ()).item()]
            print(f"   curr_nviews: {curr_nviews}")

            if curr_nviews == 1:
                image_ord = torch.randint(0, NV, (SB, 1))
            else:
                image_ord = torch.empty((SB, curr_nviews), dtype=torch.long)

            # 为每个对象生成射线
            for obj_idx in range(SB):
                if all_bboxes is not None:
                    bboxes = all_bboxes[obj_idx]
                else:
                    bboxes = None

                images = all_images[obj_idx]  # (NV, 3, H, W)
                poses = all_poses[obj_idx]  # (NV, 4, 4)
                focal = all_focals[obj_idx]
                c = None
                if all_c is not None:
                    c = all_c[obj_idx]

                if curr_nviews > 1:
                    image_ord[obj_idx] = torch.from_numpy(
                        np.random.choice(NV, curr_nviews, replace=False)
                    )

                images_0to1 = images * 0.5 + 0.5

                # 生成射线
                cam_rays = util.gen_rays(
                    poses, W, H, focal, z_near, z_far, c=c
                )  # (NV, H, W, 8)

                rgb_gt_all = images_0to1
                rgb_gt_all = (
                    rgb_gt_all.permute(0, 2, 3, 1).contiguous().reshape(-1, 3)
                )  # (NV*H*W, 3)

                # 采样像素
                if bboxes is not None:
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
            src_poses = util.batched_index_select_nd(
                all_poses, image_ord
            )  # (SB, NS, 4, 4)

            prep_time = time.time() - prep_start
            print(f"✅ 数据准备完成 ({prep_time:.3f}s)")
            print(f"   src_images: {src_images.shape}")
            print(f"   src_poses: {src_poses.shape}")
            print(f"   all_rays: {all_rays.shape}")
            print(f"   all_rgb_gt: {all_rgb_gt.shape}")

            if torch.cuda.is_available():
                gpu_mem = torch.cuda.memory_allocated(0) / 1024 ** 3
                print(f"   GPU显存: {gpu_mem:.2f} GB")

            # 前向传播
            print("\n[2] 前向传播...")
            forward_start = time.time()

            # 编码
            print("   编码源视图...", end=" ", flush=True)
            net.encode(
                src_images,
                src_poses,
                all_focals.to(device=device),
                c=all_c.to(device=device) if all_c is not None else None,
            )
            print("✅")

            # 渲染
            print("   渲染射线...", end=" ", flush=True)
            render_dict = DotMap(render_par(all_rays, want_weights=True))
            print("✅")

            coarse = render_dict.coarse
            fine = render_dict.fine
            using_fine = len(fine) > 0

            forward_time = time.time() - forward_start
            print(f"   前向传播耗时: {forward_time:.3f}s")
            print(f"   using_fine: {using_fine}")

            if torch.cuda.is_available():
                gpu_mem = torch.cuda.memory_allocated(0) / 1024 ** 3
                print(f"   GPU显存: {gpu_mem:.2f} GB")

            # 计算损失
            print("\n[3] 计算损失...")
            loss_start = time.time()

            rgb_loss = rgb_coarse_crit(coarse.rgb, all_rgb_gt)
            loss_rc = rgb_loss.item() * lambda_coarse

            if using_fine:
                fine_loss = rgb_fine_crit(fine.rgb, all_rgb_gt)
                rgb_loss = rgb_loss * lambda_coarse + fine_loss * lambda_fine
                loss_rf = fine_loss.item() * lambda_fine
                print(f"   loss_rc: {loss_rc:.6f}")
                print(f"   loss_rf: {loss_rf:.6f}")
            else:
                print(f"   loss_rc: {loss_rc:.6f}")

            total_loss = rgb_loss

            # 计算PSNR
            if using_fine:
                rgb_pred = fine.rgb
            else:
                rgb_pred = coarse.rgb

            mse = F.mse_loss(rgb_pred, all_rgb_gt)
            psnr = -10 * torch.log10(mse)

            loss_time = time.time() - loss_start
            print(f"✅ total_loss: {total_loss.item():.6f}, PSNR: {psnr.item():.2f} dB ({loss_time:.3f}s)")

            # 反向传播
            print("\n[4] 反向传播...")
            backward_start = time.time()

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            backward_time = time.time() - backward_start
            print(f"✅ 反向传播完成 ({backward_time:.3f}s)")

            if torch.cuda.is_available():
                gpu_mem = torch.cuda.memory_allocated(0) / 1024 ** 3
                gpu_peak = torch.cuda.max_memory_allocated(0) / 1024 ** 3
                print(f"   GPU显存: {gpu_mem:.2f} GB (峰值: {gpu_peak:.2f} GB)")

            # 调度器步进
            renderer.sched_step(args.batch_size)

        except Exception as e:
            print(f"\n❌ 错误: {e}")
            import traceback

            traceback.print_exc()
            break

        # 总耗时
        step_time = time.time() - step_start
        mem = psutil.virtual_memory()
        print(f"\n总耗时: {step_time:.3f}s")
        print(f"内存: {mem.used / 1024 ** 3:.2f} GB ({mem.percent}%)")

    print("\n" + "=" * 60)
    print("✅ 训练循环测试完成！")
    print("=" * 60)

    # 最终统计
    mem_final = psutil.virtual_memory()
    print(f"\n最终内存: {mem_final.used / 1024 ** 3:.2f} GB ({mem_final.percent}%)")

    if torch.cuda.is_available():
        gpu_final = torch.cuda.memory_allocated(0) / 1024 ** 3
        gpu_peak = torch.cuda.max_memory_allocated(0) / 1024 ** 3
        print(f"最终GPU显存: {gpu_final:.2f} GB (峰值: {gpu_peak:.2f} GB)")

except KeyboardInterrupt:
    print("\n\n⚠️ 用户中断（Ctrl+C）")
    print("程序在某个地方卡住了")
    mem_interrupt = psutil.virtual_memory()
    print(f"中断时内存: {mem_interrupt.used / 1024 ** 3:.2f} GB")
    if torch.cuda.is_available():
        gpu_interrupt = torch.cuda.memory_allocated(0) / 1024 ** 3
        print(f"中断时GPU显存: {gpu_interrupt:.2f} GB")

except Exception as e:
    print(f"\n❌ 错误: {e}")
    import traceback

    traceback.print_exc()

print("\n" + "=" * 60)
print("测试结束")
print("=" * 60)

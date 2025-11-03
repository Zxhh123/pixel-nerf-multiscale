import sys
import os

sys.path.insert(0, 'src')

# 尝试导入配置加载模块
try:
    from pyhocon import ConfigFactory

    # 加载配置
    conf = ConfigFactory.parse_file('conf/exp/sn64_baseline.conf')

    print("=" * 60)
    print("📋 配置文件验证")
    print("=" * 60)

    # 检查 data 配置
    print("\n[Data Config]")
    if hasattr(conf, 'data'):
        data_conf = conf.data
        print(f"  format: {data_conf.get('format', 'NOT SET')}")
        print(f"  num_workers: {data_conf.get('num_workers', 'NOT SET')}")

    # 检查 train 配置
    print("\n[Train Config]")
    if hasattr(conf, 'train'):
        train_conf = conf.train
        print(f"  epochs: {train_conf.get('epochs', 'NOT SET')}")
        print(f"  print_interval: {train_conf.get('print_interval', 'NOT SET')}")
        print(f"  save_interval: {train_conf.get('save_interval', 'NOT SET')}")
        print(f"  vis_interval: {train_conf.get('vis_interval', 'NOT SET')}")
        print(f"  eval_interval: {train_conf.get('eval_interval', 'NOT SET')}")

    # 检查 model 配置
    print("\n[Model Config]")
    if hasattr(conf, 'model') and hasattr(conf.model, 'encoder'):
        encoder_conf = conf.model.encoder
        print(f"  backbone: {encoder_conf.get('backbone', 'NOT SET')}")
        print(f"  use_multi_scale: {encoder_conf.get('use_multi_scale', 'NOT SET')}")
        print(f"  num_layers: {encoder_conf.get('num_layers', 'NOT SET')}")

    print("=" * 60)
    print("✅ 配置文件加载成功！")

except ImportError as e:
    print(f"❌ 导入错误: {e}")
    print("请确保已安装 pyhocon: pip install pyhocon")
except Exception as e:
    print(f"❌ 加载配置文件失败: {e}")

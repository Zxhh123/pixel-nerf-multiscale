@echo off
echo ========================================
echo 配置 pixel-nerf-multiscale 环境
echo ========================================

echo.
echo [1/4] 安装 PyTorch 和 torchvision...
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 --extra-index-url https://download.pytorch.org/whl/cu117

echo.
echo [2/4] 安装配置文件解析库...
pip install pyhocon==0.3.59

echo.
echo [3/4] 安装图像处理库...
pip install imageio==2.31.1 imageio-ffmpeg scikit-image opencv-python

echo.
echo [4/4] 安装其他依赖...
pip install lpips dotmap tqdm tensorboard scipy matplotlib

echo.
echo ========================================
echo 环境配置完成！
echo ========================================

echo.
echo 验证安装...
python -c "import torch; print('PyTorch version:', torch.__version__); print('CUDA available:', torch.cuda.is_available())"
python -c "import pyhocon, imageio, lpips, dotmap; print('All dependencies OK!')"

echo.
echo 按任意键退出...
pause

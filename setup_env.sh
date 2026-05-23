#!/bin/bash

echo "Starting Athena Vision Environment Setup..."

# 1. Bypass Linux RAM-disk limits by forcing pip to use the physical hard drive for temp files
echo "Configuring localized build directories..."
mkdir -p $(pwd)/.tmp_build
export TMPDIR=$(pwd)/.tmp_build

# 2. Upgrade core python packaging tools
pip install --upgrade pip wheel setuptools --no-cache-dir

# 3. Install PyTorch explicitly FIRST. 
# If we don't do this, sub-libraries like basicsr will try to download it 
# in an isolated build environment, causing massive storage duplication.
echo "Installing PyTorch & CUDA Core (This may take a while)..."
pip install torch==2.12.0 torchvision==0.27.0 --no-cache-dir

# 4. Automate the BasicSR GitHub Bug Patch
echo "Patching broken BasicSR library..."
wget -q https://files.pythonhosted.org/packages/source/b/basicsr/basicsr-1.4.2.tar.gz -O basicsr.tar.gz
tar xzf basicsr.tar.gz
sed -i "s/version=get_version()/version='1.4.2'/" basicsr-1.4.2/setup.py
pip install ./basicsr-1.4.2 --no-cache-dir

# 5. Install the remaining lightweight vision and web dependencies
echo "Installing Flask, YOLO, OpenCV, and Utilities..."
pip install Flask==3.0.0 mysql-connector-python==9.0.0 opencv-python==4.9.0.80 ultralytics==8.3.0 numpy==1.26.4 realesrgan==0.3.0 tifffile==2023.12.9 --no-cache-dir

# 6. Deep Clean
echo "Cleaning up temporary build files..."
rm -rf basicsr.tar.gz basicsr-1.4.2 $(pwd)/.tmp_build
pip cache purge

echo "Environment Setup Complete! You can now start worker.py and app.py."

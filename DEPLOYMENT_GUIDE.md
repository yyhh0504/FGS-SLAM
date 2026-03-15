# FGS-SLAM 部署与问题解决指南

> 本文档记录了 FGS-SLAM 项目的所有修改和修复，用于指导后续部署。
> 
> 记录日期：2026-03-15
> 环境：Ubuntu (Headless), CUDA 12.4, Tesla P40

---

## 目录

1. [环境配置](#1-环境配置)
2. [子模块安装](#2-子模块安装)
3. [FastGICP 自定义方法](#3-fastgicp-自定义方法)
4. [高斯光栅化器兼容性修复](#4-高斯光栅化器兼容性修复)
5. [无头环境适配](#5-无头环境适配)
6. [多进程修复](#6-多进程修复)
7. [进度条显示优化](#7-进度条显示优化)
8. [参数保存功能](#8-参数保存功能)
9. [常见问题排查](#9-常见问题排查)

---

## 1. 环境配置

### 1.1 Conda 环境

**问题**：磁盘空间不足（/data 分区 100% 满，73G/73G）

**解决方案**：
- 使用现有的 `torch` 环境（Python 3.10.13）
- 不创建新的 conda 环境

```bash
# 激活已有环境
conda activate torch

# 验证环境
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import torch; print(f'CUDA: {torch.version.cuda}')"
```

### 1.2 系统依赖

```bash
# 基础依赖
sudo apt-get update
sudo apt-get install -y build-essential cmake git wget

# PCL (Point Cloud Library) - python-pcl 需要
sudo apt-get install -y libpcl-dev

# OpenCV
sudo apt-get install -y libopencv-dev
```

---

## 2. 子模块安装

### 2.1 安装顺序

必须按以下顺序安装，因为存在依赖关系：

```bash
cd /data/coding/FGS-SLAM

# 1. diff-gaussian-rasterization
cd submodules/diff-gaussian-rasterization
pip install -e .

# 2. simple_knn
cd ../simple_knn
pip install -e .

# 3. fast_gicp (需要自定义修改，见第3节)
cd ../fast_gicp
pip install -e .

# 4. python-pcl
cd ../python-pcl
python setup.py build_ext --inplace
pip install -e .
```

### 2.2 验证安装

```python
# 测试 diff-gaussian-rasterization
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer

# 测试 simple_knn
from simple_knn._C import distCUDA2

# 测试 fast_gicp
import fast_gicp
reg = fast_gicp.FastGICP()
reg.set_max_knn_distance(1.0)  # 自定义方法

# 测试 python-pcl
import pcl
```

---

## 3. FastGICP 自定义方法

### 3.1 修改的文件

- `submodules/fast_gicp/include/fast_gicp/gicp/fast_gicp.hpp`
- `submodules/fast_gicp/src/fast_gicp/gicp/fast_gicp.cpp`
- `submodules/fast_gicp/src/python/fast_gicp.cpp`

### 3.2 添加的方法

在 `fast_gicp.hpp` 中添加以下声明：

```cpp
// 设置最大KNN搜索距离
void setMaxKNNDistance(double max_dist);

// 设置源/目标点云滤波器
void setSourceFilter(int num_filtered, const std::vector<int>& filter_indices);
void setTargetFilter(int num_filtered, const std::vector<int>& filter_indices);

// 使用滤波器计算协方差
void calculateSourceCovarianceWithFilter();
void calculateTargetCovarianceWithFilter();

// 成员变量
double max_knn_distance_;
std::vector<int> source_filter_indices_;
std::vector<int> target_filter_indices_;
int num_source_filtered_;
int num_target_filtered_;
```

### 3.3 Python 绑定

在 `python/fast_gicp.cpp` 中添加：

```cpp
.def("set_max_knn_distance", &FastGICP::setMaxKNNDistance)
.def("set_source_filter", &FastGICP::setSourceFilter)
.def("set_target_filter", &FastGICP::setTargetFilter)
.def("calculate_source_covariance_with_filter", &FastGICP::calculateSourceCovarianceWithFilter)
.def("calculate_target_covariance_with_filter", &FastGICP::calculateTargetCovarianceWithFilter)
```

---

## 4. 高斯光栅化器兼容性修复

### 4.1 问题描述

不同版本的 `diff_gaussian_rasterization` 返回不同数量的值：
- 旧版本：4个返回值 `(depth_image, rendered_image, radii, is_used)`
- 新版本：2个返回值 `(rendered_image, radii)`

### 4.2 修改的文件

- `gaussian_renderer/__init__.py`

### 4.3 修改内容

**`render()` 函数**（约第90行）：

```python
raster_out = rasterizer(...)

if len(raster_out) == 4:
    depth_image, rendered_image, radii, is_used = raster_out
elif len(raster_out) == 2:
    rendered_image, radii = raster_out
    depth_image = torch.zeros((int(viewpoint_camera.image_height), int(viewpoint_camera.image_width)), device="cuda")
    is_used = torch.ones(means3D.shape[0], dtype=torch.bool, device="cuda")
else:
    raise ValueError(f"Unexpected number of return values from rasterizer: {len(raster_out)}")
```

**同样修改 `render_2()` 和 `render_3()` 函数。**

---

## 5. 无头环境适配

### 5.1 问题描述

服务器没有显示器（Headless），无法使用交互式可视化。

### 5.2 修改的文件

- `mp_Mapper.py`
- `mp_Tracker.py` (如有需要)

### 5.3 修改内容

**1. Matplotlib 后端设置**（文件开头，必须在导入 pyplot 之前）：

```python
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
import matplotlib.pyplot as plt
```

**2. 禁用 cv2.imshow**：

```python
# 原代码
# cv2.imshow('render_result', image_render[..., ::-1])
# cv2.waitKey(1)

# 修改为保存到文件
if self.save_results:
    cv2.imwrite(f"{self.output_path}/render/render_{i}.jpg", image_render[..., ::-1])
```

**3. 禁用 plt.pause**：

```python
# 原代码
# plt.pause(1e-15)

# 修改为保存到文件
plt.savefig(f"{self.output_path}/result_{i}.png")
```

---

## 6. 多进程修复

### 6.1 问题描述

1. **Pickle 序列化错误**：`Tracker` 类无法被 pickle
2. **CUDA 设备可见性**：子进程无法看到 CUDA 设备
3. **端口冲突**：多进程通信端口被占用

### 6.2 修改的文件

- `mp_Tracker.py`
- `mp_Mapper.py`

### 6.3 修复内容

**1. 修复 Tracker 的 __getstate__ 和 __setstate__**：

```python
def __getstate__(self):
    """Custom pickle to handle non-serializable objects"""
    state = self.__dict__.copy()
    # 移除无法序列化的对象
    state['shared_cam'] = None
    state['shared_new_points'] = None
    # ... 其他需要移除的对象
    return state

def __setstate__(self, state):
    """Restore object from pickle"""
    self.__dict__.update(state)
```

**2. 设置 CUDA 设备**：

```python
# 在子进程初始化时设置
def worker_init():
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    torch.cuda.init()
```

**3. 动态端口分配**：

```python
import socket
from contextlib import closing

def find_free_port():
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(('', 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]
```

---

## 7. 进度条显示优化

### 7.1 问题描述

`calc_2d_metric()` 中的 tqdm 进度条被每5轮打印的高斯数量打断，产生多行输出。

### 7.2 修改的文件

- `mp_Mapper.py`

### 7.3 修改内容

**1. `calc_2d_metric()` 进度条**（约第406-408行）：

```python
# 原代码
for i in tqdm(range(len(image_names))):

# 修改为
with torch.no_grad():
    gaussian_num = self.gaussians.get_xyz.shape[0]
    for i in tqdm(range(len(image_names)), 
                  desc=f"Eval 2D metrics (Gaussians: {gaussian_num})", 
                  ncols=100):
```

**2. 注释掉训练阶段的 print 语句**（约第232-233行、252-253行）：

```python
# 原代码
print('gaussian_num:{}'.format(gaussian_num))

# 修改为
# print('gaussian_num:{}'.format(gaussian_num))  # Commented to avoid breaking tqdm display
```

---

## 8. 参数保存功能

### 8.1 添加的方法

在 `mp_Mapper.py` 中添加 `calculate_and_save_parameters()` 方法：

```python
def calculate_and_save_parameters(self):
    """Calculate and save system parameters to JSON file"""
    import json
    import time
    
    params = {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'num_gaussians': self.gaussians.get_xyz.shape[0],
        'fps': self.calc_fps(),
        'ate_rmse': self.calc_ate(),
        'psnr': self.calc_psnr(),
        'ssim': self.calc_ssim(),
        'lpips': self.calc_lpips(),
    }
    
    output_file = os.path.join(self.output_path, 'parameters.json')
    try:
        with open(output_file, 'w') as f:
            json.dump(params, f, indent=2)
        print(f"Parameters saved to {output_file}")
    except OSError as e:
        print(f"Warning: Could not save parameters: {e}")
    
    return params
```

### 8.2 磁盘空间处理

在保存文件时添加异常处理：

```python
try:
    # 保存操作
    cv2.imwrite(...)
except OSError as e:
    if e.errno == 28:  # No space left on device
        print(f"Warning: Disk full, skipping save")
    else:
        raise
```

---

## 9. 常见问题排查

### 9.1 CUDA 内存不足

```python
# 在代码开头设置
import torch
torch.cuda.empty_cache()
torch.cuda.set_per_process_memory_fraction(0.8)  # 限制显存使用
```

### 9.2 磁盘空间不足

```bash
# 检查磁盘空间
df -h

# 清理日志
rm -rf output/*/logs/*.log

# 清理旧的渲染结果
rm -rf output/*/render/*.jpg
```

### 9.3 子模块编译失败

```bash
# 清理并重新编译
cd submodules/<module-name>
rm -rf build dist *.egg-info
pip install -e . --no-deps --force-reinstall
```

### 9.4 Python PCL 安装失败

```bash
# 确保 PCL 已安装
sudo apt-get install libpcl-dev

# 设置 PCL 路径
export PCL_ROOT=/usr
pip install python-pcl --no-binary :all:
```

---

## 10. 快速部署脚本

```bash
#!/bin/bash
# deploy.sh - 快速部署 FGS-SLAM

set -e

echo "=== FGS-SLAM 部署脚本 ==="

# 1. 激活环境
conda activate torch

# 2. 安装子模块
echo "Installing submodules..."
cd submodules/diff-gaussian-rasterization && pip install -e . && cd ../..
cd submodules/simple_knn && pip install -e . && cd ../..
cd submodules/fast_gicp && pip install -e . && cd ../..
cd submodules/python-pcl && python setup.py build_ext --inplace && pip install -e . && cd ../..

# 3. 验证安装
echo "Verifying installation..."
python -c "from diff_gaussian_rasterization import GaussianRasterizer; print('✓ diff_gaussian_rasterization')"
python -c "from simple_knn._C import distCUDA2; print('✓ simple_knn')"
python -c "import fast_gicp; print('✓ fast_gicp')"
python -c "import pcl; print('✓ python-pcl')"

echo "=== 部署完成 ==="
```

---

## 附录：修改汇总

| 文件 | 修改类型 | 修改内容 |
|------|----------|----------|
| `mp_Mapper.py` | 添加 | Matplotlib 'Agg' 后端设置 |
| `mp_Mapper.py` | 修改 | `calc_2d_metric()` 进度条描述 |
| `mp_Mapper.py` | 注释 | 训练阶段 gaussian_num print 语句 |
| `mp_Mapper.py` | 添加 | `calculate_and_save_parameters()` 方法 |
| `mp_Tracker.py` | 添加 | `__getstate__` 和 `__setstate__` 方法 |
| `gaussian_renderer/__init__.py` | 修改 | `render()` 返回值兼容性处理 |
| `gaussian_renderer/__init__.py` | 修改 | `render_2()` 返回值兼容性处理 |
| `gaussian_renderer/__init__.py` | 修改 | `render_3()` 返回值兼容性处理 |
| `submodules/fast_gicp/include/fast_gicp/gicp/fast_gicp.hpp` | 添加 | 自定义 filter 和 KNN 方法声明 |
| `submodules/fast_gicp/src/fast_gicp/gicp/fast_gicp.cpp` | 添加 | 自定义方法实现 |
| `submodules/fast_gicp/src/python/fast_gicp.cpp` | 添加 | Python 绑定 |

---

*文档版本：1.0*
*最后更新：2026-03-15*

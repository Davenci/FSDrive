# UV 环境管理使用说明

本项目已配置使用 `uv` 进行 Python 环境和依赖管理。

## 安装 uv

如果还没有安装 uv，请运行：

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

或者使用 pip 安装：

```bash
pip install uv
```

## 初始化环境

在项目根目录下运行以下命令来同步环境：

```bash
uv sync
```

这个命令会：
- 创建虚拟环境（如果不存在）
- 安装 `pyproject.toml` 中定义的所有依赖
- 生成 `uv.lock` 锁文件

## 安装 LLaMA-Factory

由于 LLaMA-Factory 需要特殊安装，在 `uv sync` 之后还需要：

```bash
cd LLaMA-Factory
uv pip install -e ".[metrics,deepspeed,liger-kernel,bitsandbytes]" --no-build-isolation
cd ..
```

## 常用命令

### 添加新依赖

```bash
uv add package-name
```

### 添加开发依赖

```bash
uv add --dev package-name
```

### 更新依赖

```bash
uv sync --upgrade
```

### 运行 Python 脚本

```bash
uv run python your_script.py
```

### 激活虚拟环境

```bash
source .venv/bin/activate
```

## 注意事项

1. PyTorch 依赖配置为使用 CUDA 12.4 版本
2. 确保系统已安装 CUDA 12.4
3. `uv.lock` 文件应该提交到版本控制系统
4. 首次运行 `uv sync` 可能需要较长时间下载所有依赖

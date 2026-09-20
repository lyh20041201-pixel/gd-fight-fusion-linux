# GD 打架识别：Linux 自动续训交接

此仓库用于让另一台 **Linux GPU 服务器上的 Codex** 下载已封存的训练数据和断点，并自动继续训练。
GitHub 保存代码和 Release 文件；训练在你的 Linux 服务器运行。

当前断点：全画面 RGB 与互动区域 RGB 的 seed42 训练已完成；随机初始化 ST-GCN++ 已完成 8 轮，
从第 9 轮第 **401/2819** 个视频之后继续。后续分支、种子 43/44、融合拟合、校准、测试由队列自动推进。

## 直接交给另一端 Codex 的话

> 请阅读这个仓库的 AGENTS.md 和 README.md，在当前 Linux GPU 服务器上执行自动续训。
> 先检查 NVIDIA 显卡、Python 3.12、磁盘空间和 GitHub 私有仓库访问权限；随后运行
> `python3 tools/bootstrap.py --start --workdir /实际有足够空间的目录/gd-fight-training --gpu 0`。
> 持续检查控制器日志和训练状态，直到确认正式训练日志或进度实际推进后再报告已启动。
> 不修改封存的数据划分、训练超参数、源码、校验哈希或断点，不重训已完成的 RGB 分支。
> 若设备不兼容、依赖缺失、下载或检查失败，请保留日志并报告具体错误。

## 1. Linux 服务器先准备

需要 Linux x86_64、可访问的 NVIDIA GPU、Python 3.12（含 venv）、`git`、GitHub CLI `gh`、`tmux`、`bash`。
本实验固定 PyTorch 2.5.1+cu121；显卡型号未知，所以会在目标机器实际验证 CUDA 和四分支训练。

- 建议至少 32 GiB 系统内存，启动时实际可用内存至少 8 GiB，容器内还会检查 cgroup 内存限制。
- 首次自动下载建议至少 **300 GiB 可用磁盘**；控制器要求 285 GiB，覆盖分卷、合并包、解包、环境与后续缓存空间。
- 正式训练首次需要约 236 GiB 空闲空间，其中缓存最多 128 GiB，保留空间 100 GiB，其余供训练结果使用。
- 不要把工作目录放在容量不足的系统盘或临时容器层；用持久数据盘。
- 无桌面的 Ubuntu/Debian 若缺少 OpenCV 系统库，需要管理员提供 `libgl1`、`libglib2.0-0`。
- 首次安装依赖和下载需要联网；模型训练本身保持原离线策略。

仓库为私有仓库，Linux 端必须登录有读取权限的 GitHub 账号。已有登录可直接使用：

```bash
gh auth status
# 尚未登录时，由账号本人完成浏览器授权：
gh auth login --hostname github.com --git-protocol https --web

gh repo clone lyh20041201-pixel/gd-fight-fusion-linux
cd gd-fight-fusion-linux
```

CLI 安装说明：[官方 GitHub CLI](https://cli.github.com/)。

## 2. 一条命令下载、校验、配置并继续训练

将下面的路径换成服务器上空间足够的持久目录：

```bash
python3 tools/bootstrap.py --start \
  --workdir /data/gd-fight-training \
  --gpu 0
```

它会启动独立的 `tmux` 会话 `gd-fight-train`，依次执行：

1. 检查 Python、GPU、GitHub 登录、磁盘和依赖命令。
2. 从固定 Release 下载训练包分卷。已下载且校验通过的分卷会复用。
3. 校验每个分卷和合并包的 SHA256，安全解包，再校验全部封存文件。
4. 创建独立 `.venv-linux`，安装当前训练所使用的依赖版本。
5. 在独立输出目录用少量真实训练/验证视频运行四分支 GPU 冒烟检查。
6. 通过检查后启动资源监护器和正式训练队列，恢复原子断点，跳过已完成分支。

**控制器启动不等于训练已开始。** 下载、安装和 GPU 检查均需要时间。
只有 `controller-status.json` 进入 `starting_training`、队列进入 `running` 且分支日志/进度实际推进，才算正式开始。
不兼容的显卡或软件版本会停止并报告，不会自动改超参数、升级 PyTorch 或绕过校验。

## 3. 查看进度

```bash
python3 tools/bootstrap.py --status --workdir /data/gd-fight-training
tail -n 80 /data/gd-fight-training/controller.log
tmux attach -t gd-fight-train
```

按 `Ctrl+B` 然后 `D` 离开 tmux，训练会继续。

训练目录是 `/data/gd-fight-training/fight_fusion_linux`，常用文件：

| 文件 | 用途 |
|---|---|
| `controller-status.json`（工作目录下） | 下载、配置、检查、启动或失败阶段 |
| `results/fight_fusion_v1/queue_status.json` | 当前训练/融合/评测阶段 |
| `results/fight_fusion_v1/linux_guard_status.json` | 资源监护及停止原因 |
| `results/fight_fusion_v1/branches/*/seed*/progress.json` | 分支进度 |
| `results/fight_fusion_v1/logs/` | 分支与队列日志 |
| `results/fight_fusion_v1/REPORT.md` | 全部完成后的结果报告 |

## 4. 失败、重连或重启后恢复

先查看 `controller.log` 和资源监护日志，解决明确原因，再运行相同 `--start` 命令。
已有活动 tmux 会话不会重复启动。下载分卷、解包结果和环境会复用，训练恢复最近完整断点。
控制器与监护器**不自动无限重试**；内存不足、校验失败、GPU 不兼容时保留失败信息。
不要删除或重新解压覆盖正在训练的目录，不要删除 `migration/` 或修改签名“解决”错误。
不要用杀死 tmux 会话的方式停止训练；需要停止时让 Codex 按监护器状态中的 PID 向本次监护进程发送 SIGTERM，并确认训练子进程退出。

## 文件结构与来源

- `release-manifest.json`：固定 Release、分卷顺序、大小和 SHA256。
- `tools/bootstrap.py`：另一台服务器的自动执行入口。
- `AGENTS.md`：给接手 Codex 的执行边界、检查和完成判据。
- `snapshot/`：迁移包内的源代码快照，供查看和审核；正式运行使用 Release 解包后的封存代码。
- `docs/LINUX_CONTINUATION.md`：原迁移说明、资源阈值和断点来源记录。

Release 训练包总计 **10.57 GiB**，包含 3964 段视频、必要标注、预训练权重、基线权重、已选模型和训练断点。
不含 `.env`、API 密钥、Windows 虚拟环境或可重建的 127 GiB 特征缓存。
原包整体 SHA256：`19436a2844bbfe9de4fa68ca7fc3e042239cf912f1db9c39dee325532181c029`。
每个 Release 文件小于 2 GiB，符合 [GitHub Release 限制](https://docs.github.com/en/repositories/releasing-projects-on-github/about-releases#storage-and-bandwidth-quotas)。

迁移前后模型张量、优化器、AMP scaler、RNG、采样顺序和游标已核对一致；数据标签、来源划分和时序标注不变。
本机验证通过 78 项相关测试、全包文件哈希及 Linux 文件锁/进程组检查；**目标服务器 GPU 验证由自动流程实际执行**。
跨操作系统、GPU 或解码环境不承诺之后的数值逐位复现。保留“Windows 训练后迁移 Linux 续训”的实验说明。

数据及第三方模型沿用其原有使用条件；本仓库不重新授予这些资源的许可。ST-GCN++ 说明保留在 `snapshot/third_party/`。

# Linux 单卡续训包

这是 Windows 实验的独立迁移副本，继续原来的数据划分、学习率、早停、种子和训练队列。
两路 seed42 RGB 已完成，随机初始化骨架从第 9 轮已保存的第 401/2819 个视频处恢复。
其余分支、三种子融合拟合、校准和测试由原队列顺序完成。
原电脑的数据、配置、模型和续训入口不改动。

## 传输

把生成的 `.tar` 和同名 `.tar.sha256` 通过 SFTP/SCP 上传服务器。
服务器上先校验、解包（文件名替换成实际上传的文件名）：

```bash
sha256sum -c fight_fusion_linux_时间.tar.sha256
tar -xf fight_fusion_linux_时间.tar
cd fight_fusion_linux
```

包内有全部 3964 个封存视频、所需时序标注、预训练权重、基线权重、已选模型和续训断点。
不带可重建的 127 GiB 特征缓存、原始下载 ZIP/分块、`.env` 或 Windows 虚拟环境。
第一次在 Linux 使用窗口时会重新生成特征，最初几轮可能较慢。

## 环境

- Linux x86_64、Python 3.12、NVIDIA GPU 与能够运行 CUDA 12.1 PyTorch 的驱动。
- 安装脚本使用当前实验的 PyTorch 2.5.1+cu121、torchvision 0.20.1+cu121、NumPy 1.26.4、OpenCV 4.9.0.80、Ultralytics 8.2.0。
- 显卡型号未知，因此这些是待目标机器验证的兼容性要求。新的 GPU 可能需要更高版本的 PyTorch/CUDA；检查失败时保留日志，不要直接升级版本或改哈希继续训练。
- 建议至少 32 GiB 系统内存；启动检查实际可用内存至少 8 GiB，包括容器的内存限制。显存是否足够由真实 GPU 冒烟检查验证。
- 解包、安装环境后，首次启动需至少 236 GiB 空闲空间：128 GiB 特征缓存、100 GiB 保留空间、8 GiB 结果余量。为安装依赖及保存上传包留额外空间，约 270 GiB 可用空间较方便。
- 无桌面的 Linux 若 `import cv2` 提示缺少 `libGL.so.1` 等库，需要管理员安装 OpenCV 系统依赖；Debian/Ubuntu 常用 `libgl1` 和 `libglib2.0-0`。Python 3.12 的 venv 支持也需预先可用。

官方对应版本安装命令：<https://pytorch.org/get-started/previous-versions/#v251>。
CUDA 驱动兼容说明：<https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html>。

安装依赖并验证完整传输（只在首次开始训练前执行完整快照校验）：

```bash
bash scripts/linux_fight/setup.sh
```

如果 Python 命令不是 `python3.12`，用 `PYTHON_BIN=/实际路径/python3.12 bash scripts/linux_fight/setup.sh`。
安装需要网络；训练脚本保留离线执行策略。无需上传任何云 API 密钥。

## 验证并开始

指定一个物理 GPU，先做完整预检与四分支真实视频冒烟训练：

```bash
export CUDA_VISIBLE_DEVICES=0
.venv-linux/bin/python scripts/linux_fight/preflight.py --gpu-smoke
```

预检会检查不可变输入哈希、数据封存、当前续训配置及断点签名，并实际执行 CUDA 卷积和反向传播。
冒烟仅用原有训练/分支验证各类别的少量真实视频，独立保存至 `results/linux_gpu_smoke/`；不会使用新测试集，不改正式断点。
只有四分支全部通过才写 `migration/linux_gpu_acceptance.json`。正式启动还会核对机器环境与这些证据。
冒烟日志在对应输出目录的 `global.log`、`roi.log`、`skeleton_random.log`、`skeleton_ntu.log`。

在 `tmux` 会话中启动，避免 SSH 断开中断前台任务：

```bash
tmux new -s fight-train
export CUDA_VISIBLE_DEVICES=0
bash scripts/linux_fight/start.sh
```

按 `Ctrl+B`，再按 `D` 脱离 tmux；用 `tmux attach -t fight-train` 返回。
资源保护停止后，解决原因，再运行同一个 `start.sh`，它会从最近原子断点恢复。
保护器不自动重启；默认每 5 秒采样，可用内存低于 4 GiB 立即停止，连续三次低于 6 GiB 停止；显存连续三次不足 256 MiB 也停止。
保护范围是本次启动的进程组，正常中断会终止它的队列和训练子进程。`SIGKILL`、机器断电及瞬间 OOM 无法保证由用户态监护器捕获。

另开终端查看状态：

```bash
.venv-linux/bin/python scripts/run_fight_fusion.py --status
cat results/fight_fusion_v1/linux_guard_status.json
cat results/fight_fusion_v1/branches/skeleton_random/seed42/progress.json
```

队列日志位于 `results/fight_fusion_v1/logs/linux_queue_*.log`；具体分支日志也在该目录。
全部结束后看 `results/fight_fusion_v1/REPORT.md` 和各 seed 的融合报告。三个种子的选择全部封存后才会评测新测试集。

## 迁移记录与限制

- `migration/original/` 保存原始配置、清单、选择记录以及改动前的训练源代码。
- `migration/record.json` 保存 Windows→相对路径映射、原/新代码和断点哈希、恢复游标及逐项状态相等检查结果。
- 只调整包内预训练权重路径、缓存签名的路径分隔符、Linux 文件锁及创建子进程参数。
- 数据清单仅改路径；标签、来源分组、划分、时序标注等保持一致。检查视频/标注实际 SHA256 后，为副本生成新的来源封存和配置签名，记录与原实验的对应关系。
- `.pt` 的配置元数据及配置签名随副本变化，但模型张量、优化器、AMP scaler、RNG、采样顺序、训练游标和历史逐项比较一致。
- 两路完成的 RGB 只携带选定权重；不携带它们已无须使用的重复 best/resume 文件。骨架断点及本阶段 best 权重完整保留。
- 跨系统、GPU 和视频解码环境不承诺逐位复现之后的数值。结果应标注为“Windows 训练后迁移 Linux 续训”。
- 完整快照校验 `verify.py --full` 只用于首次训练前；续训启动仅检查不可变文件，因为活动断点和训练日志会合理变化。
- 单纯修改已封存训练参数、依赖版本或训练代码会触发拒绝续训，应作为有记录的新实验处理。

本包不自动连接服务器、上传数据或启动远程 GPU；这些操作需要目标主机的实际访问方式。

# Linux 续训交接包

此 Release 提供原封存训练包的 22 个分卷，总计 10.57 GiB。
每个分卷及合并包均有 SHA256 校验；下载器按仓库中的 `release-manifest.json` 自动处理。

仓库与训练包已公开，下载无需 GitHub 账号、Token 或 GitHub CLI。在 Linux 服务器上克隆默认分支，
由 Codex 阅读 `AGENTS.md` 和 `README.md` 后执行续训：

```bash
git clone https://github.com/lyh20041201-pixel/gd-fight-fusion-linux.git
cd gd-fight-fusion-linux
python3 tools/bootstrap.py --start --workdir /data/gd-fight-training --gpu 0
```

流程自动下载、核验、解包、安装固定依赖、进行真实 GPU 冒烟检查，然后启动监护器与正式续训。
已经完成的两路 RGB/seed42 会跳过；骨架从第 9 轮已保存的 401/2819 游标继续。
运行于用户的 Linux GPU 服务器，GitHub 不执行训练。

需要 Linux x86_64、Python 3.12、NVIDIA GPU、tmux 和足够磁盘空间。
此 Release 标签保留最初封存提交；匿名下载入口在默认分支，22 个训练包分卷及其 SHA256 保持不变。
本机 78 项相关测试及全包哈希已验证；目标 Linux GPU 是否兼容由服务器实际检查。
保留原来源、划分、超参数和模型选择规则，跨设备续训不宣称逐位复现。

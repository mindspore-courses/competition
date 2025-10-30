## OpenWebUI 数据迁移

## 快速步骤

> 适用于将本机 Docker 卷 open-webui 迁移到另一台机器。默认镜像使用 alpine 作为临时容器。

1) 变量与临时容器（源机器）

```powershell
$VOLUME = 'open-webui'
$TMP    = 'openwebui-bak'
$TS     = Get-Date -Format yyyyMMdd-HHmmss

docker run -d --name $TMP -v $VOLUME:/data alpine tail -f /dev/null
docker exec $TMP sh -lc "tar -czf /tmp/open-webui-backup-$TS.tar.gz -C /data ."
docker cp $TMP:/tmp/open-webui-backup-$TS.tar.gz .
```

2) 传输备份文件（任选其一）

```powershell
# 示例：本地移动/网络共享/云盘/scp（按你的环境任选一种）
# scp .\open-webui-backup-$TS.tar.gz user@target:/path
```

3) 还原到目标机器的卷

```powershell
$VOLUME = 'open-webui'
$TMP    = 'openwebui-restore'
$BACKUP = 'open-webui-backup-YYYYMMDD-HHMMSS.tar.gz'  # 替换为实际文件名

docker volume create $VOLUME
docker run -d --name $TMP -v $VOLUME:/data alpine tail -f /dev/null
docker cp $BACKUP $TMP:/tmp/
docker exec $TMP sh -lc "tar -xzf /tmp/$BACKUP -C /data"
```

4) 启动 OpenWebUI（示例）

```powershell
# 按你的镜像/版本调整挂载点（常见为 /app/backend/data）
docker run -d --name open-webui -p 3000:8080 -v $VOLUME:/app/backend/data ghcr.io/open-webui/open-webui:latest
```

## 前提

- 源与目标机器已安装并运行 Docker
- 源机器存在卷 open-webui（或替换 $VOLUME）
- 有足够磁盘空间存放压缩包与还原数据

## 验证

```powershell
# 查看备份文件大小（源）
Get-ChildItem .\open-webui-backup-*.tar.gz | Select-Object Name,Length | Sort-Object LastWriteTime -Descending | Select-Object -First 1

# 目标机器：检查卷里是否有文件
docker run --rm -v $VOLUME:/data alpine sh -lc 'ls -lah /data | head -n 50'

# 启动后访问服务（示例本机）：http://localhost:3000
```

## 清理

```powershell
# 源/目标：删除临时容器
docker rm -f openwebui-bak 2>$null
docker rm -f openwebui-restore 2>$null

# 可选：删除本地备份包（谨慎）
# Remove-Item .\open-webui-backup-*.tar.gz

# 可选：Docker 清理
# docker system prune -f
```

## 常见问题

- 卷名不一致：将 $VOLUME 改为实际卷名（用 `docker volume ls` 查看）。
- 备份命令在 PowerShell 下时间戳不生效：已通过 `$TS` 变量并由容器内 `sh -lc` 执行，避免本地 Shell 展开冲突。
- 权限问题：优先使用 `alpine` 临时容器并在容器内打包/解包到 `/tmp` 与 `/data`。
- 镜像挂载路径不同：请按所用 OpenWebUI 镜像文档调整挂载点（常见 `/app/backend/data`）。
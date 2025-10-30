# Dify 完整迁移到 ARM64 机器配置指南

## 第一个机器：数据打包

1. 使用 `tar czvf` 命令打包整个 Dify 目录：

    ```bash
    tar czvf dify-arm64.tar.gz -C /root dify/
    多线程压缩提高速度
    tar --use-compress-program="pigz -p $(nproc)" -cvf dify-arm64.tar.gz -C /root dify/
    ```

    **注意**：绝对不要使用 zip 格式压缩。

    ```shell
    # 解压命令
    tar xzvf dify.tar.gz -C /root/
    ```
2. 将打包文件复制到目标 ARM64 机器并解压。
3. 修改 Nginx 镜像为 ARM64 兼容版本：

    ```yaml
    # 修改 docker-compose.yml 或相关配置文件
    image: arm64v8/nginx:latest
    ```

## 第二个机器：环境清理与配置

### 1. 清除插件数据

在 `dify docker db` 数据库中执行以下 SQL 命令：

```sql
delete from plugin_installations where plugin_unique_identifier is not null;
delete from plugin_declarations where plugin_unique_identifier is not null;
delete from ai_model_installations where plugin_unique_identifier is not null;
delete from plugins where plugin_unique_identifier is not null;
```

### 2. 清除插件目录

删除以下目录的内容：

```bash
rm -rf /app/storage/cwd/*
rm -rf /app/storage/plugins/*
rm -rf /app/storage/cwd/plugin_packages/*
```
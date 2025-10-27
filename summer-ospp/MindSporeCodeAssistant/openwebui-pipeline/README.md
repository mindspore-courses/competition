# OpenWebUI × Dify 接入说明（Pipeline 示例）

本目录包含用于在 OpenWebUI 中通过 Pipeline 脚本调用 Dify 的示例代码。目标是让你在 OpenWebUI 的聊天界面里，直接复用 Dify 的应用或工作流能力（如文档问答、错误检索等）。

## 内容概览

- `异思 2.6.0 文档.py`
- `昇思 2.7.0 文档.py`
- `昇思报错地图.py`

## 适用场景

- 在 OpenWebUI 中对接 Dify 已有的应用（App）或工作流（Workflow）。
- 利用 Dify 的推理/知识库检索/工具调用，将结果回显到 OpenWebUI 的聊天流中。

## 开始之前

- 已安装并可正常启动 OpenWebUI。
- 有可访问的 Dify 实例（云端或自建），并已创建 `API Key`。
- 建议准备以下配置（可使用环境变量或在脚本中配置）：
  - `DIFY_API_KEY`：Dify 的 API Token。
  - `DIFY_BASE_URL`：Dify 服务地址（例如 `https://your-dify-host`，自建请用实际地址）。
  - （可选）`DIFY_APP_ID` 或其他你在脚本中使用到的参数。

## 快速上手

1. 在 OpenWebUI 的设置页中通过 函数 功能导入这些脚本。
2. 在 OpenWebUI 中启用对应的 Pipeline，并根据脚本说明设置必要的环境变量（例如 `DIFY_API_KEY`、`DIFY_BASE_URL`）。
   - Windows 可在命令提示符中使用：`setx DIFY_API_KEY "你的Token"` 和 `setx DIFY_BASE_URL "https://your-dify-host"`

## 工作原理（简述）

- Pipeline 脚本拦截或接收用户消息。
- 脚本通过 Dify 的 REST API（如聊天消息或工作流触发）发送请求，并携带必要的鉴权信息。
- 将 Dify 返回的结果格式化后，回显到 OpenWebUI 的对话流中。

## 常见问题

- 401 或鉴权失败：检查 `DIFY_API_KEY` 是否正确、是否与 `DIFY_BASE_URL` 的实例匹配。
- 无法连接或超时：确认 Dify 实例可访问（内网/公网），以及网络代理设置。
- Pipeline 未生效：确认脚本放置路径、是否已在 OpenWebUI 中启用 Pipeline；必要时重启 OpenWebUI。
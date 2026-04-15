# IC-AI-Chat-Client

基于 **FastAPI** 的聊天演示应用：**Gradio** 为主聊天界面（多主题）。后端统一走 **DeepSeek（OpenAI 兼容 API）** 或 **Ollama**，支持流式多轮对话。

| 能力 | 说明 |
|------|------|
| **一体运行** | 同一进程：`Uvicorn` + Gradio + LLM 封装，浏览器即用。 |
| **Python 集成** | `app.integrations`（`RuntimeConfig`、`stream_chat`、`stream_chat_chunks`、`list_chat_model_names`、`complete_chat`）；或 `app.ui.gradio_chat.mount_gradio_chat_app` 挂载同款 UI。 |
| **推荐集成方式** | 直接使用 `app.integrations`（`RuntimeConfig` + `stream_chat/complete_chat`）对接能力。 |

设计与键空间等说明见 [`tasks/`](tasks/)（如 `project_goal.md`）。

---

## 1. 安装依赖

需要 **Python 3.10+**。

```bash
cd IC-AI-Chat-Client
python -m pip install -r requirements.txt
```

Conda 示例：

```bash
/opt/miniconda3/bin/python -m pip install -r requirements.txt
```

主要依赖：`fastapi`、`uvicorn`、`gradio`、`python-dotenv`、`httpx`、`openai`、`redis`；测试使用 `fakeredis`。Pydantic v2 随 FastAPI 安装。

---

## 2. 配置 `.env`（独立运行必选）

Standalone 模式**仅**加载**仓库根目录**下的 `.env`（不存在则启动失败），不会从上级目录或其它路径回退。

```bash
cp .env.example .env
# 编辑 .env，至少按 LLM_BACKEND 填好密钥或 Ollama 地址
```

### 2.1 LLM 必填项

| `LLM_BACKEND` | 必填环境变量 |
|---------------|--------------|
| `deepseek`（默认） | `DEEPSEEK_API_KEY` |
| `ollama` | `OLLAMA_BASE_URL`、`OLLAMA_GENERATE_MODEL`、`OLLAMA_EMBED_MODEL` |

其它常用项：`DEEPSEEK_LLM_MODEL`、`DEEPSEEK_BASE_URL`、`DEEPSEEK_REQUEST_TIMEOUT`、`OLLAMA_REQUEST_TIMEOUT`、`USER_ID`、`SESSION_ID` 等，见 `.env.example`。

### 2.2 默认拆分口径（验收按 `LLM_TRANSPORT=http`）

口径说明（避免歧义）：

- 代码能力：同时支持 `local` 与 `http`。
- 验收/部署默认：按双进程 HTTP（UI -> LLM Service）。
- 开发回退：`LLM_TRANSPORT=local` 仅用于本地开发和排障，不作为默认验收口径。

| 变量 | 说明 |
|------|------|
| `LLM_TRANSPORT` | 推荐/验收默认 `http`：UI 进程仅通过 HTTP 调用独立 LLM 微服务。`local`：本进程直连 DeepSeek/Ollama（仅开发回退）。 |
| `LLM_SERVICE_URL` | `LLM_TRANSPORT=http` 时**必填**：微服务根 URL，无尾斜杠（如 `http://127.0.0.1:8001`）。 |
| `LLM_SERVICE_TIMEOUT_SECONDS` | 流式读取超时（秒），默认 `120`。 |
| `LLM_SERVICE_API_KEY` | 可选；若设置，UI 进程请求微服务时带 `Authorization: Bearer …`。 |

最小可运行示例（默认拆分）：

```env
# UI process
LLM_TRANSPORT=http
LLM_SERVICE_URL=http://127.0.0.1:8001
```

`scripts/start_uvicorn.sh`（推荐）已内置双进程顺序：先拉起 LLM 微服务（默认 `:8001`），确认端口就绪后再启动 UI（默认 `:8000`）。

手动分开启动时：

- **微服务进程**（仅加载 LLM 相关环境变量，不连 Redis）：

```bash
python -m uvicorn app.llm_service.main:app --host 0.0.0.0 --port 8001
```

- **UI 进程**（`app.main`）：`.env` 中设 `LLM_TRANSPORT=http` 与 `LLM_SERVICE_URL`；启动校验走 `validate_standalone_env` 的 HTTP 分支（不要求本机配置 `DEEPSEEK_API_KEY` / Ollama 变量）。微服务进程仍需通过 `validate_llm_worker_env`（与原先 `LLM_BACKEND` 规则一致）。

流式冒烟（需微服务已启动）：

- CLI 方式：`python tests/test_smoke_llm_http_stream.py --url http://127.0.0.1:8001`
- unittest 方式（默认关闭 live 调用）：`ICAI_RUN_LLM_SMOKE=1 python -m unittest tests.test_smoke_llm_http_stream -v`

常见误区：

- 误区：支持 `http` 等于默认拆分。
- 正确：验收按 `http`，`local` 只做开发回退。

### 2.3 v3.5 消息接入口与转发模式

统一消息协议字段：`message_id/session_id/turn_id/type/content/source/target/timestamp/metadata`。

| 变量 | 说明 |
|------|------|
| `CHAT_UI_INGRESS_PATH` | UI FastAPI 测试入口，默认 `/v1/messages/test`。 |
| `CHAT_UI_FORWARD_URL` | UI 下游转发地址。统一指向 LLM 流式接口（默认 `http://127.0.0.1:8001/v1/chat/stream`）。 |
| `CHAT_UI_FORWARD_TIMEOUT_SECONDS` | UI 转发超时（秒），默认 `30`。 |
| `CHAT_UI_FORWARD_API_KEY` | 可选，UI 转发时附加 `Authorization: Bearer ...`。 |

行为与兼容策略：

- 统一入口：`POST /v1/messages/test`（可通过 `CHAT_UI_INGRESS_PATH` 覆盖）。
- 兼容别名：保留 `POST /v1/messages/in` 与 `POST /v1/messages/receive`，用于历史调用兼容；新接入请使用 `/v1/messages/test`。
- `type=query`：先落库，再按 `CHAT_UI_FORWARD_URL` 转发到下游（默认指向 `/v1/chat/stream`）。
- `type!=query`：仅落库/展示，不触发下游转发。
- 与 LLM 关系：`/v1/chat/stream` 仍是唯一 LLM 对话入口；`/v1/messages/test` 是 UI 侧测试接入层。

v3.6 入口模块状态：

- ingress 路由与服务仅保留新路径：`app/messages/message_ingress_route.py`、`app/messages/message_ingress_service.py`。
- 历史兼容壳 `app/routes/message_ingress.py`、`app/services/message_ingress.py` 已移除。

`CHAT_UI_INGRESS_PATH` 与 `CHAT_UI_FORWARD_URL` 的关系（推荐先看）：

- `CHAT_UI_INGRESS_PATH`：定义“外部消息打到 UI 的入口路径”（例如 `/v1/messages/test`）。
- `CHAT_UI_FORWARD_URL`：定义“UI 在收到 `type=query` 后转发到哪里”。
- 两者职责固定：**入口负责接收，转发地址负责下游调用**；不要把两者混用。
- 推荐默认值：
  - `CHAT_UI_INGRESS_PATH=/v1/messages/test`
  - `CHAT_UI_FORWARD_URL=http://127.0.0.1:8001/v1/chat/stream`

`.env` 配置示例（默认推荐）：

```env
CHAT_UI_INGRESS_PATH=/v1/messages/test
CHAT_UI_FORWARD_URL=http://127.0.0.1:8001/v1/chat/stream
CHAT_UI_FORWARD_TIMEOUT_SECONDS=30
CHAT_UI_FORWARD_API_KEY=
```

`.env` 配置示例（自定义入口路径）：

```env
CHAT_UI_INGRESS_PATH=/v1/messages/custom
CHAT_UI_FORWARD_URL=http://127.0.0.1:8001/v1/chat/stream
CHAT_UI_FORWARD_TIMEOUT_SECONDS=30
CHAT_UI_FORWARD_API_KEY=
```

最小 ingress 请求示例（UI）：

```bash
curl -X POST "http://127.0.0.1:8000/v1/messages/test" \
  -H "Content-Type: application/json" \
  -d '{
    "message_id":"m-1",
    "session_id":"s-1",
    "turn_id":"t-1",
    "type":"query",
    "content":"你好",
    "source":"chat_ui",
    "target":"chat_llm",
    "timestamp":"2026-01-01T00:00:00+00:00",
    "metadata":{"scene":"direct"}
  }'
```

non-query 示例（不会转发，只落库/展示）：

```bash
curl -X POST "http://127.0.0.1:8000/v1/messages/test" \
  -H "Content-Type: application/json" \
  -d '{
    "message_id":"m-2",
    "session_id":"s-1",
    "turn_id":"t-1",
    "type":"plan",
    "content":"先做检索，再回答",
    "source":"chat_ui",
    "target":"chat_llm",
    "timestamp":"2026-01-01T00:00:10+00:00",
    "metadata":{"scene":"non_query_demo"}
  }'
```

自定义入口调用示例（当 `CHAT_UI_INGRESS_PATH=/v1/messages/custom`）：

```bash
curl -X POST "http://127.0.0.1:8000/v1/messages/custom" \
  -H "Content-Type: application/json" \
  -d '{
    "message_id":"m-custom-1",
    "session_id":"s-custom-1",
    "turn_id":"t-custom-1",
    "type":"query",
    "content":"请简要介绍下 RAG",
    "source":"chat_ui",
    "target":"chat_llm",
    "timestamp":"2026-01-01T00:01:00+00:00",
    "metadata":{"scene":"custom_ingress_demo"}
  }'
```

最小自动化复验（验收不通过项修复后）：

```bash
# 自定义测试入口可达性（H1）
CHAT_UI_INGRESS_PATH=/v1/messages/custom python -m unittest tests.test_ui_ingress_api -v

# ingress -> 存储 -> UI可读取证据链（H2）
python -m unittest tests.test_ui_ingress_visibility -v
```

### 2.4 Gradio 界面主题

| 变量 | 说明 |
|------|------|
| `GRADIO_UI_THEME` | 可选；**默认** `business`。仅允许：`business`（商务）、`warm`（温馨）、`minimal`（简约）。 |
| 留空 | 与 `business` 相同。 |
| 非法值 | 若**显式设置了错误取值**，启动时在 `validate_standalone_env` 阶段 **RuntimeError**。 |

代码中推荐用 `mount_gradio_chat_app(..., theme="warm")` 覆盖环境变量。`build_gradio_chat_blocks(...)` 为低层构建接口，仅在你需要自行控制挂载细节时使用。

### 2.5 服务监听

`UVICORN_HOST`、`UVICORN_PORT` 可由 `scripts/start_uvicorn.sh` 读取；直接用 `uvicorn` 命令时也可在命令行指定 `--host` / `--port`。

### 2.6 Redis 与会话

- **开关**：默认 `REDIS_ENABLED=false`（不连 Redis）；启用时必须配置可用 `REDIS_URL`，启动会 `ping` 校验，失败直接退出。
- **关键参数**：`REDIS_KEY_PREFIX`（默认 `icai:`）、`REDIS_SESSION_TTL_SECONDS`（默认 30 天）、`MEMORY_ROUNDS`（默认 3，`0` 表示不截断历史）。
- **会话键格式**：`{prefix}session:{uuid}:meta` 与 `:messages`。
- **模式约束**：`CHAT_MODE=prompt_template` 依赖 Redis（模板见 [`chat_prompt.md`](app/services/chat_prompt.md)）。
- **写回时机（Gradio）**：用户发送先写 `query`；流式成功后写 `answer` 并清理活跃 `turn_id`；失败不写 `answer`。
- **安全边界**：公网场景不要把 `USER_ID` 当多租户安全边界。

本地 Redis 示例：

```bash
docker run -d --name icai-redis -p 6379:6379 redis:7-alpine
```

#### Redis 运维：`redis_manage_ops.py`

用于查看/筛选/清理会话消息（默认读取仓库根 `.env` 的 `REDIS_URL`）。

```bash
# 1) 查看单个会话最近 10 条
python -m app.memory.redis_manage_ops --session-id <session_id> -n 10

# 2) 按用户查看最近 20 条（跨会话聚合）
python -m app.memory.redis_manage_ops --user-id local-dev -n 20

# 3) 仅看某类型（例如 query）
python -m app.memory.redis_manage_ops --user-id local-dev -n 20 --type query

# 4) 清空某会话全部消息
python -m app.memory.redis_manage_ops --session-id <session_id> --clear

# 5) 仅删除某会话最后 N 条 answer
python -m app.memory.redis_manage_ops --session-id <session_id> --clear -n 3 --type answer
```

---

## 3. 部署与准备 Ollama

当 `LLM_BACKEND=ollama` 时需可访问的 **Ollama** 服务（默认 `http://127.0.0.1:11434`）。

1. 从 [ollama.com](https://ollama.com/) 安装并确保 `ollama` 可用。  
2. 拉取模型示例：`ollama pull qwen3:1.7b`、`ollama pull all-minilm:latest`（与 `.env` 中模型名一致）。  
3. 自检：`curl -s http://127.0.0.1:11434/api/tags`  
4. 远程 Ollama：将 `OLLAMA_BASE_URL` 指向可达地址，并放行防火墙。

---

## 4. 运行与访问地址

**前提**：在仓库根目录（或 `PYTHONPATH` 含该根目录）执行，且根目录已有合法 `.env`。

启动或重启服务（推荐，一键双进程）：

```bash
./scripts/start_uvicorn.sh
```

`start_uvicorn.sh` 启动顺序（固定）：

1. 释放 `LLM_SERVICE_PORT`（默认 `8001`）占用并启动 `app.llm_service.main:app`（后台）。
2. 等待 LLM 端口进入 LISTEN。
3. 释放 `UVICORN_PORT`（默认 `8000`）占用并启动 `app.main:app`（前台）。

可选环境变量：

- `UVICORN_HOST`（默认 `0.0.0.0`）
- `UVICORN_PORT`（默认 `8000`）
- `START_LLM_SERVICE`（默认 `1`；设为 `0` 可跳过自动拉起 LLM）
- `LLM_SERVICE_HOST`（默认 `0.0.0.0`）
- `LLM_SERVICE_PORT`（默认 `8001`）
- `LLM_SERVICE_STARTUP_WAIT_SECONDS`（默认 `20`）
- `LLM_SERVICE_LOG_FILE`（默认 `/tmp/icai-llm-service.log`）
- `START_UVICORN_RELEASE_TIMEOUT_SECONDS`（默认 `20`）

风险提示：

- `start_uvicorn.sh` 会尝试结束目标端口上的任意 LISTEN 进程（先 `SIGTERM`，必要时 `SIGKILL`），请仅在本地开发环境使用并确认端口用途。
- 当 `LLM_SERVICE_PORT` 已被其它应用占用时，脚本会优先释放该端口再拉起本项目 LLM 服务，避免 UI 调到错误服务导致 `/v1/chat/stream` 返回 `404`。

直接使用 uvicorn 命令（备选）：

```bash
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

若你选择手动分开启动，推荐顺序：

```bash
# 1) 先起 LLM 微服务
python -m uvicorn app.llm_service.main:app --host 0.0.0.0 --port 8001

# 2) 再起 UI
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

**打开聊天界面**

- 默认监听：`UVICORN_HOST` / `UVICORN_PORT`（常见为 `0.0.0.0:8000`，以 `.env` 为准）。
- 浏览器访问：**根路径**或 **`/gradio`** 均可进入主聊天（根路径会 **302** 到 Gradio）。
  - 本机：`http://127.0.0.1:8000/` 或 `http://127.0.0.1:8000/gradio`
  - 局域网其它设备：`http://<服务器IP>:8000/gradio`（需 `UVICORN_HOST=0.0.0.0` 且防火墙放行端口）

| 地址 | 说明 |
|------|------|
| `/` | **302** 重定向到 Gradio。 |
| `/gradio` | **主聊天界面**（气泡布局）。 |
| `/docs` | FastAPI OpenAPI（若未关闭）。 |

验证 Gradio 气泡交互不建议手写 curl；需要时可使用 **`gradio_client`** 或 Playwright 等做 UI 层冒烟。

### 场景示例

- **DeepSeek**：`LLM_BACKEND=deepseek` + 有效 `DEEPSEEK_API_KEY`，启动后打开 `/gradio`。  
- **远程 Ollama**：`OLLAMA_BASE_URL=http://内网IP:11434` + 上述三个 Ollama 必填项。  
- **换皮肤**：`.env` 中设置 `GRADIO_UI_THEME=warm` 或 `minimal` 后重启。

若 **`Address already in use`**：换端口（如 `8001`）或使用 `./scripts/start_uvicorn.sh` 自动释放端口后重启。
若 `curl http://127.0.0.1:8001/v1/chat/stream` 返回 `404`：先执行 `curl http://127.0.0.1:8001/openapi.json`，确认标题应为 `IC-AI LLM Service`（不是其它应用）。

---

## 5. Gradio 三种主题（简要）

| 主题 | 值 | 视觉意图 |
|------|-----|----------|
| 商务（默认） | `business` | 蓝灰、卡片化顶栏，偏工作台 / 演示。 |
| 温馨 | `warm` | 暖色渐变、大圆角，偏轻社区风格。 |
| 简约 | `minimal` | Gradio 默认主题 + Markdown 顶栏。 |

设计对照图见 `tasks/` 下 `商务风格 *.png`、`温馨风格 *.png`（风格对标，非像素级还原）。

### 5.1 v3.7 客户端布局（实现基线）

- 主标题固定为 `IC-AI-Chat Client`，仅标题条使用深蓝背景（`#0d47a1`）。
- 标题下方为左右分栏：
  - 左侧元信息：`Backend`、`Model`、`User`、`Session ID`（Session ID 位于 User 下方）。
  - 右侧聊天区：可滚动消息列表 + 底部输入框与 `Send`、`Clear conversation` 按钮。
- 页面不展示 `Configuration`、`Dialog` 这类区块标题。
- 静态视觉参考：`demos/client_layout_v3.7_demo.html`。
- 小屏（<=768px）自动堆叠为“元信息在上、聊天在下”。

---

## 6. 在外部项目中集成（Python）

### 6.1 LLM 能力：`app.integrations`

不要依赖本仓库 HTTP 作为稳定契约；在宿主进程内：

```text
RuntimeConfig · validate_runtime_config · normalize_messages
stream_chat · stream_chat_chunks · ChatStreamChunk · complete_chat · list_chat_model_names
```

#### 6.1.1 方式 A：在 Python 代码中直接调用（推荐）

- **`stream_chat`**：仅拼接助手可见正文（`content_delta`），与 Gradio/SSE 行为一致。  
- **`stream_chat_chunks`**：结构化块（含可选 `reasoning_delta`、结束标记 `done`）。  
- **`list_chat_model_names`**：Ollama 使用 `GET /api/tags`；DeepSeek 无列举 API 时返回当前配置的单个模型 id。

传入 `runtime=RuntimeConfig(...)` 时**仅**使用该配置调用 LLM，**不会** DeepSeek 失败后自动切 Ollama。

**示例：流式（DeepSeek）**

```python
from app.integrations import RuntimeConfig, normalize_messages, stream_chat

cfg = RuntimeConfig(
    llm_backend="deepseek",
    deepseek_api_key="sk-你的密钥",
    deepseek_llm_model="deepseek-chat",
)
messages = normalize_messages([{"role": "user", "content": "用一句话介绍 Python。"}])
for delta in stream_chat(messages, runtime=cfg):
    print(delta, end="", flush=True)
print()
```

**示例：非流式（Ollama）**

```python
from app.integrations import RuntimeConfig, normalize_messages, complete_chat

cfg = RuntimeConfig(
    llm_backend="ollama",
    ollama_base_url="http://127.0.0.1:11434",
    ollama_generate_model="qwen3:1.7b",
    ollama_embed_model="all-minilm:latest",
)
messages = normalize_messages(
    [
        {"role": "system", "content": "你是一个简洁助手。"},
        {"role": "user", "content": "1+1等于几？"},
    ]
)
print(complete_chat(messages, runtime=cfg))
```

#### 6.1.2 方式 B：在宿主 FastAPI 中调用 chat LLM（SSE 路由示例）

当你在外部项目（如 `ic-rag-agent`）里希望通过一个 FastAPI 路由对外暴露 chat LLM，可直接调用 `app.integrations.stream_chat`，并将增量写成 SSE。

```python
from __future__ import annotations

import json
from typing import Generator

from fastapi import FastAPI
from fastapi.responses import StreamingResponse

from app.integrations import RuntimeConfig, normalize_messages, stream_chat

app = FastAPI()


def _sse(data: dict) -> str:
    return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"


@app.post("/v1/chat/stream")
def chat_stream():
    runtime = RuntimeConfig(
        llm_backend="deepseek",
        deepseek_api_key="sk-你的密钥",
        deepseek_llm_model="deepseek-chat",
    )
    messages = normalize_messages([{"role": "user", "content": "你好，请介绍一下你自己。"}])

    def event_generator() -> Generator[str, None, None]:
        try:
            for delta in stream_chat(messages, runtime=runtime):
                yield _sse({"delta": delta})
            yield _sse({"done": True})
        except Exception as exc:  # noqa: BLE001
            yield _sse({"error": str(exc)})

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
```

使用建议：

- 路由契约建议保持与本项目一致：`POST /v1/chat/stream` + SSE `delta/done/error`。
- 如果作为 `chat UI` 下游，确保 `LLM_TRANSPORT=http` 且 `LLM_SERVICE_URL` 指向该服务。
- 生产环境请把密钥放环境变量或密钥管理系统，不要写死在代码中。

### 6.2 同款 Gradio UI（推荐）：`mount_gradio_chat_app`

从 `app.ui.gradio_chat` 导入；须先 `load_dotenv` 并调用 `validate_standalone_env()`（或与主应用一致的 env），再 `mount_gradio_chat_app(app, path="/gradio")`。该 Facade 会在挂载时注入主题与 CSS，兼容 Gradio 6 参数迁移。

**易错点**

1. **`path=` 与浏览器 URL 一致**：写成 `path="/gradio"` 就访问 `http://host:port/gradio`，不要混用 `/chat`。  
2. **必须 `uvicorn.run(app, ...)`**（或命令行 uvicorn）；仅创建 `app` 不会监听端口。  
3. **IPython**：不要依赖 `if __name__ == "__main__"`；在单元末尾**单独执行** `uvicorn.run(...)`。

**最小示例（项目根目录）**

```python
from pathlib import Path

from dotenv import load_dotenv
import uvicorn
from fastapi import FastAPI

load_dotenv(Path.cwd() / ".env")  # IPython 建议改为 .env 的绝对路径
from app.config import validate_standalone_env

validate_standalone_env()

from app.ui.gradio_chat import mount_gradio_chat_app

app = FastAPI()
mount_gradio_chat_app(app, path="/gradio")

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
```

低层接口说明：

- `build_gradio_chat_blocks(...)`：仅返回 `gr.Blocks`，**不负责**在 Gradio 6 场景下注入最终挂载样式参数。
- 除非你有明确的自定义挂载需求，否则请优先使用 `mount_gradio_chat_app(...)`。

**可选参数（v3）**

- `theme="warm"` / `"minimal"`：覆盖 `GRADIO_UI_THEME`。  
- `app_config=AppConfig(...)`：顶栏与校验用配置（不传则用 `get_config()`）。  
- `runtime=RuntimeConfig(...)`：对话走内存配置，等价于 `stream_chat(..., runtime=...)`。

### 6.3 自建 SSE（思路）

在宿主路由中组装 `messages`，循环 `stream_chat(..., runtime=cfg)`，将每个 `delta` 写成 SSE `data:` 帧即可；密钥勿写进仓库。

### 6.4 Chat UI 对接下游应用（如 ic-rag-agent）

本节给出“仅迁移 Chat UI、下游已有 route/dispatcher/rag 服务”的最小对接教程。

#### 6.4.1 对接目标与数据流

- Gradio 主输入链路：`chat UI -> route llm (/v1/chat/stream)`。
- 外部消息接入链路：`external/dispatcher -> CHAT_UI_INGRESS_PATH`。
- ingress 规则：`type=query` 会继续转发到 `CHAT_UI_FORWARD_URL`；非 query 只落库/展示。

#### 6.4.2 下游（route llm）必须提供的接口契约

`POST /v1/chat/stream`，请求体兼容：

```json
{
  "messages": [{"role": "user", "content": "hello"}],
  "backend": "optional",
  "model": "optional"
}
```

返回必须是 SSE 帧：

- `data: {"delta":"..."}`（可多次）
- `data: {"done": true}`（结束）
- `data: {"error":"..."}`（异常）

#### 6.4.3 Chat UI 侧 `.env` 最小配置

```env
# 1) Gradio 主输入走下游 route llm
LLM_TRANSPORT=http
LLM_SERVICE_URL=http://<route-llm-host>:<port>

# 2) UI 测试接入口（给 dispatcher / 外部系统投递消息）
CHAT_UI_INGRESS_PATH=/v1/messages/test

# 3) ingress 收到 query 后转发到下游
CHAT_UI_FORWARD_URL=http://<route-llm-host>:<port>/v1/chat/stream
CHAT_UI_FORWARD_TIMEOUT_SECONDS=30
CHAT_UI_FORWARD_API_KEY=
```

可选：如果只想启动 UI，不自动拉起本仓库 LLM：

```bash
START_LLM_SERVICE=0 ./scripts/start_uvicorn.sh
```

#### 6.4.4 联调步骤（推荐顺序）

1. 先测下游 route llm 的 `/v1/chat/stream`（确认 SSE 正常）。
2. 启动 Chat UI，打开 `/gradio` 用输入框直测一条 query。
3. 再测 UI ingress：
   - `POST /v1/messages/test` 发送 `type=query`，应返回 `forwarded=true`；
   - `type=clarification/plan/...`，应返回 `forwarded=false` 且可在会话历史展示。

#### 6.4.5 常见坑位

- **会话不一致看不到消息**：外部投递到 ingress 的 `session_id` 必须和当前页面会话一致。
- **只启了 UI 没启下游**：会出现 `Connection refused`，先确认 `LLM_SERVICE_URL` 可达。
- **误把 ingress 当主输入链路**：用户在 Gradio 输入框默认走 `LLM_SERVICE_URL + /v1/chat/stream`，不是走 `CHAT_UI_INGRESS_PATH`。

---

## 7. 常见问题

| 现象 | 处理 |
|------|------|
| 缺少 `.env` | 从 `.env.example` 复制到仓库根目录。 |
| `Standalone .env is missing required variables` | 按 `LLM_BACKEND` 补全 §2.1 表内变量（`LLM_TRANSPORT=http` 时见 §2.2，不要求本机 LLM 密钥）。 |
| `LLM_TRANSPORT=http requires …` | 设置 `LLM_SERVICE_URL`；并先启动 `app.llm_service.main:app`。 |
| `GRADIO_UI_THEME` 报错 | 仅允许 `business`、`warm`、`minimal`（大小写不敏感）。 |
| `REDIS_ENABLED is true but REDIS_URL is missing` | 开启 Redis 时必须填写 `REDIS_URL`。 |
| 启动报 Redis / Connection refused | 确认 Redis 已启动且 `REDIS_URL` 可达。 |
| Ollama 连不上 | 检查 `OLLAMA_BASE_URL`、服务是否运行、`ollama pull` 与网络。 |
| Gradio **404** | 访问路径须与 `mount_gradio_app(..., path=...)` 一致。 |
| 端口被占用 | 换 `UVICORN_PORT` 或 `kill` 占用该端口的进程。 |

---

## 8. 生产部署与安全

- **密钥与模型 Key**：`DEEPSEEK_API_KEY` 等**仅**存在于服务端环境变量或密钥管理系统；**不要**下发到浏览器或打包进前端。本应用由 FastAPI **代为请求** DeepSeek/Ollama，浏览器只访问同源 ASGI。
- **访问控制**：公网部署时应在 **反向代理 / API 网关** 上配置认证、限流与 TLS；勿依赖 `.env` 里的 `USER_ID` 作为多租户安全边界（见 `project_goal.md` §2.4）。
- **多进程（示例）**：在 Linux 上可用 Gunicorn + Uvicorn worker（需 `pip install gunicorn`），例如：
  ```bash
  gunicorn app.main:app -k uvicorn.workers.UvicornWorker -w 2 -b 0.0.0.0:8000
  ```
  开发阶段仍推荐 `uvicorn app.main:app --reload`。Worker 数与超时需按 LLM 流式耗时调整。
- **健康检查**：可对外暴露 FastAPI 自带 `GET /docs` 或自建 `/health`（按需添加路由）。
- **单元测试**：在项目根执行 `python -m unittest discover -s tests -p 'test_*.py'`（含 `test_llm_transport`、`test_turn_lifecycle`、`test_main_routes`、`test_session_store`、`test_message_ingress`、`test_ui_ingress_api`、`test_config_v35` 等；fakeredis，无 live LLM）。
- **UI 自动化 / E2E（建议）**：当前以 `unittest` 与路由/服务层为主；Gradio 布局与 CSS 若需防回归，可后续引入浏览器 E2E（例如 Playwright）或截图/快照对比，以降低 Gradio 大版本升级带来的无感退化风险。

---

## 9. 文件组织结构

```text
app/
  main.py                # FastAPI 入口、lifespan Redis、SessionMiddleware（M3）、挂载 Gradio
  config.py              # AppConfig、RedisSettings、validate_standalone_env、get_gradio_ui_theme
  integrations.py        # 对外 LLM 稳定导出（stream_chat/RuntimeConfig/list_chat_model_names）
  runtime_config.py      # RuntimeConfig（库集成）
  messages/
    message_envelope.py         # 统一消息协议（message envelope）
    message_ingress_route.py    # v3.6：UI 测试接入口（/v1/messages/test，保留旧路径别名）
    message_ingress_service.py  # v3.6：UI ingress 处理、转发与落库
  memory/
    redis_pool.py             # Redis 连接池
    session_store.py          # 会话读写、消息追加、历史读取
    redis_runtime.py          # 运行时缓存/注入 helpers
    redis_manage_ops.py       # 本地运维脚本入口（会话查询/清理）
  llm_service/
    main.py                  # LLM Worker：POST /v1/chat/stream（SSE，唯一对话入口）
  ui/
    gradio_chat.py           # Facade：统一构建与挂载 Gradio UI
    gradio_layout.py         # 页面结构与组件装配
    chat_history_normalize.py  # Chatbot 消息行归一化（layout / handlers 共用）
    gradio_handlers.py       # 输入/流式回调/异常处理
    gradio_persistence.py    # session 与 Redis 持久化
    gradio_session_turn.py   # Starlette 会话中的活跃 turn_id
    gradio_themes.py         # business / warm / minimal 主题
    message_model.py         # Gradio 按 type 格式化会话消息
  routes/
    chat_pages.py            # 根路径重定向到 /gradio
  services/
    call_llm.py              # 本地 LLM 能力（stream_chat / stream_chat_chunks）
    llm_transport.py         # UI/SSE 统一接口（校验、local/http流式、stage消息回调）
    call_deepseek.py         # DeepSeek 客户端
    call_ollama.py           # Ollama 客户端
    llm_chunks.py            # ChatStreamChunk
    llm_models.py            # list_chat_model_names
    chat_prompt.md           # CHAT_MODE=prompt_template：{historical_message}、{current_query}
    prompt_render.py         # 读模板、按轮拼 Markdown 历史
scripts/
  start_uvicorn.sh           # 开发启动（先LLM后UI，自动释放端口并启动）
```

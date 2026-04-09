# IC-AI-Chat-Client

基于 **FastAPI** 的聊天演示应用：**Gradio** 为主聊天界面（多主题）。后端统一走 **DeepSeek（OpenAI 兼容 API）** 或 **Ollama**，支持流式多轮对话。

| 能力 | 说明 |
|------|------|
| **一体运行** | 同一进程：`Uvicorn` + Gradio + LLM 封装，浏览器即用。 |
| **Python 集成** | `app.integrations`（`RuntimeConfig`、`stream_chat`、`stream_chat_chunks`、`list_chat_model_names`、`complete_chat`）；或 `app.ui.gradio_chat.build_gradio_chat_blocks` 挂载同款 UI。 |
| **不推荐** | 把 `POST /api/chat/stream` 当作对外稳定公共 API。 |

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

主要依赖：`fastapi`、`uvicorn`、`gradio`、`jinja2`、`python-dotenv`、`httpx`、`openai`、`redis`；测试使用 `fakeredis`。Pydantic v2 随 FastAPI 安装。

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

**微服务进程**（仅加载 LLM 相关环境变量，不连 Redis）：

```bash
python -m uvicorn app.llm_service.main:app --host 0.0.0.0 --port 8001
```

**UI 进程**（`app.main`）：`.env` 中设 `LLM_TRANSPORT=http` 与 `LLM_SERVICE_URL`；启动校验走 `validate_standalone_env` 的 HTTP 分支（不要求本机配置 `DEEPSEEK_API_KEY` / Ollama 变量）。微服务进程仍需通过 `validate_llm_worker_env`（与原先 `LLM_BACKEND` 规则一致）。

流式冒烟（需微服务已启动）：`python scripts/smoke_llm_http_stream.py --url http://127.0.0.1:8001`

常见误区：

- 误区：支持 `http` 等于默认拆分。
- 正确：验收按 `http`，`local` 只做开发回退。

### 2.3 Gradio 界面主题

| 变量 | 说明 |
|------|------|
| `GRADIO_UI_THEME` | 可选；**默认** `business`。仅允许：`business`（商务）、`warm`（温馨）、`minimal`（简约）。 |
| 留空 | 与 `business` 相同。 |
| 非法值 | 若**显式设置了错误取值**，启动时在 `validate_standalone_env` 阶段 **RuntimeError**。 |

代码中可用 `build_gradio_chat_blocks(theme="warm")` **覆盖**环境变量。

### 2.4 服务监听

`UVICORN_HOST`、`UVICORN_PORT` 由 `scripts/run.py` 读取；直接用 `uvicorn` 命令时也可在命令行指定 `--host` / `--port`。

### 2.5 Redis 与会话

- **默认**：`REDIS_ENABLED=false`。不连 Redis，不注册 `POST /api/sessions` 等会话路由。
- **开启 Redis**：`REDIS_ENABLED=true` 且配置可用 `REDIS_URL`。启动时会 `ping`；失败则进程直接报错退出。
- **键前缀**：`REDIS_KEY_PREFIX`（默认 `icai:`），会话键形如 `{prefix}session:{uuid}:meta` / `:messages`（细节见 `project_goal.md` §2.3）。
- **TTL**：`REDIS_SESSION_TTL_SECONDS`（默认 30 天）；有写入时会续期。
- **记忆窗口**：`MEMORY_ROUNDS`（默认 `3`）。一轮在 Redis 里通常对应同一 `turn_id` 下的一组消息（可含多条 `query` 澄清）；无 `turn_id` 的旧数据仍按「新 `query` 起一轮」切分。`0` 表示不按轮截断、展示全部已存消息。
- **对话模式**：`CHAT_MODE=messages`（默认），多轮 `messages` 调 LLM。`prompt_template` 需 `REDIS_ENABLED=true`，模板见 [`chat_prompt.md`](app/services/chat_prompt.md)（`{historical_message}`、`{current_query}`）。
- **Gradio + Redis**：需在 `.env` 配置足够随机的 **`SECRET_KEY`**（签名 Cookie，刷新后恢复 `icai_gradio_session_id`）。
- **单条消息 JSON**（`messages` 列表里每项）必填：`user_id`、`session_id`、`type`、`content`、`timestamp`、`turn_id`。常见 `type`：`query`、`answer`、`clarification`、`rewriting`、`classification`、`reason`、`plan`、`context`、`dispatcher`。
- **旧数据**：不支持仅 `role`/`ts` 的旧行。库里有遗留数据时，请先清空对应 key 或离线迁移再升级。
- **可选类型是否在气泡里显示**：7 个 `*_MESSAGE_DISPLAY_ENABLE`（见 `.env.example`）。`query`/`answer` 始终显示。开关只影响 Gradio；`prompt_template` 拼进模型的历史仍含全部类型。
- **会话 API**（内置调试用）：`POST /api/sessions` 创建；`GET /api/sessions/{session_id}/messages` 拉取；`POST /api/sessions/{session_id}/messages` 按条追加（Body：`type`、`content`、`turn_id`，与 `USER_ID` 校验）。越权 403，无会话 404。
- **写回（Gradio + Redis）**：用户发送即写入 `query`（Starlette 会话中保留活跃 `turn_id`）；流式**成功**后写入 `answer` 并清除该 `turn_id`；**失败**不写 `answer`。Legacy 页面 `POST /api/chat/stream` 在整轮成功后按同一 `turn_id` 依次写入 `query` 与 `answer`。
- **`app.integrations`**：始终多轮 `stream_chat(messages=...)`，不受 `prompt_template` 影响。
- **安全**：`session_id` 可能泄露；当前只比对 meta 与 `.env` 的 `USER_ID`。公网勿把 `USER_ID` 当多租户边界（见 `project_goal.md` §2.4、§5）。

本地 Redis 示例：

```bash
docker run -d --name icai-redis -p 6379:6379 redis:7-alpine
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

启动或重启服务：

```bash
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
| `/legacy` | 旧版 Jinja + 前端 SSE 聊天页（仍使用 `POST /api/chat/stream`）。 |

`REDIS_ENABLED=true` 时，OpenAPI 中还可看到 **`POST /api/sessions`**、**`GET/POST /api/sessions/{session_id}/messages`**（与内置页 / 调试配合使用）。验证 Gradio 气泡交互不建议手写 curl；需要时可使用 **`gradio_client`** 或 Playwright 等做 UI 层冒烟。

### 场景示例

- **DeepSeek**：`LLM_BACKEND=deepseek` + 有效 `DEEPSEEK_API_KEY`，启动后打开 `/gradio`。  
- **远程 Ollama**：`OLLAMA_BASE_URL=http://内网IP:11434` + 上述三个 Ollama 必填项。  
- **换皮肤**：`.env` 中设置 `GRADIO_UI_THEME=warm` 或 `minimal` 后重启。

若 **`Address already in use`**：换端口（如 `8001`）或结束占用该端口的旧 `uvicorn` 进程。

---

## 5. Gradio 三种主题（简要）

| 主题 | 值 | 视觉意图 |
|------|-----|----------|
| 商务（默认） | `business` | 蓝灰、卡片化顶栏，偏工作台 / 演示。 |
| 温馨 | `warm` | 暖色渐变、大圆角，偏轻社区风格。 |
| 简约 | `minimal` | Gradio 默认主题 + Markdown 顶栏。 |

设计对照图见 `tasks/` 下 `商务风格 *.png`、`温馨风格 *.png`（风格对标，非像素级还原）。

---

## 6. 在外部项目中集成（Python）

### 6.1 LLM 能力：`app.integrations`

不要依赖本仓库 HTTP 作为稳定契约；在宿主进程内：

```text
RuntimeConfig · validate_runtime_config · normalize_messages
stream_chat · stream_chat_chunks · ChatStreamChunk · complete_chat · list_chat_model_names
```

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

### 6.2 同款 Gradio UI：`build_gradio_chat_blocks`

从 `app.ui.gradio_chat` 导入；须先 `load_dotenv` 并调用 `validate_standalone_env()`（或与主应用一致的 env），再 `gr.mount_gradio_app(app, build_gradio_chat_blocks(), path="/gradio")`。

**易错点**

1. **`path=` 与浏览器 URL 一致**：写成 `path="/gradio"` 就访问 `http://host:port/gradio`，不要混用 `/chat`。  
2. **必须 `uvicorn.run(app, ...)`**（或命令行 uvicorn）；仅创建 `app` 不会监听端口。  
3. **IPython**：不要依赖 `if __name__ == "__main__"`；在单元末尾**单独执行** `uvicorn.run(...)`。

**最小示例（项目根目录）**

```python
from pathlib import Path

from dotenv import load_dotenv
import gradio as gr
import uvicorn
from fastapi import FastAPI

load_dotenv(Path.cwd() / ".env")  # IPython 建议改为 .env 的绝对路径
from app.config import validate_standalone_env

validate_standalone_env()

from app.ui.gradio_chat import build_gradio_chat_blocks

app = FastAPI()
gr.mount_gradio_app(app, build_gradio_chat_blocks(), path="/gradio")

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
```

**可选参数（v3）**

- `theme="warm"` / `"minimal"`：覆盖 `GRADIO_UI_THEME`。  
- `app_config=AppConfig(...)`：顶栏与校验用配置（不传则用 `get_config()`）。  
- `runtime=RuntimeConfig(...)`：对话走内存配置，等价于 `stream_chat(..., runtime=...)`。

### 6.3 自建 SSE（思路）

在宿主路由中组装 `messages`，循环 `stream_chat(..., runtime=cfg)`，将每个 `delta` 写成 SSE `data:` 帧即可；密钥勿写进仓库。

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
- **单元测试**：在项目根执行 `python -m unittest discover -s tests -p 'test_*.py'`（含 `test_llm_transport`、`test_turn_lifecycle`、`test_sessions_api`、`test_session_store` 等；fakeredis，无 live LLM）。

---

## 9. 文件组织结构

```text
app/
  main.py              # FastAPI 入口、lifespan Redis、SessionMiddleware（M3）、挂载 Gradio / 静态资源 / 路由
  config.py            # AppConfig、RedisSettings、validate_standalone_env、get_gradio_ui_theme
  deps.py              # require_session_store（M3）
  integrations.py      # 对外 LLM 稳定导出
  runtime_config.py    # RuntimeConfig（库集成）
  memory/              # redis_pool、session_store、redis_runtime（Gradio 绑定 Redis）
  llm_service/
    main.py            # LLM Worker：POST /v1/chat/stream（SSE，默认拆分口径）
  ui/
    gradio_chat.py     # Gradio 主链路：query即时落库、stage消息回调落库、answer收尾
    gradio_session_turn.py  # Starlette 会话中的活跃 turn_id
    gradio_themes.py   # business / warm / minimal 主题
    message_model.py   # Gradio 按 type 格式化会话消息
  routes/              # chat_pages、chat_stream（SSE）、sessions（Redis 开启时）
  services/
    call_llm.py        # 本地 LLM 能力（stream_chat / stream_chat_chunks）
    llm_transport.py   # UI/SSE 统一接口（校验、local/http流式、stage消息回调）
    call_deepseek.py   # DeepSeek 客户端
    call_ollama.py     # Ollama 客户端
    llm_chunks.py      # ChatStreamChunk（M2）
    llm_models.py      # list_chat_model_names（M2）
    chat_prompt.md     # CHAT_MODE=prompt_template：{historical_message}、{current_query}
    prompt_render.py   # 读模板、按轮拼 Markdown 历史
scripts/
  run.py               # 开发启动（reload）
  smoke_llm_http_stream.py  # 对 LLM 微服务 SSE 的冒烟脚本
tests/                 # unittest（含阶段落库、chat_stream持久化、llm_service路由级流式、依赖边界等）
  test_gradio_stage_persist.py     # P0：stage消息实时写Redis与同turn_id
  test_chat_stream_persist.py      # P0：chat_stream主路径append_memory_message
  test_llm_service_stream_api.py   # P1：/v1/chat/stream 真实端点SSE测试
  test_gradio_chat_dependencies.py # P2：UI不直接依赖call_llm细节
tasks/                 # 设计文档、里程碑、参考图（含 m3_plan_v3.2*.md）
```

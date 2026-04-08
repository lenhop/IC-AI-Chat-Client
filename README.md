# IC-AI-Chat-Client

基于 **FastAPI** 的聊天演示应用：**Gradio** 为主聊天界面（多主题），**Jinja2** 为可选旧版页面。后端统一走 **DeepSeek（OpenAI 兼容 API）** 或 **Ollama**，支持流式多轮对话。

| 能力 | 说明 |
|------|------|
| **一体运行** | 同一进程：`Uvicorn` + Gradio + LLM 封装，浏览器即用。 |
| **Python 集成** | `app.integrations`（`RuntimeConfig`、`stream_chat`、`stream_chat_chunks`、`list_chat_model_names`、`complete_chat`）；或 `app.ui.gradio_chat.build_gradio_chat_blocks` 挂载同款 UI。 |
| **不推荐** | 把 `POST /api/chat/stream` 当作对外公共 API（仅供内置页 / 调试）。 |

**文档与计划**：`tasks/project_goal.md`（总目标与里程碑）· `tasks/m1_plan.md` · `tasks/m1_plan_v2.md` · `tasks/m1_plan_v3.md` · **`tasks/m2_plan.md`（M2：chunk 抽象、模型列表、无 Redis）** · **`tasks/m3_plan.md`（M3：Redis 会话与刷新恢复）**。

**里程碑（摘要）**：M1 已交付 **FastAPI + Gradio/Jinja + DeepSeek/Ollama**；M2 为 **`ChatStreamChunk` / `stream_chat_chunks`**、**`list_chat_model_names`**（Ollama 拉 tags，DeepSeek 返回当前配置模型）及生产说明；**M3** 为 **可选 Redis 会话**（`REDIS_ENABLED`、键前缀与 TTL、内置会话 API、Gradio 签名 Cookie + `/legacy` sessionStorage）；**M4** Route/Dispatcher 时间线（见 `project_goal.md` §4）。

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

主要依赖：`fastapi`、`uvicorn`、`gradio`、`jinja2`、`python-dotenv`、`httpx`、`openai`、`redis`（M3 会话）；测试使用 `fakeredis`。Pydantic v2 随 FastAPI 安装。

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

### 2.2 Gradio 界面主题

| 变量 | 说明 |
|------|------|
| `GRADIO_UI_THEME` | 可选；**默认** `business`。仅允许：`business`（商务）、`warm`（温馨）、`minimal`（简约）。 |
| 留空 | 与 `business` 相同。 |
| 非法值 | 若**显式设置了错误取值**，启动时在 `validate_standalone_env` 阶段 **RuntimeError**。 |

代码中可用 `build_gradio_chat_blocks(theme="warm")` **覆盖**环境变量。

### 2.3 服务监听

`UVICORN_HOST`、`UVICORN_PORT` 由 `scripts/run.py` 读取；直接用 `uvicorn` 命令时也可在命令行指定 `--host` / `--port`。

### 2.4 Redis 与会话（M3，可选）

`REDIS_ENABLED=false`（默认）时**不连接 Redis**，行为与 M2 一致；不注册 `POST /api/sessions` 等会话路由。

| 变量 | 说明 |
|------|------|
| `REDIS_ENABLED` | `true` / `false`（或 `1` / `yes` / `on`）；为 `true` 时必须配置可用 `REDIS_URL`。 |
| `REDIS_URL` | 例如 `redis://127.0.0.1:6379/0`。 |
| `REDIS_KEY_PREFIX` | 默认 `icai:`；键形如 `{prefix}session:{uuid}:meta` / `:messages`（见 `project_goal.md` §2.3）。 |
| `REDIS_SESSION_TTL_SECONDS` | 默认 `2592000`（30 天）；每次写入会对待命中的 key 续期。 |
| `MEMORY_ROUNDS` | 默认 `3`；Gradio 首屏只展示最近 N **轮**（一轮 = `query` + `answer`）；`0` = 展示全部已存消息。 |
| `CHAT_MODE` | `messages`（默认）：与 M2 相同，多轮 `messages` 调 LLM。`prompt_template`：**必须** `REDIS_ENABLED=true`，使用 [`app/services/chat_prompt.md`](app/services/chat_prompt.md) 中的 `{historical_message}` 与 `{current_query}` 拼成**单条 user** 再流式调用 LLM（内置 Gradio / `/legacy` SSE）。 |

启用 Redis 后，进程启动时会对 Redis **ping**；**连不上则抛出 `RuntimeError` 且进程不进入可服务状态**。Gradio 路径依赖 **`SECRET_KEY`**（与 Starlette `SessionMiddleware` 签名 Cookie，用于同一浏览器内刷新后恢复 `session_id`）；请与 `.env.example` 一样设为足够随机的值。

**本地起 Redis 示例**：

```bash
docker run -d --name icai-redis -p 6379:6379 redis:7-alpine
```

**内置会话 API**（仅供内置 UI / 调试，非对外稳定契约）：

- `POST /api/sessions`：创建会话，返回 `session_id`（归属当前 `.env` 的 `USER_ID`）。
- `GET /api/sessions/{session_id}/messages`：拉取已存消息（`user_id` 与 meta 不一致时 **403**，无此会话时 **404**）。

**刷新恢复**：`/legacy` 使用 **sessionStorage**（`icai_legacy_session_id`）；`/gradio` 使用 **签名会话 Cookie** 中的 `icai_gradio_session_id`。流式结束后服务端将本轮 **user + assistant** 全文写入 Redis（失败仅打日志，不阻断 SSE）。每条落库记录含 **`user_id`、`session_id`、`type`**（如 `query` / `answer`，可扩展 `plan` 等）、**`content`、`timestamp`（UTC 字符串）**；旧版仅 `role`+`ts` 的数据在读时自动归一化。

**`app.integrations`**：仍走多轮 `stream_chat(messages=...)`，不受 `CHAT_MODE=prompt_template` 影响。

**安全提示**：`session_id` 可被猜测或泄露；M3 仅校验 meta 中的 `user_id` 与当前配置的 `USER_ID` 是否一致。**不能**将 `.env` 中的 `USER_ID` 当作公网多租户身份边界；生产应把真实身份放在网关 / OIDC，并自行加强会话绑定（见 `project_goal.md` §2.4、§5）。

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

```bash
python scripts/run.py
```

或：

```bash
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

| 地址 | 说明 |
|------|------|
| `/` | **302** 重定向到 Gradio。 |
| `/gradio` | **主聊天界面**（`Chatbot` 为 `type="messages"` 气泡布局）。 |
| `/legacy` | Jinja2 + 静态资源的旧版页（`POST /api/chat/stream`）。 |
| `/docs` | FastAPI OpenAPI（若未关闭）。 |

`REDIS_ENABLED=true` 时，OpenAPI 中还可看到 **`POST /api/sessions`**、**`GET /api/sessions/{session_id}/messages`**（与内置页配合使用）。

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
| `Standalone .env is missing required variables` | 按 `LLM_BACKEND` 补全 §2.1 表内变量。 |
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
- **单元测试**：在项目根执行 `python -m unittest discover -s tests -p 'test_*.py'`（M2 chunk / 模型列表；M3 `test_session_store` 使用 fakeredis）。

---

## 9. 仓库结构（速览）

```text
app/
  main.py              # FastAPI 入口、lifespan Redis、SessionMiddleware（M3）、挂载 Gradio / 静态资源 / 路由
  config.py            # AppConfig、RedisSettings、validate_standalone_env、get_gradio_ui_theme
  deps.py              # require_session_store（M3）
  integrations.py      # 对外 LLM 稳定导出
  runtime_config.py    # RuntimeConfig（库集成）
  memory/                # M3：keys、redis_pool、session_store、runtime（Gradio 绑定）
  ui/
    gradio_chat.py       # build_gradio_chat_blocks(...)
    gradio_themes.py     # business / warm / minimal 主题
  routes/                # chat_pages、chat_stream（SSE）、sessions（Redis 开启时）
  services/
    call_llm.py          # stream_chat、stream_chat_chunks、路由与回落
    call_deepseek.py     # DeepSeek 客户端
    call_ollama.py       # Ollama 客户端
    llm_chunks.py        # ChatStreamChunk（M2）
    llm_models.py        # list_chat_model_names（M2）
    chat_prompt.md       # CHAT_MODE=prompt_template：{historical_message}、{current_query}
    prompt_render.py     # 读模板、按轮拼 Markdown 历史
scripts/run.py           # 开发启动（reload）
tests/                   # unittest（test_llm_chunks、test_session_store、test_prompt_render）
tasks/                   # 设计文档、里程碑、参考图
```

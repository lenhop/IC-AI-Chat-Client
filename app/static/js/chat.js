/**
 * M1/M3 chat client: maintains messages[], POSTs to /api/chat/stream, parses SSE.
 *
 * M3: when data-redis-sessions=true, sessionStorage holds icai_legacy_session_id;
 * POST /api/sessions + GET messages hydrate history; stream body includes session_id.
 *
 * Input: user textarea + send button.
 * Output: appends user/assistant bubbles; streams assistant text via SSE frames.
 */

(function () {
  "use strict";

  const messagesEl = document.getElementById("messages");
  const inputEl = document.getElementById("user-input");
  const sendBtn = document.getElementById("btn-send");
  const stopBtn = document.getElementById("btn-stop");
  const bannerEl = document.getElementById("banner");
  const appRoot = document.querySelector(".app");
  const redisSessions =
    appRoot && appRoot.getAttribute("data-redis-sessions") === "true";
  const SESSION_STORAGE_KEY = "icai_legacy_session_id";

  /** @type {{ role: string, content: string }[]} */
  let messages = [];

  /** @type {string|null} */
  let sessionId = null;

  let abortController = null;

  /**
   * Show or hide the error banner.
   * @param {string} text
   * @param {boolean} isError
   */
  function showBanner(text, isError) {
    if (!text) {
      bannerEl.textContent = "";
      bannerEl.classList.add("banner--hidden");
      return;
    }
    bannerEl.textContent = text;
    bannerEl.classList.remove("banner--hidden");
    if (!isError) {
      bannerEl.style.borderColor = "var(--border)";
      bannerEl.style.background = "var(--assistant-bubble)";
      bannerEl.style.color = "var(--muted)";
    }
  }

  /**
   * Create a message div in the list.
   * @param {"user"|"assistant"} role
   * @param {string} content
   * @returns {HTMLElement}
   */
  function appendBubble(role, content) {
    const wrap = document.createElement("div");
    wrap.className = "msg msg--" + role;

    const label = document.createElement("div");
    label.className = "msg-role";
    label.textContent = role === "user" ? "你" : "助手";

    const body = document.createElement("div");
    body.className = "msg-body";
    body.textContent = content;

    wrap.appendChild(label);
    wrap.appendChild(body);
    messagesEl.appendChild(wrap);
    messagesEl.scrollTop = messagesEl.scrollHeight;
    return body;
  }

  /**
   * Parse SSE chunks from a text buffer; returns remaining incomplete buffer.
   * @param {string} buffer
   * @param {(obj: object) => void} onEvent
   * @returns {string}
   */
  function consumeSseBuffer(buffer, onEvent) {
    const parts = buffer.split("\n\n");
    const rest = parts.pop() || "";
    for (const block of parts) {
      const lines = block.split("\n");
      for (const line of lines) {
        if (line.startsWith("data:")) {
          const raw = line.slice(5).trim();
          if (raw === "[DONE]") {
            continue;
          }
          let parsed;
          try {
            parsed = JSON.parse(raw);
          } catch (e) {
            console.warn("bad sse json", raw);
            continue;
          }
          // Key point: callback errors (e.g. obj.error) must bubble to outer try/catch.
          onEvent(parsed);
        }
      }
    }
    return rest;
  }

  /**
   * Allocate a server session id (Redis-backed).
   * @returns {Promise<string>}
   */
  async function createSessionFromApi() {
    const res = await fetch("/api/sessions", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
    });
    if (!res.ok) {
      throw new Error("创建会话失败 (" + res.status + ")");
    }
    const data = await res.json();
    if (!data.session_id) {
      throw new Error("会话响应缺少 session_id");
    }
    return String(data.session_id);
  }

  /**
   * Ensure legacy session id and render stored messages (M3).
   * @returns {Promise<void>}
   */
  async function bootstrapRedisSession() {
    if (!redisSessions) {
      return;
    }
    try {
      let sid = sessionStorage.getItem(SESSION_STORAGE_KEY);
      if (!sid) {
        sid = await createSessionFromApi();
        sessionStorage.setItem(SESSION_STORAGE_KEY, sid);
      }
      sessionId = sid;
      let msgRes = await fetch(
        "/api/sessions/" + encodeURIComponent(sid) + "/messages"
      );
      if (msgRes.status === 404 || msgRes.status === 403) {
        sessionStorage.removeItem(SESSION_STORAGE_KEY);
        sid = await createSessionFromApi();
        sessionStorage.setItem(SESSION_STORAGE_KEY, sid);
        sessionId = sid;
        msgRes = await fetch(
          "/api/sessions/" + encodeURIComponent(sid) + "/messages"
        );
      }
      if (!msgRes.ok) {
        throw new Error("加载历史失败 (" + msgRes.status + ")");
      }
      const msgData = await msgRes.json();
      const list = msgData.messages || [];
      messages = list.slice();
      messagesEl.innerHTML = "";
      for (let i = 0; i < list.length; i += 1) {
        const m = list[i];
        if (m.role === "user" || m.role === "assistant") {
          appendBubble(m.role, m.content || "");
        }
      }
    } catch (err) {
      console.warn("bootstrapRedisSession", err);
      const m = err && err.message ? err.message : String(err);
      showBanner(m, true);
    }
  }

  /**
   * Send current input and stream assistant reply.
   */
  async function sendMessage() {
    const text = (inputEl.value || "").trim();
    if (!text) {
      showBanner("请输入内容。", true);
      return;
    }

    showBanner("", false);
    messages.push({ role: "user", content: text });
    appendBubble("user", text);
    inputEl.value = "";

    const assistantBody = appendBubble("assistant", "");
    messages.push({ role: "assistant", content: "" });

    sendBtn.disabled = true;
    stopBtn.disabled = false;
    abortController = new AbortController();

    const payload = { messages: messages.slice(0, -1) };
    if (sessionId) {
      payload.session_id = sessionId;
    }

    try {
      const res = await fetch("/api/chat/stream", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
        signal: abortController.signal,
      });

      if (!res.ok) {
        const errJson = await res.json().catch(function () {
          return {};
        });
        const msg = errJson.error || "请求失败 (" + res.status + ")";
        throw new Error(msg);
      }

      const reader = res.body && res.body.getReader();
      if (!reader) {
        throw new Error("浏览器不支持流式读取。");
      }

      const decoder = new TextDecoder();
      let buffer = "";

      while (true) {
        const { done, value } = await reader.read();
        if (done) {
          break;
        }
        buffer += decoder.decode(value, { stream: true });
        buffer = consumeSseBuffer(buffer, function (obj) {
          if (obj.error) {
            throw new Error(obj.error);
          }
          if (obj.delta) {
            assistantBody.textContent += obj.delta;
            const last = messages[messages.length - 1];
            if (last && last.role === "assistant") {
              last.content += obj.delta;
            }
            messagesEl.scrollTop = messagesEl.scrollHeight;
          }
        });
      }
    } catch (err) {
      if (err.name === "AbortError") {
        showBanner("已停止生成。", false);
      } else {
        const m = err && err.message ? err.message : String(err);
        showBanner(m, true);
        assistantBody.textContent += "\n\n[错误] " + m;
      }
    } finally {
      sendBtn.disabled = false;
      stopBtn.disabled = true;
      abortController = null;
    }
  }

  sendBtn.addEventListener("click", function () {
    sendMessage();
  });

  stopBtn.addEventListener("click", function () {
    if (abortController) {
      abortController.abort();
    }
  });

  inputEl.addEventListener("keydown", function (ev) {
    if (ev.key === "Enter" && !ev.shiftKey) {
      ev.preventDefault();
      if (!sendBtn.disabled) {
        sendMessage();
      }
    }
  });

  bootstrapRedisSession();
})();

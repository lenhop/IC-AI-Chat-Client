"""
Gradio ``Theme`` presets for m1_plan_v3: business (default), warm, minimal.

Visual intent (not pixel-perfect vs reference PNGs in ``tasks/``):
- business: cool slate/blue, crisp cards, professional density.
- warm: soft rose/stone gradient, larger radius, friendly tone.
- minimal: Gradio default theme, little chrome.
"""

from __future__ import annotations

from typing import Literal

import gradio as gr

from app.config import AppConfig

GradioUiTheme = Literal["business", "warm", "minimal"]

ALLOWED_UI_THEMES: tuple[str, ...] = ("business", "warm", "minimal")


def normalize_ui_theme(name: str | None) -> GradioUiTheme:
    """Return a valid theme id; unknown values fall back to ``business``."""
    key = (name or "business").strip().lower()
    if key in ALLOWED_UI_THEMES:
        return key  # type: ignore[return-value]
    return "business"


def build_gradio_theme(theme: GradioUiTheme) -> gr.Theme:
    """Build a ``gr.Theme`` instance for the given preset."""
    if theme == "minimal":
        return gr.themes.Default()

    if theme == "warm":
        return (
            gr.themes.Soft(
                primary_hue=gr.themes.colors.rose,
                secondary_hue=gr.themes.colors.orange,
                neutral_hue=gr.themes.colors.stone,
                radius_size=gr.themes.sizes.radius_lg,
                font=(
                    gr.themes.GoogleFont("Nunito"),
                    "ui-sans-serif",
                    "system-ui",
                    "sans-serif",
                ),
            ).set(
                body_background_fill="linear-gradient(165deg, #fff5f3 0%, #fffdfb 40%, #fef6ee 100%)",
                block_background_fill="#ffffff",
                block_border_width="0px",
                block_shadow="0 8px 28px rgba(244, 114, 182, 0.08)",
                button_primary_background_fill="linear-gradient(90deg, #fb7185, #fdba74)",
                button_primary_background_fill_hover="linear-gradient(90deg, #f43f5e, #fb923c)",
            )
        )

    # business (default)
    return (
        gr.themes.Soft(
            primary_hue=gr.themes.colors.blue,
            secondary_hue=gr.themes.colors.slate,
            neutral_hue=gr.themes.colors.slate,
            radius_size=gr.themes.sizes.radius_md,
            font=(
                gr.themes.GoogleFont("Source Sans 3"),
                "ui-sans-serif",
                "system-ui",
                "sans-serif",
            ),
        ).set(
            body_background_fill="#eef2f7",
            block_background_fill="#ffffff",
            block_border_width="1px",
            block_border_color="#e2e8f0",
            block_shadow="0 1px 3px rgba(15, 23, 42, 0.06)",
            button_primary_background_fill="#2563eb",
            button_primary_background_fill_hover="#1d4ed8",
        )
    )


def theme_header_html(cfg: AppConfig, theme: GradioUiTheme) -> str:
    """Top banner HTML aligned with the selected visual family."""
    model_label = (
        cfg.deepseek_llm_model if cfg.llm_backend == "deepseek" else cfg.ollama_generate_model
    )
    if theme == "business":
        return f"""
<div style="padding:14px 18px;margin-bottom:12px;border-radius:10px;background:linear-gradient(90deg,#1e40af 0%,#2563eb 100%);color:#fff;
  box-shadow:0 4px 14px rgba(37,99,235,0.25);font-family:system-ui,Segoe UI,sans-serif;">
  <div style="font-size:1.05rem;font-weight:650;letter-spacing:0.02em;">IC-AI Chat · 商务工作台</div>
  <div style="opacity:0.92;font-size:0.88rem;margin-top:6px;">
    后端 <code style="background:rgba(255,255,255,0.15);padding:2px 6px;border-radius:4px;">{cfg.llm_backend}</code>
    · 模型 <code style="background:rgba(255,255,255,0.15);padding:2px 6px;border-radius:4px;">{model_label}</code>
    · 用户 <code style="background:rgba(255,255,255,0.15);padding:2px 6px;border-radius:4px;">{cfg.user_id}</code>
  </div>
  <div style="opacity:0.85;font-size:0.82rem;margin-top:8px;">主入口：<a href="/gradio" style="color:#bfdbfe;">/gradio</a></div>
</div>
"""
    if theme == "warm":
        return f"""
<div style="padding:16px 18px;margin-bottom:14px;border-radius:16px;background:linear-gradient(120deg,#fff1f2,#fff7ed);
  border:1px solid #fecdd3;box-shadow:0 6px 24px rgba(251,113,133,0.12);font-family:Nunito,system-ui,sans-serif;">
  <div style="font-size:1.08rem;font-weight:700;color:#9f1239;">今天想聊点什么？</div>
  <div style="color:#57534e;font-size:0.9rem;margin-top:8px;line-height:1.5;">
    <span style="background:#ffe4e6;color:#be123c;padding:2px 8px;border-radius:999px;font-size:0.78rem;">{cfg.llm_backend}</span>
    &nbsp;·&nbsp;模型 {model_label}
    &nbsp;·&nbsp;{cfg.user_id}
  </div>
  <div style="margin-top:10px;font-size:0.82rem;"><a href="/gradio" style="color:#e11d48;">Gradio 主界面</a></div>
</div>
"""
    # minimal
    return (
        f"**后端** `{cfg.llm_backend}` · **模型** `{model_label}` · **用户** `{cfg.user_id}`  \n"
        "主入口：[打开 `/gradio`](/gradio)"
    )


def theme_extra_css(theme: GradioUiTheme) -> str:
    """
    PC-first layout: Gradio defaults to a narrow centered column (mobile-like).

    Empty Chatbot often uses ``min-width: auto`` so the column shrinks to a
    \"small\" strip until messages appear; we force the main column and every
    block row to **stretch to 100%** from the first paint (stable large layout).
    """
    base = """
/* ----- Shell: always use full viewport width ----- */
.gradio-container {
  max-width: none !important;
  width: 100% !important;
  padding-left: max(20px, env(safe-area-inset-left)) !important;
  padding-right: max(20px, env(safe-area-inset-right)) !important;
  box-sizing: border-box !important;
}
.gradio-container > .main,
.gradio-container .main.fillable,
.gradio-container .wrap {
  max-width: none !important;
  width: 100% !important;
  align-self: stretch !important;
}
.contain {
  max-width: none !important;
  width: 100% !important;
}

/* ----- Flex chain: do not shrink main column to content (empty chat) ----- */
.gradio-container .main .wrap,
.gradio-container .main form,
.gradio-container .main .form {
  width: 100% !important;
  max-width: none !important;
  display: flex !important;
  flex-direction: column !important;
  align-items: stretch !important;
}
.gradio-container .main .wrap > .gr-column,
.gradio-container .main .wrap > div[class*="column"] {
  flex: 1 1 100% !important;
  width: 100% !important;
  max-width: 100% !important;
  min-width: 0 !important;
  align-self: stretch !important;
}
/* Our root column: stay wide even when chat history is empty */
.icai-chat-root.gr-column {
  width: 100% !important;
  max-width: 100% !important;
  flex: 1 1 100% !important;
  min-width: min(100%, calc(100vw - 40px)) !important;
}
/* Rows (input + buttons) span full column width */
.icai-chat-root .gr-row,
.icai-chat-root .form > .gr-row {
  width: 100% !important;
}
/* Chatbot block: full width from first load (not only after messages) */
.icai-chat-root .block.gr-chatbot,
.icai-chat-root [class*="chatbot"] {
  width: 100% !important;
  max-width: 100% !important;
  min-width: 0 !important;
}
.icai-chat-root .block.gr-chatbot > div,
.icai-chat-root .block.gr-chatbot .wrap {
  width: 100% !important;
  max-width: 100% !important;
}
"""
    if theme == "business":
        return base + (
            "footer { opacity: 0.75; }\n"
            ".message-wrap.user { border-left: 3px solid #2563eb !important; }\n"
        )
    if theme == "warm":
        return base + "footer { opacity: 0.8; }\n"
    return base

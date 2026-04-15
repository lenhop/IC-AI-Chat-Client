"""
Gradio ``Theme`` presets for m1_plan_v3: business (default), warm, minimal.

Legacy ``theme_header_html`` (Chinese banners) was removed in v3.7; the live UI
uses ``gradio_layout`` + CSS (``theme_extra_css``) instead.

Visual intent (not pixel-perfect vs reference PNGs in ``tasks/``):
- business: cool slate/blue, crisp cards, professional density.
- warm: soft rose/stone gradient, larger radius, friendly tone.
- minimal: Gradio default theme, little chrome.
"""

from __future__ import annotations

from typing import Literal

import gradio as gr

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

/* ----- v3.7 layout: title bar + two columns ----- */
.icai-chat-root {
  gap: 0 !important;
}
.icai-title-container {
  margin: 0 !important;
  margin-bottom: 0 !important;
  padding: 0 !important;
  border: none !important;
  background: transparent !important;
  box-shadow: none !important;
  width: 100% !important;
  max-width: 100% !important;
}
.icai-title-container > div,
.icai-title-container .html-container,
.icai-title-container .prose {
  border: none !important;
  padding: 0 !important;
  background: transparent !important;
}
.icai-client-title-bar {
  display: block;
  width: 100%;
  margin: 0;
  padding: 14px 20px;
  box-sizing: border-box;
  background: #0d47a1;
  color: #ffffff;
  font-size: 1.125rem;
  font-weight: 600;
  letter-spacing: 0.02em;
  border-radius: 0;
  border: none;
}
.icai-main-row {
  gap: 0 !important;
  min-height: calc(100vh - 160px);
}
.icai-sidebar {
  background: #ffffff;
  border-right: 1px solid #e2e8f0;
  padding: 16px 18px;
}
.icai-meta-field {
  margin-bottom: 10px !important;
}
/* Sidebar metadata: Markdown blocks (no bordered Textbox) */
.icai-sidebar .icai-meta-md.block {
  border: none !important;
  background: transparent !important;
  box-shadow: none !important;
  padding: 0 0 12px 0 !important;
}
.icai-sidebar .icai-meta-md .prose,
.icai-sidebar .icai-meta-md .wrap {
  border: none !important;
  background: transparent !important;
  box-shadow: none !important;
}
.icai-sidebar .icai-meta-md p {
  margin: 0.2rem 0 0 0 !important;
  font-size: 0.9375rem;
  color: #111827;
}
.icai-sidebar .icai-meta-md strong {
  font-weight: 600;
  color: #374151;
  font-size: 0.75rem;
  text-transform: uppercase;
  letter-spacing: 0.04em;
}
.icai-session-md code {
  font-size: 0.82rem !important;
  word-break: break-all;
}
.icai-chat-panel {
  background: #ffffff;
  min-height: 0 !important;
  padding: 16px;
}
.icai-chat-panel .gr-chatbot,
.icai-chat-panel .icai-dialog-box {
  min-height: 360px !important;
  max-height: calc(100vh - 280px) !important;
}
.icai-chat-input {
  margin-top: 12px !important;
}
.icai-chat-actions {
  gap: 12px !important;
  margin-top: 8px !important;
}

/* ----- Chatbot: single bubble layer (no inner bordered card) ----- */
/*
 * Gradio bubble layout stacks: (1) block chrome on .gr-chatbot, (2) .bot/.user bubble,
 * (3) optional .message-bubble-border (scoped hash changes across builds — use [class*=]).
 * We keep one visible layer: strip (1) and (3), flatten inner wrappers under .bot/.user.
 */
.icai-chat-panel .block.gr-chatbot,
.icai-chat-panel .block.gr-chatbot > .wrap {
  border: none !important;
  box-shadow: none !important;
  background: transparent !important;
}
/* Inner markdown frame: remove second border (hash-suffix safe). */
.icai-chat-panel .message-wrap [class*="message-bubble-border"],
.icai-chat-panel .gr-chatbot [class*="message-bubble-border"] {
  border: none !important;
  border-width: 0 !important;
  background: transparent !important;
  box-shadow: none !important;
  padding: 0 !important;
  margin: 0 !important;
  outline: none !important;
}
/* Selectable wrapper between .bot and .message-content: no extra card. */
.icai-chat-panel .message-row.bubble .bot > div:not(.avatar-container),
.icai-chat-panel .message-row.bubble .user > div:not(.avatar-container) {
  border: none !important;
  background: transparent !important;
  box-shadow: none !important;
  outline: none !important;
}
.icai-chat-panel .message-wrap .prose.chatbot.md,
.icai-chat-panel .message-wrap .prose.md,
.icai-chat-panel .gr-chatbot .prose.chatbot.md,
.icai-chat-panel .gr-chatbot .prose.md {
  background: transparent !important;
  border: none !important;
  box-shadow: none !important;
  opacity: 1 !important;
}
.icai-chat-panel .message-wrap .bot .md,
.icai-chat-panel .message-wrap .user .md,
.icai-chat-panel .gr-chatbot .bot .md,
.icai-chat-panel .gr-chatbot .user .md {
  background: transparent !important;
  border: none !important;
  box-shadow: none !important;
}
/* Collapse extra full-width panels inside a message row (bubble layout) */
.icai-chat-panel .message-wrap .message-row .panel,
.icai-chat-panel .message-wrap .panel-full-width {
  border: none !important;
  background: transparent !important;
  box-shadow: none !important;
}
/* MessageContent root: no nested fill (markdown code blocks keep their own chrome). */
.icai-chat-panel .message-wrap .bot .message-content,
.icai-chat-panel .message-wrap .user .message-content,
.icai-chat-panel .gr-chatbot .bot .message-content,
.icai-chat-panel .gr-chatbot .user .message-content {
  background: transparent !important;
  border: none !important;
  box-shadow: none !important;
  padding: 0 !important;
  margin: 0 !important;
}
@media (max-width: 768px) {
  .icai-main-row {
    flex-direction: column !important;
  }
  .icai-sidebar {
    border-right: none;
    border-bottom: 1px solid #e2e8f0;
  }
  .icai-chat-panel .gr-chatbot,
  .icai-chat-panel .icai-dialog-box {
    max-height: 60vh !important;
  }
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

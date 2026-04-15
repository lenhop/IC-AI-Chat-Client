"""
Facade entrypoints for Gradio chat UI.

This module exposes a stable external interface while delegating concrete
responsibilities to layout, handler, and persistence services.
"""

from __future__ import annotations

from dataclasses import dataclass
import inspect
import logging
from typing import Optional

import gradio as gr
from fastapi import FastAPI

from app.config import AppConfig, MessageDisplayOptions, get_config, get_gradio_ui_theme, validate_app_config_for_ui
from app.runtime_config import RuntimeConfig, validate_runtime_config
from app.ui.gradio_handlers import GradioHandlerService
from app.ui.gradio_layout import GradioLayoutService
from app.ui.gradio_persistence import GradioPersistenceService
from app.ui.gradio_themes import (
    GradioUiTheme,
    build_gradio_theme,
    normalize_ui_theme,
    theme_extra_css,
)
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class GradioChatBuildContext:
    """Resolved immutable build context for Gradio chat assembly."""

    app_config: AppConfig
    theme_key: GradioUiTheme
    gradio_theme: gr.Theme
    extra_css: str
    page_title: str
    backend_label: str
    model_label: str
    user_label: str


class GradioChatFacade:
    """Single facade interface for building and mounting Gradio chat."""

    @classmethod
    def _resolve_backend_label(cls, backend_key: str) -> str:
        """Return user-facing backend label from internal backend key."""
        backend_map = {"deepseek": "DeepSeek", "ollama": "Ollama"}
        normalized_backend = (backend_key or "").strip().lower()
        return backend_map.get(normalized_backend, backend_key)

    @classmethod
    def _resolve_model_label(cls, app_config: AppConfig) -> str:
        """Return currently active model id for sidebar metadata."""
        if app_config.llm_backend == "deepseek":
            return app_config.deepseek_llm_model
        return app_config.ollama_generate_model

    @classmethod
    def _mount_supports_theme_css(cls) -> bool:
        """
        Return whether ``gr.mount_gradio_app`` accepts ``theme`` and ``css`` kwargs.

        Some installed Gradio builds expose a mount signature without these
        parameters. In that case we must omit them to avoid ``TypeError``.
        """
        try:
            params = inspect.signature(gr.mount_gradio_app).parameters
        except (TypeError, ValueError) as exc:
            logger.warning("Unable to inspect gr.mount_gradio_app signature: %s", exc)
            return False
        return "theme" in params and "css" in params

    @classmethod
    def _resolve_context(
        cls,
        app_config: Optional[AppConfig] = None,
        *,
        theme: Optional[str] = None,
        runtime: Optional[RuntimeConfig] = None,
    ) -> GradioChatBuildContext:
        """
        Resolve validated config and presentation assets for UI build/mount.

        Args:
            app_config: Optional explicit app configuration.
            theme: Optional theme override.
            runtime: Optional runtime override for transport calls.

        Returns:
            Fully resolved build context.
        """
        resolved_cfg = app_config if app_config is not None else get_config()
        validate_app_config_for_ui(resolved_cfg)
        if runtime is not None:
            validate_runtime_config(runtime)

        theme_key: GradioUiTheme = normalize_ui_theme(
            theme if theme is not None else get_gradio_ui_theme()
        )
        return GradioChatBuildContext(
            app_config=resolved_cfg,
            theme_key=theme_key,
            gradio_theme=build_gradio_theme(theme_key),
            extra_css=theme_extra_css(theme_key),
            page_title="IC-AI-Chat Client",
            backend_label=cls._resolve_backend_label(resolved_cfg.llm_backend),
            model_label=cls._resolve_model_label(resolved_cfg),
            user_label=resolved_cfg.user_id,
        )

    @classmethod
    def build_blocks(
        cls,
        app_config: Optional[AppConfig] = None,
        *,
        theme: Optional[str] = None,
        runtime: Optional[RuntimeConfig] = None,
    ) -> gr.Blocks:
        """
        Build Gradio blocks for mounting under FastAPI.

        Notes:
            Theme/CSS are injected at mount-time to avoid Gradio 6 constructor
            migration warnings.
        """
        context = cls._resolve_context(app_config, theme=theme, runtime=runtime)
        display_options = MessageDisplayOptions.from_app_config(context.app_config)

        def _page_load(request: gr.Request):
            return GradioPersistenceService.hydrate_or_create_session(
                request,
                user_id=context.app_config.user_id,
                llm_backend=context.app_config.llm_backend,
                memory_rounds=context.app_config.memory_rounds,
                display_options=display_options,
            )

        def _user_turn(message: str, history, session_id, request: Optional[gr.Request] = None):
            return GradioHandlerService.handle_user_turn(
                message,
                history,
                session_id,
                request=request,
                user_id=context.app_config.user_id,
            )

        def _stream_assistant(history, session_id, request: Optional[gr.Request] = None):
            return GradioHandlerService.stream_assistant(
                history,
                session_id,
                request=request,
                user_id=context.app_config.user_id,
                chat_mode=context.app_config.chat_mode,
                memory_rounds=context.app_config.memory_rounds,
                runtime=runtime,
            )

        def _clear_chat(session_id, request: Optional[gr.Request] = None):
            return GradioHandlerService.handle_clear_chat(
                session_id,
                request=request,
                user_id=context.app_config.user_id,
            )

        return GradioLayoutService.build_blocks(
            page_title=context.page_title,
            theme_key=context.theme_key,
            backend_label=context.backend_label,
            model_label=context.model_label,
            user_label=context.user_label,
            on_load=_page_load,
            on_user_turn=_user_turn,
            on_stream_assistant=_stream_assistant,
            on_clear_chat=_clear_chat,
        )

    @classmethod
    def mount_on_fastapi(
        cls,
        app: FastAPI,
        *,
        path: str = "/gradio",
        app_config: Optional[AppConfig] = None,
        theme: Optional[str] = None,
        runtime: Optional[RuntimeConfig] = None,
    ) -> FastAPI:
        """Mount Gradio chat under FastAPI with Gradio 6-compatible theme/css injection."""
        context = cls._resolve_context(app_config, theme=theme, runtime=runtime)
        blocks = cls.build_blocks(app_config=context.app_config, theme=context.theme_key, runtime=runtime)
        mount_kwargs = {"path": path}
        if cls._mount_supports_theme_css():
            mount_kwargs["theme"] = context.gradio_theme
            mount_kwargs["css"] = context.extra_css
        else:
            logger.warning(
                "Current gradio.mount_gradio_app does not support theme/css kwargs; "
                "mounting without them for compatibility."
            )
        return gr.mount_gradio_app(app, blocks, **mount_kwargs)


def build_gradio_chat_blocks(
    app_config: Optional[AppConfig] = None,
    *,
    theme: Optional[str] = None,
    runtime: Optional[RuntimeConfig] = None,
) -> gr.Blocks:
    """
    Backward-compatible low-level builder for ``gr.Blocks`` only.

    Prefer :func:`mount_gradio_chat_app` for production mounting, because that
    path injects theme/css during ``gr.mount_gradio_app`` (Gradio 6 compatible).
    """
    return GradioChatFacade.build_blocks(app_config, theme=theme, runtime=runtime)


def mount_gradio_chat_app(
    app: FastAPI,
    *,
    path: str = "/gradio",
    app_config: Optional[AppConfig] = None,
    theme: Optional[str] = None,
    runtime: Optional[RuntimeConfig] = None,
) -> FastAPI:
    """Recommended facade wrapper for mounting Gradio chat into FastAPI."""
    return GradioChatFacade.mount_on_fastapi(
        app,
        path=path,
        app_config=app_config,
        theme=theme,
        runtime=runtime,
    )

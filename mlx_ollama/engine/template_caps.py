"""Detect chat template capabilities by inspecting the Jinja2 template string."""

import logging
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class TemplateCaps:
    supports_tools: bool = False
    supports_enable_thinking: bool = False
    has_thinking_tags: bool = False


def detect_caps(tokenizer: Any) -> TemplateCaps:
    """Inspect the tokenizer's chat_template to determine supported features."""
    tpl = getattr(tokenizer, "chat_template", None)
    if tpl is None:
        return TemplateCaps()

    # Handle list-of-dicts format (named templates)
    if isinstance(tpl, list):
        tpl = " ".join(t.get("template", "") for t in tpl if isinstance(t, dict))

    return TemplateCaps(
        supports_tools="tools" in tpl,
        supports_enable_thinking="enable_thinking" in tpl,
        has_thinking_tags="<think>" in tpl or "thinking" in tpl.lower(),
    )

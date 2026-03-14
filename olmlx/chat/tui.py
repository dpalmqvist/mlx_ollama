"""Rich-based terminal UI for chat."""

import json
import logging

from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.text import Text

logger = logging.getLogger(__name__)


class ChatTUI:
    """Terminal UI using Rich for markdown rendering and panels."""

    def __init__(self):
        self.console = Console()

    def display_welcome(self, model_name: str, tools: list[dict]) -> None:
        """Show welcome panel with model and tool info."""
        lines = [f"Model: {model_name}"]
        if tools:
            lines.append(f"Tools: {len(tools)} available")
            for tool in tools:
                name = tool.get("function", {}).get("name", "?")
                desc = tool.get("function", {}).get("description", "")
                lines.append(f"  - {name}: {desc}")
        else:
            lines.append("Tools: none (use --mcp-config or ~/.olmlx/mcp.json)")
        lines.append("")
        lines.append("Commands: /exit, /clear, /tools, /system <prompt>, /model <name>")
        self.console.print(Panel("\n".join(lines), title="olmlx chat", border_style="blue"))

    def get_user_input(self) -> str | None:
        """Prompt for user input. Returns None on exit (Ctrl+D/Ctrl+C)."""
        try:
            lines = []
            prompt = "[bold green]> [/bold green]"
            while True:
                if not lines:
                    line = self.console.input(prompt)
                else:
                    line = self.console.input("[dim]... [/dim]")

                if line.endswith("\\"):
                    lines.append(line[:-1])
                else:
                    lines.append(line)
                    break

            return "\n".join(lines)
        except (EOFError, KeyboardInterrupt):
            return None

    def stream_response(self, initial_text: str = "") -> "StreamContext":
        """Return a context manager for streaming response display."""
        return StreamContext(self.console, initial_text)

    def display_thinking(self, text: str) -> None:
        """Show thinking in a dimmed panel."""
        self.console.print(
            Panel(
                Text(text, style="dim"),
                title="thinking",
                border_style="dim",
            )
        )

    def display_tool_call(self, name: str, arguments: dict) -> None:
        """Show tool invocation panel."""
        args_str = json.dumps(arguments, indent=2) if arguments else "{}"
        self.console.print(
            Panel(
                f"[bold]{name}[/bold]\n{args_str}",
                title="tool call",
                border_style="yellow",
            )
        )

    def display_tool_result(self, name: str, result: str) -> None:
        """Show tool response panel."""
        # Truncate long results
        display = result if len(result) <= 2000 else result[:2000] + "\n... (truncated)"
        self.console.print(
            Panel(display, title=f"tool result: {name}", border_style="green")
        )

    def display_tool_error(self, name: str, error: str) -> None:
        """Show tool error panel."""
        self.console.print(
            Panel(
                f"[red]{error}[/red]",
                title=f"tool error: {name}",
                border_style="red",
            )
        )

    def display_error(self, message: str) -> None:
        """Show red error panel."""
        self.console.print(Panel(f"[red]{message}[/red]", title="error", border_style="red"))

    def display_tools(self, tools: list[dict]) -> None:
        """Show available tools."""
        if not tools:
            self.console.print("[dim]No tools available[/dim]")
            return
        lines = []
        for tool in tools:
            func = tool.get("function", {})
            name = func.get("name", "?")
            desc = func.get("description", "")
            lines.append(f"[bold]{name}[/bold]: {desc}")
        self.console.print(Panel("\n".join(lines), title="available tools", border_style="blue"))


class StreamContext:
    """Context manager for streaming display.

    Uses plain Text during streaming to avoid O(n^2) Markdown re-parsing.
    Renders final Markdown on exit.
    """

    def __init__(self, console: Console, initial_text: str = ""):
        self.console = console
        self._chunks: list[str] = [initial_text] if initial_text else []
        self.live: Live | None = None

    def __enter__(self):
        initial = "".join(self._chunks)
        self.live = Live(
            Text(initial),
            console=self.console,
            refresh_per_second=10,
        )
        self.live.__enter__()
        return self

    def __exit__(self, *args):
        if self.live is not None:
            # Render final output as Markdown
            text = self.get_text()
            if text:
                self.live.update(Markdown(text))
            self.live.__exit__(*args)
            self.live = None

    def update(self, token: str) -> None:
        """Append a token and refresh the display."""
        self._chunks.append(token)
        if self.live is not None:
            self.live.update(Text("".join(self._chunks)))

    def get_text(self) -> str:
        """Return the accumulated text."""
        return "".join(self._chunks)

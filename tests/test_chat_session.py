"""Tests for olmlx.chat.session."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from olmlx.chat.config import ChatConfig
from olmlx.chat.session import ChatSession


def _make_session(
    *,
    mcp=None,
    model_name="test:latest",
    thinking=True,
    max_turns=25,
    system_prompt=None,
):
    config = ChatConfig(
        model_name=model_name,
        thinking=thinking,
        max_turns=max_turns,
        system_prompt=system_prompt,
    )
    manager = MagicMock()
    return ChatSession(config=config, manager=manager, mcp=mcp)


class TestChatSessionInit:
    def test_empty_messages(self):
        session = _make_session()
        assert session.messages == []

    def test_system_prompt_in_messages(self):
        session = _make_session(system_prompt="Be helpful.")
        assert len(session.messages) == 1
        assert session.messages[0] == {"role": "system", "content": "Be helpful."}


class TestAgentLoop:
    @pytest.mark.asyncio
    async def test_simple_text_response(self):
        """Model returns plain text, no tool calls."""
        session = _make_session()

        async def fake_stream(*args, **kwargs):
            yield {"text": "Hello", "done": False}
            yield {"text": " world", "done": False}
            yield {"text": "", "done": True, "stats": MagicMock()}

        with patch("olmlx.chat.session.generate_chat", return_value=fake_stream()):
            events = []
            async for event in session.send_message("Hi"):
                events.append(event)

        # Should have token events and a done event
        token_events = [e for e in events if e["type"] == "token"]
        assert len(token_events) == 2
        assert token_events[0]["text"] == "Hello"
        assert token_events[1]["text"] == " world"

        done_events = [e for e in events if e["type"] == "done"]
        assert len(done_events) == 1

        # Messages should contain user and assistant
        assert session.messages[-2]["role"] == "user"
        assert session.messages[-2]["content"] == "Hi"
        assert session.messages[-1]["role"] == "assistant"
        assert session.messages[-1]["content"] == "Hello world"

    @pytest.mark.asyncio
    async def test_thinking_extracted(self):
        """Model output with <think> tags should emit thinking event."""
        session = _make_session()

        async def fake_stream(*args, **kwargs):
            yield {"text": "<think>Let me think</think>The answer is 42", "done": False}
            yield {"text": "", "done": True, "stats": MagicMock()}

        with patch("olmlx.chat.session.generate_chat", return_value=fake_stream()):
            events = []
            async for event in session.send_message("What is the answer?"):
                events.append(event)

        thinking_events = [e for e in events if e["type"] == "thinking"]
        assert len(thinking_events) == 1
        assert "Let me think" in thinking_events[0]["text"]

        # Assistant message should have the visible text only
        assert session.messages[-1]["content"] == "The answer is 42"

    @pytest.mark.asyncio
    async def test_tool_call_agent_loop(self):
        """Model calls a tool, result is fed back, model continues."""
        mcp = MagicMock()
        mcp.get_tools_for_chat.return_value = [
            {
                "type": "function",
                "function": {
                    "name": "read_file",
                    "description": "Read a file",
                    "parameters": {"type": "object", "properties": {"path": {"type": "string"}}},
                },
            }
        ]
        mcp.call_tool = AsyncMock(return_value="file contents here")

        session = _make_session(mcp=mcp)
        call_count = 0

        async def fake_generate(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                # First call: model makes a tool call
                yield {
                    "text": '<tool_call>{"name": "read_file", "arguments": {"path": "/tmp/test.txt"}}</tool_call>',
                    "done": False,
                }
                yield {"text": "", "done": True, "stats": MagicMock()}
            else:
                # Second call: model responds with the result
                yield {"text": "The file contains: file contents here", "done": False}
                yield {"text": "", "done": True, "stats": MagicMock()}

        with patch("olmlx.chat.session.generate_chat", side_effect=lambda *a, **kw: fake_generate()):
            events = []
            async for event in session.send_message("Read /tmp/test.txt"):
                events.append(event)

        # Should have tool_call and tool_result events
        tool_call_events = [e for e in events if e["type"] == "tool_call"]
        assert len(tool_call_events) == 1
        assert tool_call_events[0]["name"] == "read_file"

        tool_result_events = [e for e in events if e["type"] == "tool_result"]
        assert len(tool_result_events) == 1
        assert "file contents here" in tool_result_events[0]["result"]

        # MCP should have been called
        mcp.call_tool.assert_awaited_once_with("read_file", {"path": "/tmp/test.txt"})

        # Should have two generate_chat calls
        assert call_count == 2

    @pytest.mark.asyncio
    async def test_max_turns_limit(self):
        """Agent loop stops after max_turns to prevent infinite loops."""
        mcp = MagicMock()
        mcp.get_tools_for_chat.return_value = [
            {
                "type": "function",
                "function": {
                    "name": "ping",
                    "description": "Ping",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ]
        mcp.call_tool = AsyncMock(return_value="pong")

        session = _make_session(mcp=mcp, max_turns=2)

        async def fake_generate(*args, **kwargs):
            # Always make a tool call
            yield {
                "text": '<tool_call>{"name": "ping", "arguments": {}}</tool_call>',
                "done": False,
            }
            yield {"text": "", "done": True, "stats": MagicMock()}

        with patch("olmlx.chat.session.generate_chat", side_effect=lambda *a, **kw: fake_generate()):
            events = []
            async for event in session.send_message("Keep pinging"):
                events.append(event)

        # Should have max_turns_exceeded event
        exceeded = [e for e in events if e["type"] == "max_turns_exceeded"]
        assert len(exceeded) == 1

    @pytest.mark.asyncio
    async def test_tool_call_error_fed_back(self):
        """Tool call error is fed back to the model as a tool result."""
        mcp = MagicMock()
        mcp.get_tools_for_chat.return_value = [
            {
                "type": "function",
                "function": {
                    "name": "fail_tool",
                    "description": "A tool that fails",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ]
        mcp.call_tool = AsyncMock(side_effect=RuntimeError("Connection refused"))

        session = _make_session(mcp=mcp)
        call_count = 0

        async def fake_generate(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                yield {
                    "text": '<tool_call>{"name": "fail_tool", "arguments": {}}</tool_call>',
                    "done": False,
                }
                yield {"text": "", "done": True, "stats": MagicMock()}
            else:
                yield {"text": "The tool failed, sorry.", "done": False}
                yield {"text": "", "done": True, "stats": MagicMock()}

        with patch("olmlx.chat.session.generate_chat", side_effect=lambda *a, **kw: fake_generate()):
            events = []
            async for event in session.send_message("Use the tool"):
                events.append(event)

        error_events = [e for e in events if e["type"] == "tool_error"]
        assert len(error_events) == 1
        assert "Connection refused" in error_events[0]["error"]

        # Error should be fed back as tool result in messages
        tool_msgs = [m for m in session.messages if m["role"] == "tool"]
        assert len(tool_msgs) == 1
        assert "Error" in tool_msgs[0]["content"]

    @pytest.mark.asyncio
    async def test_no_mcp_no_tools(self):
        """Without MCP, no tools are passed to generate_chat."""
        session = _make_session(mcp=None)

        async def fake_stream(*args, **kwargs):
            # Verify no tools kwarg
            assert kwargs.get("tools") is None
            yield {"text": "Hi", "done": False}
            yield {"text": "", "done": True, "stats": MagicMock()}

        with patch("olmlx.chat.session.generate_chat", side_effect=lambda *a, **kw: fake_stream(*a, **kw)):
            events = []
            async for event in session.send_message("Hello"):
                events.append(event)

    @pytest.mark.asyncio
    async def test_cache_info_skipped(self):
        """cache_info chunks from generate_chat should be silently skipped."""
        session = _make_session()

        async def fake_stream(*args, **kwargs):
            yield {"cache_info": True, "cache_read_tokens": 100, "cache_creation_tokens": 50}
            yield {"text": "Hello", "done": False}
            yield {"text": "", "done": True, "stats": MagicMock()}

        with patch("olmlx.chat.session.generate_chat", return_value=fake_stream()):
            events = []
            async for event in session.send_message("Hi"):
                events.append(event)

        # No cache_info events should leak through
        assert all(e["type"] != "cache_info" for e in events)


class TestClearHistory:
    def test_clear_removes_messages(self):
        session = _make_session()
        session.messages.append({"role": "user", "content": "Hi"})
        session.messages.append({"role": "assistant", "content": "Hello"})
        session.clear_history()
        assert session.messages == []

    def test_clear_preserves_system_prompt(self):
        session = _make_session(system_prompt="Be helpful.")
        session.messages.append({"role": "user", "content": "Hi"})
        session.clear_history()
        assert len(session.messages) == 1
        assert session.messages[0]["role"] == "system"
        assert session.messages[0]["content"] == "Be helpful."

    def test_clear_uses_updated_system_prompt(self):
        """After changing config.system_prompt, clear_history uses the new one."""
        session = _make_session(system_prompt="Old prompt.")
        session.config.system_prompt = "New prompt."
        session.clear_history()
        assert len(session.messages) == 1
        assert session.messages[0]["content"] == "New prompt."

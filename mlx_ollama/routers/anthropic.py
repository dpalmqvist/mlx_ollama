import asyncio
import json
import logging
import uuid

from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse

from mlx_ollama.engine.inference import generate_chat
from mlx_ollama.engine.tool_parser import _make_tool_use_id, parse_model_output
from mlx_ollama.schemas.anthropic import (
    AnthropicContentBlock,
    AnthropicMessagesRequest,
    AnthropicMessagesResponse,
    AnthropicUsage,
)

router = APIRouter()
logger = logging.getLogger(__name__)


def _make_msg_id() -> str:
    return f"msg_{uuid.uuid4().hex[:24]}"


# --- Tool/message conversion ---

def _convert_tools(req: AnthropicMessagesRequest) -> list[dict] | None:
    """Convert Anthropic tool definitions to OpenAI-style for chat templates."""
    if not req.tools:
        return None
    tools = []
    for tool in req.tools:
        tools.append({
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description or "",
                "parameters": tool.input_schema.model_dump(exclude_none=True),
            },
        })
    return tools


def _convert_messages(req: AnthropicMessagesRequest) -> list[dict]:
    """Convert Anthropic message format to internal chat format.

    Handles text, tool_use, and tool_result content blocks.
    """
    messages = []

    # System message
    if req.system:
        if isinstance(req.system, str):
            messages.append({"role": "system", "content": req.system})
        else:
            text = " ".join(b.text for b in req.system if b.text)
            messages.append({"role": "system", "content": text})

    for msg in req.messages:
        if isinstance(msg.content, str):
            messages.append({"role": msg.role, "content": msg.content})
            continue

        if msg.role == "assistant":
            text_parts = []
            tool_calls = []
            for block in msg.content:
                if block.type == "thinking":
                    # Skip thinking blocks in history — model regenerates its own
                    continue
                elif block.type == "text" and block.text:
                    text_parts.append(block.text)
                elif block.type == "tool_use":
                    tool_calls.append({
                        "id": block.id or _make_tool_use_id(),
                        "type": "function",
                        "function": {
                            "name": block.name,
                            "arguments": block.input or {},
                        },
                    })

            entry = {"role": "assistant", "content": " ".join(text_parts) if text_parts else ""}
            if tool_calls:
                entry["tool_calls"] = tool_calls
            messages.append(entry)

        elif msg.role == "user":
            text_parts = []
            for block in msg.content:
                if block.type == "text" and block.text:
                    text_parts.append(block.text)
                elif block.type == "tool_result":
                    result_content = ""
                    if isinstance(block.content, str):
                        result_content = block.content
                    elif isinstance(block.content, list):
                        result_content = " ".join(
                            b.get("text", "") if isinstance(b, dict) else str(b)
                            for b in block.content
                        )
                    messages.append({
                        "role": "tool",
                        "tool_call_id": block.tool_use_id or "",
                        "content": result_content,
                    })
            if text_parts:
                messages.append({"role": "user", "content": " ".join(text_parts)})

    return messages


def _build_options(req: AnthropicMessagesRequest) -> dict:
    opts = {}
    if req.temperature is not None:
        opts["temperature"] = req.temperature
    if req.top_p is not None:
        opts["top_p"] = req.top_p
    if req.top_k is not None:
        opts["top_k"] = req.top_k
    if req.stop_sequences:
        opts["stop"] = req.stop_sequences
    return opts


# --- SSE helpers ---

def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


@router.post("/v1/messages")
async def anthropic_messages(req: AnthropicMessagesRequest, request: Request):
    logger.info("Anthropic request: model=%s stream=%s tools=%d messages=%d max_tokens=%d",
                req.model, req.stream, len(req.tools or []), len(req.messages), req.max_tokens)
    manager = request.app.state.model_manager
    messages = _convert_messages(req)
    options = _build_options(req)
    tools = _convert_tools(req)
    has_tools = bool(tools)
    tool_names = {t["function"]["name"] for t in tools} if tools else None
    msg_id = _make_msg_id()
    logger.debug("Converted %d messages, %d tools", len(messages), len(tools or []))

    if req.stream:
        result = await generate_chat(
            manager, req.model, messages, options,
            tools=tools, stream=True, max_tokens=req.max_tokens,
        )

        async def stream_sse():
            # message_start
            yield _sse("message_start", {
                "type": "message_start",
                "message": {
                    "id": msg_id,
                    "type": "message",
                    "role": "assistant",
                    "content": [],
                    "model": req.model,
                    "stop_reason": None,
                    "stop_sequence": None,
                    "usage": {
                        "input_tokens": 0, "output_tokens": 0,
                        "cache_creation_input_tokens": 0,
                        "cache_read_input_tokens": 0,
                    },
                },
            })

            if has_tools:
                # With tools: buffer fully (needed for tag parsing), emit pings as keepalive
                full_text = ""
                output_tokens = 0
                last_ping = asyncio.get_running_loop().time()
                ping_interval = 5.0

                async for chunk in result:
                    if chunk.get("done"):
                        stats = chunk.get("stats")
                        if stats:
                            output_tokens = stats.eval_count
                        break
                    full_text += chunk.get("text", "")

                    # Send keepalive pings while buffering
                    now = asyncio.get_running_loop().time()
                    if now - last_ping >= ping_interval:
                        yield _sse("ping", {"type": "ping"})
                        last_ping = now

                logger.info("Raw model output (%d chars): %s", len(full_text), full_text[:1000])

                thinking, visible_text, tool_uses = parse_model_output(
                    full_text, has_tools, tool_names=tool_names,
                )

                if tool_uses:
                    logger.info("Parsed %d tool call(s): %s",
                                len(tool_uses), [tu["name"] for tu in tool_uses])

                block_idx = 0

                if thinking:
                    yield _sse("content_block_start", {
                        "type": "content_block_start",
                        "index": block_idx,
                        "content_block": {"type": "thinking", "thinking": ""},
                    })
                    chunk_size = 1000
                    for i in range(0, len(thinking), chunk_size):
                        yield _sse("content_block_delta", {
                            "type": "content_block_delta",
                            "index": block_idx,
                            "delta": {"type": "thinking_delta", "thinking": thinking[i:i + chunk_size]},
                        })
                    yield _sse("content_block_stop", {"type": "content_block_stop", "index": block_idx})
                    block_idx += 1

                yield _sse("content_block_start", {
                    "type": "content_block_start",
                    "index": block_idx,
                    "content_block": {"type": "text", "text": ""},
                })
                if visible_text:
                    chunk_size = 100
                    for i in range(0, len(visible_text), chunk_size):
                        yield _sse("content_block_delta", {
                            "type": "content_block_delta",
                            "index": block_idx,
                            "delta": {"type": "text_delta", "text": visible_text[i:i + chunk_size]},
                        })
                yield _sse("content_block_stop", {"type": "content_block_stop", "index": block_idx})
                block_idx += 1

                for tool_use in tool_uses:
                    yield _sse("content_block_start", {
                        "type": "content_block_start",
                        "index": block_idx,
                        "content_block": {
                            "type": "tool_use",
                            "id": tool_use["id"],
                            "name": tool_use["name"],
                            "input": {},
                        },
                    })
                    yield _sse("content_block_delta", {
                        "type": "content_block_delta",
                        "index": block_idx,
                        "delta": {"type": "input_json_delta", "partial_json": json.dumps(tool_use["input"])},
                    })
                    yield _sse("content_block_stop", {"type": "content_block_stop", "index": block_idx})
                    block_idx += 1

                stop_reason = "tool_use" if tool_uses else "end_turn"

            else:
                # Without tools: stream incrementally with thinking state machine
                block_idx = 0
                output_tokens = 0
                buffer = ""

                # States: "init", "thinking", "text"
                state = "init"
                text_block_started = False

                async for chunk in result:
                    if chunk.get("done"):
                        stats = chunk.get("stats")
                        if stats:
                            output_tokens = stats.eval_count
                        break

                    token_text = chunk.get("text", "")
                    buffer += token_text

                    # State machine for incremental streaming
                    while buffer:
                        if state == "init":
                            # Check if output starts with <think>
                            if buffer.startswith("<think>"):
                                state = "thinking"
                                buffer = buffer[7:]  # consume <think>
                                yield _sse("content_block_start", {
                                    "type": "content_block_start",
                                    "index": block_idx,
                                    "content_block": {"type": "thinking", "thinking": ""},
                                })
                            elif len(buffer) < 7 and "<think>".startswith(buffer):
                                # Could still be a <think> tag — wait for more tokens
                                break
                            else:
                                state = "text"
                                # Don't consume — let text state handle it

                        elif state == "thinking":
                            # Look for </think> closing tag
                            end_idx = buffer.find("</think>")
                            if end_idx >= 0:
                                # Emit thinking content up to the close tag
                                thinking_chunk = buffer[:end_idx]
                                if thinking_chunk:
                                    yield _sse("content_block_delta", {
                                        "type": "content_block_delta",
                                        "index": block_idx,
                                        "delta": {"type": "thinking_delta", "thinking": thinking_chunk},
                                    })
                                yield _sse("content_block_stop", {"type": "content_block_stop", "index": block_idx})
                                block_idx += 1
                                buffer = buffer[end_idx + 8:]  # consume </think>
                                state = "text"
                            elif len(buffer) > 8:
                                # Emit everything except the last 8 chars (might be partial </think>)
                                safe = buffer[:-8]
                                buffer = buffer[-8:]
                                yield _sse("content_block_delta", {
                                    "type": "content_block_delta",
                                    "index": block_idx,
                                    "delta": {"type": "thinking_delta", "thinking": safe},
                                })
                                break
                            else:
                                break

                        elif state == "text":
                            if not text_block_started:
                                yield _sse("content_block_start", {
                                    "type": "content_block_start",
                                    "index": block_idx,
                                    "content_block": {"type": "text", "text": ""},
                                })
                                text_block_started = True
                            # Emit all buffered text
                            if buffer:
                                yield _sse("content_block_delta", {
                                    "type": "content_block_delta",
                                    "index": block_idx,
                                    "delta": {"type": "text_delta", "text": buffer},
                                })
                                buffer = ""
                            break

                # Flush any remaining buffer
                if state == "thinking":
                    if buffer:
                        yield _sse("content_block_delta", {
                            "type": "content_block_delta",
                            "index": block_idx,
                            "delta": {"type": "thinking_delta", "thinking": buffer},
                        })
                    yield _sse("content_block_stop", {"type": "content_block_stop", "index": block_idx})
                    block_idx += 1
                    # Model ended mid-think — still need a text block
                    yield _sse("content_block_start", {
                        "type": "content_block_start",
                        "index": block_idx,
                        "content_block": {"type": "text", "text": ""},
                    })
                    yield _sse("content_block_stop", {"type": "content_block_stop", "index": block_idx})
                elif text_block_started:
                    yield _sse("content_block_stop", {"type": "content_block_stop", "index": block_idx})
                elif state == "text" and not text_block_started:
                    # Transitioned to text but never emitted — emit empty text block
                    yield _sse("content_block_start", {
                        "type": "content_block_start",
                        "index": block_idx,
                        "content_block": {"type": "text", "text": ""},
                    })
                    if buffer:
                        yield _sse("content_block_delta", {
                            "type": "content_block_delta",
                            "index": block_idx,
                            "delta": {"type": "text_delta", "text": buffer},
                        })
                    yield _sse("content_block_stop", {"type": "content_block_stop", "index": block_idx})
                else:
                    # state == "init" — no output at all, emit empty text block
                    yield _sse("content_block_start", {
                        "type": "content_block_start",
                        "index": block_idx,
                        "content_block": {"type": "text", "text": ""},
                    })
                    yield _sse("content_block_stop", {"type": "content_block_stop", "index": block_idx})

                stop_reason = "end_turn"

            # message_delta + message_stop
            yield _sse("message_delta", {
                "type": "message_delta",
                "delta": {"stop_reason": stop_reason, "stop_sequence": None},
                "usage": {"output_tokens": output_tokens},
            })
            yield _sse("message_stop", {"type": "message_stop"})

        return StreamingResponse(stream_sse(), media_type="text/event-stream")
    else:
        result = await generate_chat(
            manager, req.model, messages, options,
            tools=tools, stream=False, max_tokens=req.max_tokens,
        )
        text = result.get("text", "")
        stats = result.get("stats")

        logger.debug("Raw model output (%d chars): %s", len(text), text[:500])

        thinking, visible_text, tool_uses = parse_model_output(
            text, has_tools, tool_names=tool_names,
        )

        content_blocks = []

        if thinking:
            content_blocks.append(
                AnthropicContentBlock(type="thinking", text=thinking)
            )

        if visible_text:
            content_blocks.append(
                AnthropicContentBlock(type="text", text=visible_text)
            )

        for tu in tool_uses:
            content_blocks.append(AnthropicContentBlock(
                type="tool_use",
                id=tu["id"],
                name=tu["name"],
                input=tu["input"],
            ))

        if not content_blocks:
            content_blocks.append(AnthropicContentBlock(type="text", text=""))

        stop_reason = "tool_use" if tool_uses else "end_turn"
        usage = AnthropicUsage(
            input_tokens=stats.prompt_eval_count if stats else 0,
            output_tokens=stats.eval_count if stats else 0,
        )
        return AnthropicMessagesResponse(
            id=msg_id,
            content=content_blocks,
            model=req.model,
            stop_reason=stop_reason,
            usage=usage,
        )

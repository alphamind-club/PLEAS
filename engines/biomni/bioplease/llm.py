import os
import re
from typing import Any, Iterator, Literal, Optional

import openai
from langchain_anthropic import ChatAnthropic

# from langchain_aws import ChatBedrock
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage
from langchain_core.outputs import ChatResult
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama
from langchain_openai import AzureChatOpenAI, ChatOpenAI

# Only params MiniMax actually accepts (whitelist is safer than blacklist)
# tools/tool_choice are intentionally excluded — MiniMax rejects them with
# error 2013 when the payload uses OpenAI's tool-call format.
_MINIMAX_ALLOWED = frozenset({
    "model", "messages", "temperature", "max_tokens", "max_completion_tokens",
    "stream", "top_p", "n", "extra_body",
    "extra_headers", "extra_query", "timeout",
})


def _msg_to_dict(msg) -> dict:
    """Normalise a LangChain message object or plain dict to {role, content}."""
    if isinstance(msg, dict):
        return msg
    # LangChain BaseMessage subclasses
    from langchain_core.messages import (
        HumanMessage, AIMessage, SystemMessage, ToolMessage, FunctionMessage
    )
    if isinstance(msg, SystemMessage):
        return {"role": "system", "content": str(msg.content)}
    if isinstance(msg, HumanMessage):
        return {"role": "user", "content": str(msg.content)}
    if isinstance(msg, (ToolMessage, FunctionMessage)):
        # Inline tool results as user turn
        tool_id = getattr(msg, "tool_call_id", "") or ""
        return {"role": "user", "content": f"[Tool Result (id={tool_id})]\n{msg.content}"}
    if isinstance(msg, AIMessage):
        content = str(msg.content) if msg.content else ""
        # Inline any tool_calls as text so MiniMax doesn't choke
        if getattr(msg, "tool_calls", None):
            for tc in msg.tool_calls:
                fn_name = tc.get("name", "") if isinstance(tc, dict) else getattr(tc, "name", "")
                fn_args = tc.get("args", "") if isinstance(tc, dict) else getattr(tc, "args", "")
                content += f"\n[Tool Call: {fn_name}({fn_args})]"
        return {"role": "assistant", "content": content.strip()}
    # Fallback
    role = getattr(msg, "type", "user")
    return {"role": role, "content": str(getattr(msg, "content", str(msg)))}


def _sanitize_messages_for_minimax(messages: list) -> list:
    """Sanitize messages for MiniMax API compatibility.

    MiniMax requires:
    1. No role="tool" messages — convert to user messages
    2. No tool_calls fields — inline as text
    3. No consecutive same-role messages — merge them
    4. Must start with system or user (not assistant)
    """
    sanitized = []
    for msg in messages:
        sanitized.append(_msg_to_dict(msg))

    # Now all entries are plain dicts — apply role-level sanitisation
    cleaned = []
    for msg in sanitized:
        role = msg.get("role", "user")
        content = str(msg.get("content") or "")

        # Convert tool result messages → user message (in case dict path missed them)
        if role == "tool":
            tool_call_id = msg.get("tool_call_id", "")
            cleaned.append({"role": "user", "content": f"[Tool Result (id={tool_call_id})]\n{content}"})
            continue

        # Strip tool_calls from assistant messages — inline as text
        if role == "assistant" and msg.get("tool_calls"):
            tc_text = ""
            for tc in msg.get("tool_calls", []):
                fn = tc.get("function", tc) if isinstance(tc, dict) else {}
                tc_text += f"\n[Tool Call: {fn.get('name', '')}({fn.get('arguments', '')})]"
            cleaned.append({"role": "assistant", "content": (content + tc_text).strip()})
            continue

        cleaned.append({"role": role, "content": content})

    # Merge consecutive same-role messages (MiniMax requires alternating turns)
    merged = []
    for msg in cleaned:
        if merged and merged[-1]["role"] == msg["role"]:
            merged[-1]["content"] = merged[-1]["content"] + "\n\n" + msg["content"]
        else:
            merged.append(dict(msg))

    return merged


class _MiniMaxClientWrapper:
    """Wraps the openai completions client to strip params MiniMax doesn't support."""

    def __init__(self, client):
        self._client = client

    def create(self, **kwargs):
        # Whitelist: only keep params MiniMax supports
        rejected = [k for k in kwargs if k not in _MINIMAX_ALLOWED]
        filtered = {k: v for k, v in kwargs.items() if k in _MINIMAX_ALLOWED}

        # Sanitize messages — merge consecutive same-role, strip tool roles/calls
        if "messages" in filtered:
            filtered["messages"] = _sanitize_messages_for_minimax(filtered["messages"])

        # MiniMax requires the last message to be user or system, not assistant
        msgs = filtered.get("messages", [])
        if msgs and isinstance(msgs[-1], dict) and msgs[-1].get("role") == "assistant":
            msgs.append({"role": "user", "content": "Continue."})

        # Clamp temperature to valid range (0.0, 1.0]
        if "temperature" in filtered:
            filtered["temperature"] = max(0.01, min(1.0, float(filtered["temperature"])))

        return self._client.create(**filtered)

    def __getattr__(self, name):
        return getattr(self._client, name)



class ChatAnthropicNormalizer(ChatAnthropic):
    """Normalises complex list content back into strings while preserving think blocks."""
    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        result = super()._generate(messages, stop=stop, run_manager=run_manager, **kwargs)
        for g in result.generations:
            if hasattr(g, "message") and hasattr(g.message, "content"):
                if isinstance(g.message.content, list):
                    text_parts = []
                    for block in g.message.content:
                        if isinstance(block, dict):
                            if block.get('type') == 'thinking':
                                text_parts.append(f"<think>\n{block.get('thinking', '')}\n</think>")
                            elif block.get('type') == 'text':
                                text_parts.append(block.get('text', ''))
                        elif isinstance(block, str):
                            text_parts.append(block)
                    g.message.content = "\n".join(text_parts).strip()
            if hasattr(g, "text") and isinstance(g.text, list):
                if isinstance(g.text[0], dict):
                    g.text = "\n".join(b.get('text', b.get('thinking', '')) for b in g.text if isinstance(b, dict))
                
        return result

class MiniMaxChat(ChatOpenAI):
    """ChatOpenAI subclass for MiniMax that strips unsupported params.

    MiniMax rejects: stop, presence_penalty, frequency_penalty, logit_bias,
    logprobs, top_logprobs, stream_options, parallel_tool_calls.
    Error code: 400 / invalid chat setting (2013).
    Also strips <think>...</think> reasoning blocks from response content
    so downstream parsers only see the actual answer.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Wrap the underlying openai client to strip bad params at call time
        self.client = _MiniMaxClientWrapper(self.client)

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        result = super()._generate(messages, stop=None, run_manager=run_manager, **kwargs)
        # Strip <think>...</think> blocks from every generation.
        # ChatResult.generations is List[ChatGeneration], not List[List[ChatGeneration]].
        for g in result.generations:
            if hasattr(g, "message") and hasattr(g.message, "content"):
                if isinstance(g.message.content, str):
                    g.message.content = re.sub(
                        r"<think>.*?</think>", "", g.message.content, flags=re.DOTALL
                    ).strip()
            if hasattr(g, "text") and isinstance(g.text, str):
                g.text = re.sub(
                    r"<think>.*?</think>", "", g.text, flags=re.DOTALL
                ).strip()
        return result

SourceType = Literal["OpenAI", "AzureOpenAI", "Anthropic", "Ollama", "Gemini", "Bedrock", "Groq", "MiniMax", "Custom"]
ALLOWED_SOURCES: set[str] = set(SourceType.__args__)


def get_llm(
    model: str = "gpt-4o-mini",
    temperature: float = 0.7,
    stop_sequences: list[str] | None = None,
    source: SourceType | None = None,
    base_url: str | None = None,
    api_key: str | None = None,
    extended_thinking: bool = False,
) -> BaseChatModel:
    """
    Get a language model instance based on the specified model name and source.
    This function supports models from OpenAI, Azure OpenAI, Anthropic, Ollama, Gemini, Bedrock, and custom model serving.
    Args:
        model (str): The model name to use
        temperature (float): Temperature setting for generation
        stop_sequences (list): Sequences that will stop generation
        source (str): Source provider: "OpenAI", "AzureOpenAI", "Anthropic", "Ollama", "Gemini", "Bedrock", "Groq", "MiniMax", or "Custom"
                      If None, will attempt to auto-detect from model name
        base_url (str): The base URL for custom model serving (e.g., "http://localhost:8000/v1"), default is None
        api_key (str): The API key for the custom llm
    """
    # Auto-detect source from model name if not specified
    if source is None:
        env_source = os.getenv("LLM_SOURCE")
        if env_source in ALLOWED_SOURCES:
            source = env_source
        else:
            if model[:7] == "claude-":
                source = "Anthropic"
            elif model[:4] == "gpt-":
                source = "OpenAI"
            elif model.startswith("azure-"):
                source = "AzureOpenAI"
            elif model[:7] == "gemini-":
                source = "Gemini"
            elif "groq" in model.lower():
                source = "Groq"
            elif model.startswith("MiniMax-") or "minimax" in model.lower():
                source = "MiniMax"
            elif base_url is not None:
                source = "Custom"
            elif "/" in model or any(
                name in model.lower()
                for name in ["llama", "mistral", "qwen", "gemma", "phi", "dolphin", "orca", "vicuna", "deepseek"]
            ):
                source = "Ollama"
            elif model.startswith(
                ("anthropic.claude-", "amazon.titan-", "meta.llama-", "mistral.", "cohere.", "ai21.", "us.")
            ):
                source = "Bedrock"
            else:
                raise ValueError("Unable to determine model source. Please specify 'source' parameter.")

    # Create appropriate model based on source
    if source == "OpenAI":
        return ChatOpenAI(model=model, temperature=temperature, stop_sequences=stop_sequences)
    elif source == "AzureOpenAI":
        API_VERSION = "2024-12-01-preview"
        model = model.replace("azure-", "")
        return AzureChatOpenAI(
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            azure_endpoint=os.getenv("OPENAI_ENDPOINT"),
            azure_deployment=model,
            openai_api_version=API_VERSION,
            temperature=temperature,
        )
    elif source == "Anthropic":
        kwargs = {
            "model": model,
            "temperature": temperature,
            "max_tokens": 4096,
            "stop_sequences": stop_sequences,
        }
        # Enable extended thinking for supported models
        if extended_thinking:
            kwargs["thinking"] = {"type": "enabled", "budget_tokens": 8000}
            # Temperature must be 1 when thinking is enabled
            kwargs["temperature"] = 1
            # max_tokens must be greater than thinking.budget_tokens
            kwargs["max_tokens"] = 16384
        return ChatAnthropicNormalizer(**kwargs)
    elif source == "Gemini":
        if extended_thinking:
            # The OpenAI-compat endpoint doesn't support thinking config.
            # Fall back to the native ChatGoogleGenerativeAI with thinking enabled.
            # max_output_tokens must be >> thinking_budget so Gemini reserves
            # enough tokens for the actual text response (not just thinking).
            from langchain_google_genai import ChatGoogleGenerativeAI
            return ChatGoogleGenerativeAI(
                model=model,
                google_api_key=os.getenv("GEMINI_API_KEY"),
                temperature=1,  # required for thinking
                thinking_budget=8192,
                max_output_tokens=24576,  # thinking (8192) + response (16384)
            )
        # Standard (non-thinking) path via OpenAI-compat endpoint
        # If you want to use ChatGoogleGenerativeAI, you need to pass the stop sequences upon invoking the model.
        # return ChatGoogleGenerativeAI(
        #     model=model,
        #     temperature=temperature,
        #     google_api_key=api_key,
        # )
        gemini_kwargs = dict(
            model=model,
            temperature=temperature,
            api_key=os.getenv("GEMINI_API_KEY"),
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
            stop_sequences=stop_sequences,
        )
        return ChatOpenAI(**gemini_kwargs)
    elif source == "Groq":
        return ChatOpenAI(
            model=model,
            temperature=temperature,
            api_key=os.getenv("GROQ_API_KEY"),
            base_url="https://api.groq.com/openai/v1",
            stop_sequences=stop_sequences,
        )
    elif source == "Ollama":
        return ChatOllama(
            model=model,
            temperature=temperature,
        )
        
    elif source == "Perplexity":
        return ChatOpenAI(
            model=model,
            temperature=temperature,
            api_key=os.getenv("PPLX_API_KEY"),
            base_url="https://api.perplexity.ai",
        )
    
    elif source == "MiniMax":
        api_key_to_use = api_key or os.getenv("MINIMAX_API_KEY")
        
        # fallback to dotenv if os.getenv didn't pick it up
        if not api_key_to_use:
            try:
                from dotenv import dotenv_values
                env_dict = dotenv_values(".env")
                api_key_to_use = env_dict.get("MINIMAX_API_KEY")
            except ImportError:
                pass
            
        if not api_key_to_use or "dummy" in api_key_to_use.lower() or "sk-" not in api_key_to_use:
            api_key_to_use = os.getenv("MINIMAX_API_KEY")

        if model in ("MiniMax-M2.7", "MiniMax-Text-01"):
            # MiniMax M2.7 supports the Anthropic API format explicitly
            return ChatAnthropicNormalizer(
                model=model,
                temperature=max(0.01, min(1.0, temperature)), # keep it constrained if needed
                max_tokens=4096,
                stop_sequences=stop_sequences,
                anthropic_api_key=api_key_to_use,
                anthropic_api_url="https://api.minimax.io/anthropic",
            )
            
        # MiniMax M2.5 via OpenAI-compatible API
        # Temperature must be in range (0.0, 1.0], recommended: 1.0
        # Uses MiniMaxChat subclass to strip unsupported params (stop, penalties, etc.)
        # that LangChain injects internally — causes error 2013 if passed.
        temp = max(0.01, min(1.0, temperature))  # Clamp to valid range
        return MiniMaxChat(
            model=model,
            temperature=temp,
            api_key=api_key_to_use,
            base_url="https://api.minimax.io/v1",
        )

    # elif source == "Bedrock":
    #     return ChatBedrock(
    #         model=model,
    #         temperature=temperature,
    #         stop_sequences=stop_sequences,
    #         region_name=os.getenv("AWS_REGION", "us-east-1"),
    #     )
    elif source == "Custom":
        # Custom LLM serving such as SGLang. Must expose an openai compatible API.
        assert base_url is not None, "base_url must be provided for customly served LLMs"
        llm = ChatOpenAI(
            model=model,
            temperature=temperature,
            max_tokens=8192,
            stop_sequences=stop_sequences,
            base_url=base_url,
            api_key=api_key,
        )
        return llm
    else:
        raise ValueError(
            f"Invalid source: {source}. Valid options are 'OpenAI', 'AzureOpenAI', 'Anthropic', 'Gemini', 'Groq', 'MiniMax', 'Perplexity', 'Bedrock', 'Ollama', or 'Custom'"
        )

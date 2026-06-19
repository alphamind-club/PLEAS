"""
Memory management system for the BioPLEASE agent to reduce token usage.

This module implements a two-tier memory system:
1. Short-term memory: Recent messages kept verbatim (configurable window)
2. Long-term memory: Compressed summaries of older conversations

The system automatically manages token budgets and provides efficient context retrieval.
"""

import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage


@dataclass
class MemoryConfig:
    """Configuration for memory management."""
    
    # Short-term memory settings
    short_window: int = 6  # Number of recent messages to keep verbatim
    max_message_length: int = 1200  # Max chars per message before truncation
    
    # Long-term memory settings
    summary_max_chars: int = 2500  # Max chars for the rolling summary
    summary_update_threshold: int = 4  # Update summary every N messages
    
    # Compression settings
    compression_ratio: float = 0.3  # Target compression ratio for summaries
    keep_critical_info: bool = True  # Preserve key info (goals, errors, files)
    
    # Token budget management
    estimate_tokens_per_char: float = 0.3  # Rough estimate: 1 token ≈ 3-4 chars
    max_total_tokens: int = 8000  # Max tokens for all memory context
    
    # Persistence
    auto_save: bool = True
    save_directory: Optional[str] = None


@dataclass
class MemorySnapshot:
    """A snapshot of memory state at a specific point in time."""
    
    timestamp: float
    phase: str
    short_term: List[Dict[str, Any]]
    long_term_summary: str
    metadata: Dict[str, Any] = field(default_factory=dict)


class MemoryManager:
    """
    Manages short-term and long-term memory for the agent.
    
    This class provides:
    - Automatic message history pruning
    - Intelligent summary compression
    - Token budget management
    - Memory persistence across sessions
    - Retrieval of relevant context
    """
    
    def __init__(self, config: Optional[MemoryConfig] = None, llm=None):
        """
        Initialize the memory manager.
        
        Args:
            config: Configuration for memory management
            llm: Language model for generating summaries
        """
        self.config = config or MemoryConfig()
        self.llm = llm
        
        # Memory storage
        self.long_term_summary: str = ""
        self.short_term_messages: List[BaseMessage] = []
        self.events: List[Dict[str, Any]] = []
        self.metadata: Dict[str, Any] = {}
        
        # Statistics
        self.stats = {
            "total_messages": 0,
            "compressions": 0,
            "tokens_saved": 0,
            "last_compression": None,
        }
        
        # Setup save directory if configured
        if self.config.auto_save and self.config.save_directory:
            os.makedirs(self.config.save_directory, exist_ok=True)
    
    def add_message(self, message: BaseMessage, phase: Optional[str] = None) -> None:
        """
        Add a new message to short-term memory.
        
        Args:
            message: The message to add
            phase: Optional phase tag (PLAN, EXECUTE, etc.)
        """
        self.short_term_messages.append(message)
        self.stats["total_messages"] += 1
        
        # Check if we need to compress
        if len(self.short_term_messages) > self.config.short_window + self.config.summary_update_threshold:
            self.compress_to_long_term(phase)
    
    def add_messages(self, messages: List[BaseMessage], phase: Optional[str] = None) -> None:
        """Add multiple messages at once."""
        for msg in messages:
            self.add_message(msg, phase)
    
    def compress_to_long_term(self, phase: Optional[str] = None) -> None:
        """
        Compress older messages into the long-term summary.
        
        Args:
            phase: Optional phase tag for logging
        """
        if not self.llm:
            # Fallback: simple truncation without LLM
            self._simple_compression()
            return
        
        # Determine messages to compress
        messages_to_keep = self.config.short_window
        messages_to_compress = self.short_term_messages[:-messages_to_keep]
        
        if not messages_to_compress:
            return
        
        # Generate new summary
        new_summary = self._generate_summary(messages_to_compress)
        
        # Update long-term memory
        self.long_term_summary = new_summary
        self.short_term_messages = self.short_term_messages[-messages_to_keep:]
        
        # Log event
        self.events.append({
            "timestamp": time.time(),
            "phase": phase or "unknown",
            "action": "compress",
            "messages_compressed": len(messages_to_compress),
        })
        
        self.stats["compressions"] += 1
        self.stats["last_compression"] = time.time()
        
        # Auto-save if configured
        if self.config.auto_save:
            self.save()
    
    def _generate_summary(self, messages: List[BaseMessage]) -> str:
        """
        Generate a compressed summary of messages using the LLM.
        
        Args:
            messages: Messages to summarize
            
        Returns:
            Compressed summary text
        """
        # Extract message contents
        msg_texts = []
        for m in messages:
            role = m.__class__.__name__.replace("Message", "").lower()
            content = getattr(m, "content", str(m))
            
            # Truncate very long messages
            if len(content) > self.config.max_message_length:
                content = content[:self.config.max_message_length] + " […]"
            
            msg_texts.append(f"[{role}] {content}")
        
        # Create summarization prompt
        prompt = self._create_summary_prompt(msg_texts)
        
        # Generate summary
        try:
            response = self.llm.invoke([HumanMessage(content=prompt)])
            new_summary = getattr(response, "content", str(response)).strip()
        except Exception as e:
            print(f"Warning: Summary generation failed: {e}")
            # Fallback to simple concatenation
            new_summary = self._fallback_summary(messages)
        
        # Combine with existing summary
        if self.long_term_summary:
            combined = f"{self.long_term_summary}\n\n[RECENT UPDATE]\n{new_summary}"
        else:
            combined = new_summary
        
        # Enforce max length
        if len(combined) > self.config.summary_max_chars:
            # Try to trim intelligently, keeping the most recent parts
            combined = "…" + combined[-(self.config.summary_max_chars - 1):]
        
        return combined
    
    def _create_summary_prompt(self, message_texts: List[str]) -> str:
        """Create a prompt for summarizing messages."""
        messages_str = "\n".join(message_texts)
        
        prompt = f"""Compress the following conversation history into a concise summary.

REQUIREMENTS:
- Keep ONLY critical information: goals, decisions, data/files used, errors and fixes
- Use bullet points (max 12 bullets)
- Be extremely concise - aim for {int(len(messages_str) * self.config.compression_ratio)} characters
- Focus on facts, not process
- Preserve specific names (files, tools, parameters)

PREVIOUS CONTEXT:
{self.long_term_summary if self.long_term_summary else "(none)"}

RECENT MESSAGES TO COMPRESS:
{messages_str}

COMPRESSED SUMMARY:"""
        return prompt
    
    def _fallback_summary(self, messages: List[BaseMessage]) -> str:
        """Simple fallback summary when LLM is unavailable."""
        summaries = []
        for m in messages[-5:]:  # Last 5 messages
            content = getattr(m, "content", str(m))
            role = m.__class__.__name__.replace("Message", "")
            
            # Extract key phrases
            if len(content) > 100:
                content = content[:100] + "…"
            summaries.append(f"{role}: {content}")
        
        return "\n".join(summaries)
    
    def _simple_compression(self) -> None:
        """Simple compression without LLM - just keep recent messages."""
        self.short_term_messages = self.short_term_messages[-self.config.short_window:]
    
    def get_context_messages(self, include_summary: bool = True) -> List[BaseMessage]:
        """
        Get the full context including both long-term and short-term memory.
        
        Args:
            include_summary: Whether to prepend the long-term summary
            
        Returns:
            List of messages representing the full context
        """
        context = []
        
        # Add long-term summary as a system message
        if include_summary and self.long_term_summary:
            summary_msg = SystemMessage(
                content=f"[MEMORY: LONG-TERM CONTEXT]\n{self.long_term_summary}"
            )
            context.append(summary_msg)
        
        # Add short-term messages
        context.extend(self.short_term_messages)
        
        return context
    
    def estimate_token_count(self) -> int:
        """
        Estimate the total token count of current memory.
        
        Returns:
            Estimated token count
        """
        total_chars = len(self.long_term_summary)
        
        for msg in self.short_term_messages:
            content = getattr(msg, "content", str(msg))
            total_chars += len(content)
        
        return int(total_chars * self.config.estimate_tokens_per_char)
    
    def is_over_budget(self) -> bool:
        """Check if memory is over the token budget."""
        return self.estimate_token_count() > self.config.max_total_tokens
    
    def force_compression(self, target_messages: int = None) -> None:
        """
        Force compression to reduce memory size.
        
        Args:
            target_messages: Target number of short-term messages to keep
        """
        if target_messages is None:
            target_messages = max(3, self.config.short_window // 2)
        
        messages_to_compress = self.short_term_messages[:-target_messages]
        
        if messages_to_compress:
            new_summary = self._generate_summary(messages_to_compress)
            self.long_term_summary = new_summary
            self.short_term_messages = self.short_term_messages[-target_messages:]
            
            self.stats["compressions"] += 1
    
    def clear(self, keep_long_term: bool = False) -> None:
        """
        Clear memory.
        
        Args:
            keep_long_term: Whether to keep the long-term summary
        """
        self.short_term_messages = []
        if not keep_long_term:
            self.long_term_summary = ""
        self.events = []
    
    def add_event(self, phase: str, note: str, metadata: Optional[Dict] = None) -> None:
        """
        Add a breadcrumb event to track agent progress.
        
        Args:
            phase: Phase name (PLAN, EXECUTE, etc.)
            note: Description of the event
            metadata: Optional additional data
        """
        event = {
            "timestamp": time.time(),
            "phase": phase,
            "note": note,
        }
        if metadata:
            event["metadata"] = metadata
        
        self.events.append(event)
    
    def create_snapshot(self, phase: str) -> MemorySnapshot:
        """
        Create a snapshot of current memory state.
        
        Args:
            phase: Current phase
            
        Returns:
            MemorySnapshot object
        """
        return MemorySnapshot(
            timestamp=time.time(),
            phase=phase,
            short_term=[
                {
                    "role": m.__class__.__name__,
                    "content": getattr(m, "content", str(m)),
                }
                for m in self.short_term_messages
            ],
            long_term_summary=self.long_term_summary,
            metadata={
                "stats": self.stats.copy(),
                "events_count": len(self.events),
            },
        )
    
    def save(self, filepath: Optional[str] = None) -> None:
        """
        Save memory to disk.
        
        Args:
            filepath: Optional custom filepath
        """
        if filepath is None:
            if self.config.save_directory:
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                filepath = os.path.join(
                    self.config.save_directory,
                    f"memory_{timestamp}.json"
                )
            else:
                return  # No save path configured
        
        data = {
            "long_term_summary": self.long_term_summary,
            "short_term_messages": [
                {
                    "role": m.__class__.__name__,
                    "content": getattr(m, "content", str(m)),
                }
                for m in self.short_term_messages
            ],
            "events": self.events,
            "metadata": self.metadata,
            "stats": self.stats,
            "timestamp": time.time(),
        }
        
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    
    def load(self, filepath: str) -> None:
        """
        Load memory from disk.
        
        Args:
            filepath: Path to saved memory file
        """
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        self.long_term_summary = data.get("long_term_summary", "")
        
        # Reconstruct messages
        self.short_term_messages = []
        for msg_data in data.get("short_term_messages", []):
            role = msg_data["role"]
            content = msg_data["content"]
            
            if role == "HumanMessage":
                msg = HumanMessage(content=content)
            elif role == "AIMessage":
                msg = AIMessage(content=content)
            elif role == "SystemMessage":
                msg = SystemMessage(content=content)
            else:
                msg = HumanMessage(content=content)  # Default
            
            self.short_term_messages.append(msg)
        
        self.events = data.get("events", [])
        self.metadata = data.get("metadata", {})
        self.stats = data.get("stats", self.stats)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get memory statistics."""
        return {
            **self.stats,
            "short_term_count": len(self.short_term_messages),
            "long_term_length": len(self.long_term_summary),
            "events_count": len(self.events),
            "estimated_tokens": self.estimate_token_count(),
            "over_budget": self.is_over_budget(),
        }
    
    def __repr__(self) -> str:
        """String representation of memory state."""
        stats = self.get_stats()
        return (
            f"MemoryManager("
            f"short={stats['short_term_count']}, "
            f"long_len={stats['long_term_length']}, "
            f"tokens~{stats['estimated_tokens']}, "
            f"compressions={stats['compressions']})"
        )

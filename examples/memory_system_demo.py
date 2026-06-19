"""
Example: Using the Enhanced Memory System in Biomni

This script demonstrates how to use the new memory management features
to reduce token usage in long conversations.
"""

# Make repository importable when running this script from the repo root.
# This inserts the project root (one level up from examples/) into sys.path so
# `from bioplease...` works without installing the package.
import os
import sys
repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

# Provide a tiny shim for optional packages that may not be installed in the
# runtime environment used for running examples. `a1.py` does `from dotenv
# import load_dotenv`, so if `python-dotenv` is not present we create a
# minimal module with a no-op `load_dotenv` so imports succeed.
try:
    import dotenv  # noqa: F401
except Exception:
    import types
    _dotenv = types.ModuleType("dotenv")
    def load_dotenv(*args, **kwargs):
        return None
    _dotenv.load_dotenv = load_dotenv
    sys.modules["dotenv"] = _dotenv

# Shim for langchain_core.messages used in examples when LangChain isn't
# installed. This provides minimal Message classes and registers a fake
# module at 'langchain_core.messages' so `from langchain_core.messages import ...`
# succeeds when `a1.py` imports those names at module import time.
try:
    from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage  # noqa: F401
except Exception:
    import types

    class BaseMessage:
        def __init__(self, content: str, **kwargs):
            self.content = content

    class HumanMessage(BaseMessage):
        pass

    class AIMessage(BaseMessage):
        pass

    class SystemMessage(BaseMessage):
        pass

    # Create a fake package and submodule so import hooks work the same as
    # the real langchain package (enables `from langchain_core.messages import ...`).
    _lc_pkg = types.ModuleType("langchain_core")
    _lc_mod = types.ModuleType("langchain_core.messages")
    _lc_mod.HumanMessage = HumanMessage
    _lc_mod.AIMessage = AIMessage
    _lc_mod.SystemMessage = SystemMessage
    _lc_mod.BaseMessage = BaseMessage
    # Make it behave like a package so submodule imports work
    _lc_pkg.__path__ = []
    sys.modules["langchain_core"] = _lc_pkg
    sys.modules["langchain_core.messages"] = _lc_mod

    # Minimal prompts shim used by a1.py
    _lc_prompts = types.ModuleType("langchain_core.prompts")
    class ChatPromptTemplate:
        def __init__(self, *args, **kwargs):
            pass
        @classmethod
        def from_messages(cls, msgs):
            return cls()
        def format_prompt(self, **kwargs):
            return ""
    _lc_prompts.ChatPromptTemplate = ChatPromptTemplate
    sys.modules["langchain_core.prompts"] = _lc_prompts

from bioplease.agent.a1 import A1
import json


def example_basic_usage():
    """Basic usage with default memory settings."""
    print("\n" + "="*60)
    print("EXAMPLE 1: Basic Usage with Default Settings")
    print("="*60 + "\n")
    
    # Create agent with default memory configuration
    agent = A1(path="./data", llm="gpt-4o-mini")
    
    print("✓ Agent created with default memory settings:")
    print(f"  - Short window: {agent.memory_config.short_window} messages")
    print(f"  - Summary max: {agent.memory_config.summary_max_chars} chars")
    print(f"  - Token budget: {agent.memory_config.max_total_tokens} tokens")
    print(f"  - Compression ratio: {agent.memory_config.compression_ratio}")


def example_custom_configuration():
    """Customize memory settings for different use cases."""
    print("\n" + "="*60)
    print("EXAMPLE 2: Custom Memory Configuration")
    print("="*60 + "\n")
    
    agent = A1(path="./data", llm="gpt-4o-mini")
    
    # Scenario 1: Long conversation - minimize tokens
    print("Scenario 1: Long Conversation (Aggressive Compression)")
    agent.configure_memory(
        short_window=4,           # Keep fewer messages
        compression_ratio=0.2,    # More aggressive compression
        max_total_tokens=6000     # Lower budget
    )
    
    # Scenario 2: Complex task - need more context
    print("\nScenario 2: Complex Task (More Context)")
    agent.configure_memory(
        short_window=10,          # Keep more messages
        compression_ratio=0.4,    # Gentler compression
        max_total_tokens=12000    # Higher budget
    )
    
    # Scenario 3: Cost optimization
    print("\nScenario 3: Cost Optimization (Minimal Tokens)")
    agent.configure_memory(
        short_window=3,
        compression_ratio=0.15,
        max_total_tokens=4000,
        auto_save=True
    )


def example_memory_monitoring():
    """Monitor memory usage during agent execution."""
    print("\n" + "="*60)
    print("EXAMPLE 3: Memory Monitoring")
    print("="*60 + "\n")
    
    agent = A1(path="./data", llm="gpt-4o-mini")
    
    # Initialize state (normally done by agent.go())
    from langchain_core.messages import HumanMessage, AIMessage
    state = {
        "messages": [],
        "artifacts": {}
    }
    
    # Simulate some conversation
    print("Simulating conversation...")
    for i in range(10):
        state["messages"].append(HumanMessage(content=f"User message {i+1}"))
        state["messages"].append(AIMessage(content=f"AI response {i+1}"))
    
    # Ensure memory is initialized
    agent._ensure_mem(state)
    
    # Add messages to memory manager
    memory_manager = state["artifacts"]["memory_manager"]
    for msg in state["messages"]:
        memory_manager.add_message(msg, phase="TEST")
    
    # Get statistics
    stats = agent.get_memory_stats(state)
    
    print("\nMemory Statistics:")
    print(f"  Total messages processed: {stats['total_messages']}")
    print(f"  Short-term messages: {stats['short_term_count']}")
    print(f"  Long-term summary length: {stats['long_term_length']} chars")
    print(f"  Compressions performed: {stats['compressions']}")
    print(f"  Estimated tokens: ~{stats['estimated_tokens']}")
    print(f"  Over budget: {stats['over_budget']}")
    
    # Pretty print full summary
    agent.print_memory_summary(state)


def example_save_and_load():
    """Save and load memory snapshots."""
    print("\n" + "="*60)
    print("EXAMPLE 4: Save and Load Memory")
    print("="*60 + "\n")
    
    agent = A1(path="./data", llm="gpt-4o-mini")
    
    # Create a state with some messages
    from langchain_core.messages import HumanMessage, AIMessage
    state = {
        "messages": [
            HumanMessage(content="What is CRISPR?"),
            AIMessage(content="CRISPR is a gene editing technology..."),
            HumanMessage(content="How does it work?"),
            AIMessage(content="It uses guide RNA to target specific DNA sequences..."),
        ],
        "artifacts": {}
    }
    
    # Initialize memory
    memory_manager = agent._ensure_mem(state)
    for msg in state["messages"]:
        memory_manager.add_message(msg, phase="LEARN")
    
    # Save memory
    save_path = "./data/bioplease_data/memory_snapshots/example_session.json"
    agent.save_memory_snapshot(state, save_path)
    print(f"✓ Memory saved to: {save_path}")
    
    # Clear memory
    agent.clear_memory(state)
    print("✓ Memory cleared")
    
    # Load memory back
    agent.load_memory_snapshot(state, save_path)
    print("✓ Memory loaded")
    
    # Verify
    stats = agent.get_memory_stats(state)
    print(f"\nRestored memory has {stats['short_term_count']} messages")


def example_manual_compression():
    """Manually trigger memory compression."""
    print("\n" + "="*60)
    print("EXAMPLE 5: Manual Memory Compression")
    print("="*60 + "\n")
    
    agent = A1(path="./data", llm="gpt-4o-mini")
    
    # Create state with many messages
    from langchain_core.messages import HumanMessage, AIMessage
    state = {"messages": [], "artifacts": {}}
    
    # Add 20 messages
    for i in range(20):
        state["messages"].append(HumanMessage(content=f"Question {i+1}"))
        state["messages"].append(AIMessage(content=f"Answer {i+1}"))
    
    # Initialize memory
    memory_manager = agent._ensure_mem(state)
    
    print(f"Before compression: {len(state['messages'])} messages")
    
    # Manually compress
    for msg in state["messages"]:
        memory_manager.add_message(msg)
    
    # Force compression to keep only 3 messages
    memory_manager.force_compression(target_messages=3)
    
    stats = agent.get_memory_stats(state)
    print(f"After compression: {stats['short_term_count']} messages in short-term")
    print(f"Long-term summary: {stats['long_term_length']} chars")
    print(f"Token savings: ~{len(state['messages']) * 100 - stats['estimated_tokens']} tokens")


def example_real_world_scenario():
    """Real-world scenario: Multi-step bioinformatics analysis."""
    print("\n" + "="*60)
    print("EXAMPLE 6: Real-World Bioinformatics Analysis")
    print("="*60 + "\n")
    
    agent = A1(path="./data", llm="gpt-4o-mini")
    
    # Configure for long-running analysis
    agent.configure_memory(
        short_window=6,
        compression_ratio=0.25,
        max_total_tokens=8000,
        auto_save=True
    )
    
    print("Configured for long-running analysis")
    print(f"Memory will auto-save to: {agent.memory_config.save_directory}")
    
    # Simulate a multi-step analysis workflow
    tasks = [
        "Load gene expression data from TCGA",
        "Perform quality control and normalization",
        "Identify differentially expressed genes",
        "Perform pathway enrichment analysis",
        "Generate visualization plots",
        "Write summary report"
    ]
    
    state = {"messages": [], "artifacts": {}}
    memory_manager = agent._ensure_mem(state)
    
    from langchain_core.messages import HumanMessage, AIMessage
    
    for i, task in enumerate(tasks, 1):
        # Simulate user request and agent response
        state["messages"].append(HumanMessage(content=task))
        state["messages"].append(AIMessage(content=f"Completed: {task}"))
        
        # Add to memory
        memory_manager.add_message(state["messages"][-2], phase="EXECUTE")
        memory_manager.add_message(state["messages"][-1], phase="EXECUTE")
        
        # Check memory after each task
        stats = agent.get_memory_stats(state)
        print(f"\nAfter task {i}: {stats['short_term_count']} messages, "
              f"~{stats['estimated_tokens']} tokens")
        
        if stats['compressions'] > 0:
            print(f"  → Memory compressed (total compressions: {stats['compressions']})")
    
    # Final summary
    print("\n" + "-"*60)
    agent.print_memory_summary(state)


def main():
    """Run all examples."""
    print("\n" + "="*70)
    print(" BIOPLEASE ENHANCED MEMORY SYSTEM - EXAMPLES")
    print("="*70)
    
    try:
        example_basic_usage()
        example_custom_configuration()
        example_memory_monitoring()
        example_save_and_load()
        example_manual_compression()
        example_real_world_scenario()
        
        print("\n" + "="*70)
        print(" All examples completed successfully!")
        print("="*70 + "\n")
        
    except Exception as e:
        print(f"\n❌ Error running examples: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

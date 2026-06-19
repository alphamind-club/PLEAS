"""
Example: Using MiniMax M2.5 with BioPLEASE

This example demonstrates how to configure and use MiniMax M2.5 models
in BioPLEASE's phase-based agent system.

MiniMax M2.5 models are accessed via their OpenAI-compatible API.
Model names typically follow the pattern: "abab-2.5-*"
"""

import os
from bioplease.agent import A1

# =============================================================================
# STEP 1: Set your MiniMax API Key
# =============================================================================
# Option A: Set environment variable before running
#   export MINIMAX_API_KEY="your-minimax-api-key"

# Option B: Set it in Python (not recommended for production)
# os.environ["MINIMAX_API_KEY"] = "your-minimax-api-key"

# =============================================================================
# STEP 2: Basic Usage - Use MiniMax M2.5 for all phases
# =============================================================================
def example_basic_usage():
    """Use MiniMax M2.5 for all phases"""
    print("="*60)
    print("Example 1: Basic MiniMax M2.5 Usage")
    print("="*60)
    
    # Create agent with MiniMax M2.5
    agent = A1(
        path="./data/minimax_example",
        llm="MiniMax-M2.5",  # MiniMax M2.5 model
        source="MiniMax",     # Specify MiniMax as the source
        cost_budget=1.0
    )
    
    # Run a simple query
    result = agent.run("Explain the CRISPR-Cas9 mechanism in 3 sentences")
    print(f"Result: {result}")


# =============================================================================
# STEP 3: Phase-Specific Configuration
# =============================================================================
def example_phase_specific():
    """Configure different models for different phases"""
    print("\n" + "="*60)
    print("Example 2: Phase-Specific MiniMax Configuration")
    print("="*60)
    
    agent = A1(
        path="./data/minimax_phase_example",
        llm="MiniMax-M2.5",  # Default model
        source="MiniMax",
        cost_budget=2.0
    )
    
    # Configure specific models for each phase
    agent.configure_phase_llms(
        PLAN=("MiniMax-M2.5", "MiniMax"),      # Use MiniMax for planning
        LEARN=("MiniMax-M2.5", "MiniMax"),     # Use MiniMax for learning
        EXECUTE=("MiniMax-M2.5", "MiniMax"),   # Use MiniMax for execution
        ASSESS=("MiniMax-M2.5", "MiniMax"),    # Use MiniMax for assessment
        SHARE=("MiniMax-M2.5", "MiniMax"),     # Use MiniMax for sharing
    )
    
    # Run a task
    result = agent.run("Design a primer for PCR amplification of the TP53 gene")
    print(f"Result: {result}")


# =============================================================================
# STEP 4: Mixed Model Configuration (MiniMax + Others)
# =============================================================================
def example_mixed_models():
    """Use MiniMax for some phases, other models for others"""
    print("\n" + "="*60)
    print("Example 3: Mixed Model Configuration")
    print("="*60)
    
    agent = A1(
        path="./data/minimax_mixed_example",
        llm="gpt-4o-mini",  # Default to GPT-4o-mini
        source="OpenAI",
        cost_budget=2.0
    )
    
    # Mix MiniMax with other providers
    agent.configure_phase_llms(
        PLAN=("MiniMax-M2.5", "MiniMax"),              # MiniMax for planning
        LEARN=("gemini-2.0-flash-thinking-exp", "Gemini"),  # Gemini for literature review
        EXECUTE=("gpt-4o", "OpenAI"),                   # GPT-4o for code execution
        ASSESS=("claude-sonnet-4-5-20250929", "Anthropic"),  # Claude for assessment
        SHARE=("gpt-4o-mini", "OpenAI"),                # GPT-4o-mini for reports
    )
    
    # Run a complex task
    result = agent.run("Analyze differential gene expression in cancer vs normal tissue")
    print(f"Result: {result}")


# =============================================================================
# STEP 5: Multi-Model Testing Configuration
# =============================================================================
def get_minimax_test_config():
    """
    Configuration for adding MiniMax to multi-model testing
    (for use in bio_52_multi_model_test.py)
    """
    return {
        "name": "MiniMax M2.5",
        "llm": "MiniMax-M2.5",
        "source": "MiniMax",
        "cost_budget": 1.0,
        "phase_llms": {
            "PLAN": ("MiniMax-M2.5", "MiniMax"),
            "LEARN": ("MiniMax-M2.5", "MiniMax"),
            "EXECUTE": ("MiniMax-M2.5", "MiniMax"),
            "ASSESS": ("MiniMax-M2.5", "MiniMax"),
            "SHARE": ("MiniMax-M2.5", "MiniMax"),
        }
    }


# =============================================================================
# Available MiniMax M2.5 Models
# =============================================================================
"""
MiniMax M2.5 Models:
- MiniMax-M2.5           : Peak Performance, Ultimate Value (~60 tps)
- MiniMax-M2.5-highspeed : Same performance, faster (~100 tps)
- MiniMax-M2.1           : Powerful programming capabilities (~60 tps)
- MiniMax-M2.1-highspeed : Faster M2.1 version (~100 tps)
- MiniMax-M2              : Agentic capabilities, Advanced reasoning

Context Length: 204,800 tokens for all models
Temperature Range: (0.0, 1.0], recommended: 1.0

Special Features:
- reasoning_split: Set via extra_body parameter to separate thinking content
- Supports function calling and tool use
- Interleaved thinking capabilities

Check MiniMax documentation for the latest model names and capabilities:
https://platform.minimax.io/docs/api-reference/text-openai-api
"""


# =============================================================================
# Environment Setup Instructions
# =============================================================================
"""
To use MiniMax M2.5:

1. Get API Key:
   - Sign up at https://platform.minimax.io/
   - Generate an API key from your account dashboard

2. Set Environment Variable:
   export MINIMAX_API_KEY="your-api-key-here"
   
   # Or for OpenAI SDK compatibility:
   export OPENAI_BASE_URL=https://api.minimax.io/v1
   export OPENAI_API_KEY="your-api-key-here"

3. Install BioPLEASE (if not already):
   pip install -e .

4. Run this example:
   python examples/minimax_example.py

For official documentation:
https://platform.minimax.io/docs/api-reference/text-openai-api
"""


if __name__ == "__main__":
    # Check if API key is set
    if not os.getenv("MINIMAX_API_KEY"):
        print("⚠️  WARNING: MINIMAX_API_KEY environment variable is not set!")
        print("Please set it before running:")
        print("  export MINIMAX_API_KEY='your-api-key'")
        print()
    
    # Run examples (comment out if you don't have API key set)
    # example_basic_usage()
    # example_phase_specific()
    # example_mixed_models()
    
    # Print the multi-model test configuration
    print("\nTo add MiniMax to multi-model testing, add this to LLM_MODELS:")
    import json
    print(json.dumps(get_minimax_test_config(), indent=2))

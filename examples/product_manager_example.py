"""Example: Using the Product Manager Agent with BioPLE

This example demonstrates how to use the Product Manager agent to monitor
agent progress, track costs, and communicate with users.
"""

import time
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from bioplease.agent.pm_integration import create_pm_enhanced_agent


def main():
    """Main example function"""
    print("=" * 60)
    print("BioPLE Product Manager Agent Example")
    print("=" * 60)
    print()

    # Create agent with Product Manager integrated
    print("1. Creating agent with Product Manager...")
    agent = create_pm_enhanced_agent(
        path="./data",
        llm="gpt-4o-mini",
        source="OpenAI",
        cost_budget=0.50
    )
    print("   ✓ Agent created with PM monitoring enabled")
    print()

    # Configure phase-specific LLMs (optional)
    print("2. Configuring phase-specific LLMs...")
    agent.configure_phase_llms(
        PLAN=("gpt-4o-mini", "OpenAI"),
        LEARN=("gpt-4o-mini", "OpenAI"),
        EXECUTE=("gpt-4o-mini", "OpenAI"),
        ASSESS=("gpt-4o-mini", "OpenAI"),
        SHARE=("gpt-4o-mini", "OpenAI"),
    )
    print("   ✓ Phase LLMs configured")
    print()

    # Get initial progress report
    print("3. Initial Progress Report:")
    print(agent.get_progress_report())
    print()

    # Run a task
    print("4. Executing agent task...")
    print("   Task: Plan a CRISPR screen for T cell exhaustion genes")
    print()

    try:
        # Execute the task
        agent.go(
            "Plan a Novel CRISPR screen to identify genes that regulate T cell exhaustion, "
            "generate 32 genes that maximize the perturbation effect."
        )

        # Wait a moment for logs to be written
        time.sleep(2)

    except Exception as e:
        print(f"   Note: Task execution encountered: {e}")
        print("   (This is expected if running without proper API keys)")

    print()

    # Get progress update
    print("5. Progress Update After Task:")
    print(agent.get_progress_report(detailed=True))
    print()

    # Check budget status
    print("6. Budget Status Check:")
    budget_status = agent.check_budget()
    print(f"   Total Cost: ${budget_status['total_cost']:.4f}")
    if budget_status['budget_limit']:
        print(f"   Budget Limit: ${budget_status['budget_limit']:.2f}")
        print(f"   Budget Remaining: ${budget_status['budget_remaining']:.4f}")
        print(f"   Budget Used: {budget_status['budget_used_percent']:.1f}%")
        if budget_status['warning']:
            print(f"   ⚠️  {budget_status['warning']}")
    print()

    # Add a lesson learned
    print("7. Adding Lesson Learned:")
    agent.add_lesson(
        "CRISPR screen planning requires careful gene selection based on "
        "biological pathways and previous research"
    )
    print("   ✓ Lesson added to PM knowledge base")
    print()

    # Get PM communication message
    print("8. PM Update Message:")
    pm_message = agent.pm_update()
    print(f"   {pm_message.content}")
    print()

    # Save session state
    print("9. Saving PM Session State:")
    if hasattr(agent, 'product_manager'):
        state_file = agent.product_manager.save_session_state()
        print(f"   ✓ Session state saved to: {state_file}")
    print()

    # Interactive mode (optional)
    print("=" * 60)
    print("Interactive Mode (optional)")
    print("Type 'progress' for progress report")
    print("Type 'budget' for budget status")
    print("Type 'exit' to quit")
    print("=" * 60)
    print()

    while True:
        user_input = input("Command: ").strip().lower()

        if user_input == "exit":
            print("Exiting...")
            break

        elif user_input == "progress":
            print(agent.get_progress_report(detailed=True))
            print()

        elif user_input == "budget":
            budget_status = agent.check_budget()
            print(f"Total Cost: ${budget_status['total_cost']:.4f}")
            if budget_status['budget_limit']:
                print(f"Budget Used: {budget_status['budget_used_percent']:.1f}%")
            print()

        elif user_input.startswith("go "):
            task = user_input[3:]
            print(f"Executing: {task}")
            try:
                agent.go(task)
                time.sleep(1)
                print(agent.get_progress_report())
            except Exception as e:
                print(f"Error: {e}")
            print()

        else:
            print("Unknown command. Try 'progress', 'budget', 'go <task>', or 'exit'")
            print()


if __name__ == "__main__":
    main()

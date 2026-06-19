"""
Interactive Product Manager Agent Example

This example demonstrates how to use the enhanced PM agent with:
- Decision tracking
- Interactive queries (synchronous and asynchronous)
- Report generation with graphs
- Non-blocking interaction with the main agent process

The PM agent runs independently and doesn't interfere with the main agent's execution.
"""

import time
from bioplease.agent.pm_integration import create_pm_enhanced_agent
from bioplease.agent.product_manager import DecisionType


def example_synchronous_queries():
    """Example: Using synchronous PM queries"""
    print("=" * 70)
    print("EXAMPLE 1: Synchronous PM Queries")
    print("=" * 70)

    # Create agent with PM integration
    agent = create_pm_enhanced_agent(
        path="./data",
        llm="gpt-4o-mini",
        cost_budget=0.50
    )

    print("\n1. Querying PM for status:")
    status = agent.query_pm("status")
    print(status)

    print("\n2. Querying PM for cost:")
    cost = agent.query_pm("cost")
    print(cost)

    print("\n3. Querying PM for phase:")
    phase = agent.query_pm("phase")
    print(phase)

    print("\n4. Getting help:")
    help_text = agent.query_pm("help")
    print(help_text)


def example_async_queries():
    """Example: Using asynchronous PM queries that don't block execution"""
    print("\n" + "=" * 70)
    print("EXAMPLE 2: Asynchronous PM Queries")
    print("=" * 70)

    # Create agent with PM integration
    agent = create_pm_enhanced_agent(
        path="./data",
        llm="gpt-4o-mini",
        cost_budget=0.50
    )

    # Start the async query listener
    agent.product_manager.start_async_query_listener()

    print("\n1. Sending async queries...")
    agent.ask_pm_async("status")
    agent.ask_pm_async("cost")
    agent.ask_pm_async("decisions")

    # Simulate doing other work
    print("2. Doing other work while PM processes queries...")
    time.sleep(0.5)

    # Retrieve responses
    print("\n3. Retrieving responses:")
    while True:
        response = agent.get_pm_response(timeout=0.1)
        if response is None:
            break
        print(f"\nQ: {response['query']}")
        print(f"A: {response['response']}")
        print(f"Timestamp: {response['timestamp']}")

    # Stop the listener
    agent.product_manager.stop_async_query_listener()


def example_decision_tracking():
    """Example: Tracking agent decisions"""
    print("\n" + "=" * 70)
    print("EXAMPLE 3: Decision Tracking")
    print("=" * 70)

    # Create agent with PM integration
    agent = create_pm_enhanced_agent(
        path="./data",
        llm="gpt-4o-mini",
        cost_budget=0.50
    )

    # Log some decisions (this would normally happen during agent execution)
    print("\n1. Logging decisions...")

    agent.log_decision(
        decision_type=DecisionType.PHASE_TRANSITION,
        description="Transitioning from PLAN to LEARN phase",
        rationale="Planning complete, ready to gather information",
        alternatives=["Stay in PLAN phase", "Skip to EXECUTE"],
        cost_impact=0.002
    )

    agent.log_decision(
        decision_type=DecisionType.TOOL_SELECTION,
        description="Selected PubMed search tool",
        rationale="Most relevant for literature search on protein interactions",
        alternatives=["Google Scholar", "ArXiv", "Manual search"],
        cost_impact=0.001
    )

    agent.log_decision(
        decision_type=DecisionType.MODEL_CHOICE,
        description="Using GPT-4o-mini for LEARN phase",
        rationale="Cost-efficient for information gathering",
        alternatives=["GPT-4o", "Claude Sonnet"],
        cost_impact=-0.05  # Negative = cost savings
    )

    agent.log_decision(
        decision_type=DecisionType.STRATEGY_CHANGE,
        description="Switching to iterative approach",
        rationale="Initial approach not yielding good results",
        alternatives=["Continue with original approach", "Abort and restart"],
        cost_impact=0.01
    )

    # Get decision summary
    print("\n2. Decision Summary:")
    summary = agent.get_decision_summary()
    print(summary)

    # Query decisions
    print("\n3. Querying recent decisions:")
    response = agent.query_pm("decisions")
    print(response)


def example_report_generation():
    """Example: Generating comprehensive reports with graphs"""
    print("\n" + "=" * 70)
    print("EXAMPLE 4: Report Generation with Graphs")
    print("=" * 70)

    # Create agent with PM integration
    agent = create_pm_enhanced_agent(
        path="./data",
        llm="gpt-4o-mini",
        cost_budget=0.50
    )

    # Simulate some progress (normally this happens during agent.go())
    print("\n1. Simulating agent progress...")
    for i in range(5):
        agent.product_manager.current_phase = ["PLAN", "LEARN", "EXECUTE", "ASSESS", "SHARE"][i % 5]
        agent.product_manager.current_iteration = i
        snapshot = agent.product_manager.create_progress_snapshot()
        time.sleep(0.2)  # Simulate time passing

    # Log some decisions
    agent.log_decision(
        DecisionType.PHASE_TRANSITION,
        "Moved to LEARN phase",
        rationale="Planning complete"
    )
    agent.log_decision(
        DecisionType.TOOL_SELECTION,
        "Selected genomics tools",
        rationale="Task requires genomic analysis"
    )

    # Generate comprehensive report
    print("\n2. Generating comprehensive report...")
    report = agent.generate_pm_report(include_graphs=True)

    print(f"\n3. Report generated!")
    print(f"   Text report: {report.get('text_report_path')}")
    print(f"   Metadata: {report.get('metadata_path')}")

    if report.get('graphs'):
        print(f"\n4. Graphs generated:")
        for graph_type, path in report['graphs'].items():
            print(f"   {graph_type}: {path}")
    else:
        print("\n4. No graphs generated (matplotlib may not be installed)")


def example_integration_with_agent_workflow():
    """Example: Integrating PM queries with actual agent workflow"""
    print("\n" + "=" * 70)
    print("EXAMPLE 5: Integration with Agent Workflow")
    print("=" * 70)

    # Create agent with PM integration
    agent = create_pm_enhanced_agent(
        path="./data",
        llm="gpt-4o-mini",
        cost_budget=0.50
    )

    # Start async listener for non-blocking queries
    agent.product_manager.start_async_query_listener()

    print("\n1. Starting agent task (simulated)...")
    print("   Note: In real usage, you would call agent.go(prompt) here")

    # Simulate agent working
    phases = ["PLAN", "LEARN", "EXECUTE", "ASSESS", "SHARE"]

    for i, phase in enumerate(phases):
        print(f"\n   Phase: {phase}")
        agent.product_manager.current_phase = phase
        agent.product_manager.current_iteration = i

        # Log a decision
        agent.log_decision(
            DecisionType.PHASE_TRANSITION,
            f"Entering {phase} phase",
            rationale=f"Completed previous phase successfully"
        )

        # User can query PM at any time without blocking
        agent.ask_pm_async("status")

        # Simulate work
        time.sleep(0.3)

        # Check for PM responses (non-blocking)
        response = agent.get_pm_response(timeout=0.1)
        if response:
            print(f"\n   [PM Update] {response['response']}")

    print("\n2. Agent task complete!")

    # Get final summary
    print("\n3. Final PM Summary:")
    print(agent.query_pm("progress"))

    print("\n4. Decision Summary:")
    print(agent.get_decision_summary())

    # Generate final report
    print("\n5. Generating final report...")
    report = agent.generate_pm_report()
    print(f"   Report saved to: {report.get('text_report_path')}")

    # Stop async listener
    agent.product_manager.stop_async_query_listener()


def main():
    """Run all examples"""
    print("\n" + "=" * 70)
    print("INTERACTIVE PRODUCT MANAGER AGENT EXAMPLES")
    print("=" * 70)

    try:
        # Run examples
        example_synchronous_queries()
        example_async_queries()
        example_decision_tracking()
        example_report_generation()
        example_integration_with_agent_workflow()

        print("\n" + "=" * 70)
        print("All examples completed successfully!")
        print("=" * 70)

    except Exception as e:
        print(f"\nError running examples: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

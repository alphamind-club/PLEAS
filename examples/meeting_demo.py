"""Demo script showing the new meeting and phase logging features

This script demonstrates:
1. Phase-organized logging with individual step files
2. Inter-agent communication via CommunicationHub
3. Multi-agent meetings facilitated by the Product Manager

Usage:
    python examples/meeting_demo.py
"""

import os
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from bioplease.agent.phase_logger import PhaseLogger
from bioplease.agent.communication_hub import CommunicationHub, MessageType, MessagePriority
from bioplease.agent.meeting import MeetingFacilitator
from bioplease.llm import get_llm


def demo_phase_logger():
    """Demonstrate the PhaseLogger functionality"""
    print("\n" + "="*80)
    print("DEMO 1: Phase-Organized Logging")
    print("="*80 + "\n")

    # Create a temporary logs directory
    logs_dir = "./data/demo_logs"
    os.makedirs(logs_dir, exist_ok=True)

    # Initialize phase logger
    logger = PhaseLogger(base_logs_dir=logs_dir, enabled=True)
    print(f"✓ PhaseLogger initialized: {logs_dir}\n")

    # Simulate logging for different phases
    phases = ["PLAN", "LEARN", "EXECUTE", "ASSESS", "SHARE"]

    for phase in phases:
        logger.set_phase(phase)
        print(f"Phase: {phase}")

        # Log a prompt
        prompt_content = f"This is a sample prompt for the {phase} phase. " \
                        f"The agent should analyze and respond appropriately."
        logger.log_prompt(prompt_content, metadata={"model": "gpt-4o", "temperature": 0.7})
        print(f"  ✓ Logged prompt (step {logger.phase_counters[phase]})")

        # Log a response
        response_content = f"This is a sample response from the {phase} phase. " \
                          f"The agent has completed the requested analysis."
        logger.log_response(response_content, metadata={"tokens": 150, "cost": 0.002})
        print(f"  ✓ Logged response (step {logger.phase_counters[phase]})")

        # Show phase summary
        summary = logger.get_phase_summary(phase)
        print(f"  Phase {phase}: {summary['step_count']} steps in {summary['directory']}")
        print()

    # Get overall summary
    print("\nOverall Summary:")
    all_summary = logger.get_all_phases_summary()
    print(f"Total log entries: {all_summary['total_entries']}")
    print(f"Base directory: {all_summary['base_dir']}\n")

    for phase, info in all_summary['phases'].items():
        if info['exists'] and info['step_count'] > 0:
            print(f"  {phase}: {info['step_count']} steps")

    # Export a consolidated log
    print("\n✓ Exporting consolidated PLAN log...")
    export_file = logger.export_phase_log("PLAN")
    print(f"  Exported to: {export_file}")

    # Save manifest
    print("\n✓ Saving manifest...")
    manifest_file = logger.save_manifest()
    print(f"  Manifest saved to: {manifest_file}")

    print("\n" + "="*80 + "\n")


def demo_communication_hub():
    """Demonstrate the CommunicationHub functionality"""
    print("\n" + "="*80)
    print("DEMO 2: Inter-Agent Communication")
    print("="*80 + "\n")

    # Create communication hub
    comm_hub = CommunicationHub(logs_dir="./data/demo_logs/communication")
    print("✓ CommunicationHub initialized\n")

    # Register agents
    agents = [
        ("product_manager", "ProductManager", "Coordinates project activities"),
        ("planner", "ScientificPlanner", "Designs research strategies"),
        ("executor", "TechnicalExecutor", "Implements experiments"),
        ("assessor", "QualityAssessor", "Reviews and validates outputs"),
    ]

    print("Registering agents:")
    for agent_id, agent_type, description in agents:
        comm_hub.register_agent(
            agent_id=agent_id,
            agent_type=agent_type,
            description=description,
            capabilities=[agent_type.lower()]
        )
        print(f"  ✓ {agent_type} ({agent_id})")

    print()

    # Subscribe to topics
    print("Setting up topic subscriptions:")
    comm_hub.subscribe_to_topic("planner", "research_strategy")
    comm_hub.subscribe_to_topic("executor", "research_strategy")
    print("  ✓ planner and executor subscribed to 'research_strategy'")

    comm_hub.subscribe_to_topic("assessor", "quality_review")
    comm_hub.subscribe_to_topic("product_manager", "quality_review")
    print("  ✓ assessor and product_manager subscribed to 'quality_review'")

    print()

    # Send messages
    print("Sending messages:\n")

    # Broadcast message
    msg1 = comm_hub.send_message(
        sender="product_manager",
        content="Starting new research project on protein folding prediction.",
        message_type=MessageType.BROADCAST,
        priority=MessagePriority.HIGH
    )
    print(f"  [{msg1.message_id}] PM broadcasts: {msg1.content[:50]}...")

    # Topic message
    msg2 = comm_hub.send_message(
        sender="planner",
        content="Proposing AlphaFold-based approach with custom training data.",
        message_type=MessageType.TOPIC,
        topic="research_strategy",
        phase_context="PLAN"
    )
    print(f"  [{msg2.message_id}] Planner on 'research_strategy': {msg2.content[:50]}...")

    # Direct message
    msg3 = comm_hub.send_message(
        sender="executor",
        content="Implementation complete. Ready for quality review.",
        message_type=MessageType.DIRECT,
        recipients=["assessor", "product_manager"],
        in_reply_to=msg2.message_id,
        phase_context="EXECUTE"
    )
    print(f"  [{msg3.message_id}] Executor to assessor+PM: {msg3.content[:50]}...")

    print()

    # Query message history
    print("Message history for 'research_strategy' topic:")
    strategy_msgs = comm_hub.get_message_history(topic="research_strategy")
    for msg in strategy_msgs:
        print(f"  - {msg.sender}: {msg.content[:60]}...")

    print()

    # Export communication log
    print("✓ Exporting communication log...")
    export_path = "./data/demo_logs/communication_export.json"
    comm_hub.export_communication_log(export_path)
    print(f"  Exported to: {export_path}")

    print("\n" + "="*80 + "\n")

    return comm_hub


def demo_meeting_facilitator(comm_hub):
    """Demonstrate the MeetingFacilitator functionality"""
    print("\n" + "="*80)
    print("DEMO 3: Multi-Agent Meeting")
    print("="*80 + "\n")

    # Initialize phase logger with some sample data
    logs_dir = "./data/demo_logs"
    logger = PhaseLogger(base_logs_dir=logs_dir, enabled=True)

    # Add some sample phase logs
    logger.set_phase("PLAN")
    logger.log_prompt("What is the best approach for analyzing gene expression data?")
    logger.log_response("We should use differential expression analysis with DESeq2, "
                       "followed by pathway enrichment analysis.")

    logger.set_phase("EXECUTE")
    logger.log_prompt("Implement the DESeq2 analysis pipeline.")
    logger.log_response("Pipeline implemented. Found 523 differentially expressed genes "
                       "with FDR < 0.05. Top pathways enriched: immune response, apoptosis.")

    print("Sample phase logs created for meeting context.\n")

    # Create meeting facilitator
    try:
        facilitator = MeetingFacilitator(
            phase_logger=logger,
            comm_hub=comm_hub,
            llm_factory=get_llm,
            llm_model="gpt-4o-mini"  # Using mini for demo
        )
        print("✓ MeetingFacilitator initialized")
        print(f"  Available personas: {', '.join(facilitator.personas.keys())}\n")

        print("="*80)
        print("NOTE: The following would conduct a real multi-agent meeting using LLMs.")
        print("Each agent persona would discuss the project based on phase logs.")
        print("Skipping actual meeting execution in this demo to save API calls.")
        print("="*80)
        print()

        # To actually run a meeting, uncomment this:
        # results = facilitator.quick_standup(save_transcript=True)
        # print(f"Meeting completed! Transcript: {results['transcript_path']}")

        print("Meeting types available:")
        print("  1. facilitator.quick_standup() - Quick status update")
        print("  2. facilitator.planning_session(focus_area) - Detailed planning")
        print("  3. facilitator.retrospective() - Review and lessons learned")
        print("  4. facilitator.conduct_meeting(topic, prompts) - Custom meeting")

    except Exception as e:
        print(f"Note: Meeting facilitator requires LLM access: {e}")
        print("Set up your API keys to run actual meetings.")

    print("\n" + "="*80 + "\n")


def main():
    """Run all demos"""
    print("\n" + "#"*80)
    print("# BioPLE Phase Logging & Meeting System Demo")
    print("#"*80)

    # Demo 1: Phase Logger
    demo_phase_logger()

    # Demo 2: Communication Hub
    comm_hub = demo_communication_hub()

    # Demo 3: Meeting Facilitator
    demo_meeting_facilitator(comm_hub)

    print("\n" + "#"*80)
    print("# Demo Complete!")
    print("# Check ./data/demo_logs/ for generated files")
    print("#"*80 + "\n")


if __name__ == "__main__":
    main()

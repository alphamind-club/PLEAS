"""Example: Concurrent Product Manager Monitoring

This example demonstrates how to run the Product Manager in a separate thread
to monitor agent progress in real-time while the agent is executing tasks.

This is useful for:
- Long-running agent tasks
- Real-time progress monitoring
- Budget tracking during execution
- User interaction while agent is running
"""

import time
import threading
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from bioplease.agent.pm_integration import create_pm_enhanced_agent


class PMMonitorThread(threading.Thread):
    """Thread for monitoring agent progress in real-time"""

    def __init__(self, agent, update_interval=10, budget_limit=None):
        """Initialize PM monitor thread

        Args:
            agent: A1 agent instance with PM attached
            update_interval: Seconds between progress updates
            budget_limit: Optional budget limit for alerts
        """
        super().__init__(daemon=True)
        self.agent = agent
        self.update_interval = update_interval
        self.budget_limit = budget_limit
        self.running = True
        self.last_cost = 0.0

    def run(self):
        """Run the monitoring loop"""
        print("\n[PM Monitor] Starting real-time monitoring...")
        print(f"[PM Monitor] Update interval: {self.update_interval}s")
        if self.budget_limit:
            print(f"[PM Monitor] Budget limit: ${self.budget_limit:.2f}")
        print("[PM Monitor] Press Ctrl+C to stop\n")

        while self.running:
            try:
                # Create progress snapshot
                if hasattr(self.agent, 'product_manager'):
                    snapshot = self.agent.product_manager.create_progress_snapshot()

                    # Check if there's new activity (cost increased)
                    if snapshot.total_cost > self.last_cost:
                        print(f"\n[PM Monitor] Update at {snapshot.timestamp.strftime('%H:%M:%S')}")
                        print(f"[PM Monitor] Phase: {snapshot.phase}")
                        print(f"[PM Monitor] Cost: ${snapshot.total_cost:.4f} (Δ ${snapshot.total_cost - self.last_cost:.4f})")
                        print(f"[PM Monitor] Time: {self._format_time(snapshot.time_elapsed)}")

                        self.last_cost = snapshot.total_cost

                        # Check budget
                        if self.budget_limit:
                            budget_status = self.agent.check_budget(self.budget_limit)
                            if budget_status['warning']:
                                print(f"[PM Monitor] ⚠️  {budget_status['warning']}")

                        # Show current task if available
                        if snapshot.current_task:
                            task_preview = snapshot.current_task[:100]
                            print(f"[PM Monitor] Task: {task_preview}...")

                time.sleep(self.update_interval)

            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"[PM Monitor] Error: {e}")
                time.sleep(self.update_interval)

    def stop(self):
        """Stop the monitoring loop"""
        self.running = False

    def _format_time(self, seconds):
        """Format seconds to readable time"""
        if seconds < 60:
            return f"{int(seconds)}s"
        elif seconds < 3600:
            return f"{int(seconds // 60)}m {int(seconds % 60)}s"
        else:
            return f"{int(seconds // 3600)}h {int((seconds % 3600) // 60)}m"


def main():
    """Main function with concurrent PM monitoring"""
    print("=" * 60)
    print("BioPLE Product Manager - Concurrent Monitoring Example")
    print("=" * 60)
    print()

    # Create agent with PM
    print("Creating agent with Product Manager...")
    agent = create_pm_enhanced_agent(
        path="./data",
        llm="gpt-4o-mini",
        source="OpenAI",
        cost_budget=0.50
    )
    print("✓ Agent created")
    print()

    # Start PM monitor thread
    print("Starting PM monitor thread...")
    monitor = PMMonitorThread(
        agent=agent,
        update_interval=5,  # Update every 5 seconds
        budget_limit=0.50
    )
    monitor.start()
    print("✓ PM monitor started")
    print()

    try:
        # Run agent tasks
        print("=" * 60)
        print("Executing Agent Tasks")
        print("=" * 60)
        print()

        tasks = [
            "Plan a CRISPR screen to identify genes that regulate T cell exhaustion",
            # Add more tasks as needed
        ]

        for i, task in enumerate(tasks, 1):
            print(f"Task {i}/{len(tasks)}: {task}")
            print()

            try:
                # Execute task (this may take a while)
                agent.go(task)

                # Wait for logs to be written
                time.sleep(2)

                print(f"✓ Task {i} completed")
                print()

            except Exception as e:
                print(f"⚠️  Task {i} error: {e}")
                print()

        # Wait a bit for final monitoring updates
        time.sleep(5)

    except KeyboardInterrupt:
        print("\n\nInterrupted by user")

    finally:
        # Stop monitor
        print("\nStopping PM monitor...")
        monitor.stop()
        monitor.join(timeout=2)
        print("✓ PM monitor stopped")
        print()

        # Final report
        print("=" * 60)
        print("FINAL PROGRESS REPORT")
        print("=" * 60)
        print(agent.get_progress_report(detailed=True))
        print()

        # Save session state
        if hasattr(agent, 'product_manager'):
            state_file = agent.product_manager.save_session_state()
            print(f"Session state saved to: {state_file}")


if __name__ == "__main__":
    main()

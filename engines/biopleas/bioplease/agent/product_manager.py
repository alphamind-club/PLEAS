"""Product Manager Agent for BioPLE

This module implements a Product Manager (PM) agent following the PLEAS framework.
The PM agent monitors agent progress, tracks costs and time, and communicates with users.

Key responsibilities:
- Monitor logs from /root/mywork/BioPLEASE/data/bioplease_data/logs
- Track progress, costs, and time spent
- Provide progress reports to users
- Coordinate with other agents using human messages

Based on the PLEAS AI Agent Framework:
- Plan, Learn, Execute, Assess, Share, Effort (PLEASe)
- A.G.E evaluation: Achievement, Growth, Effort
"""

import os
import glob
import json
import time
import threading
import queue
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum

from langchain_core.messages import HumanMessage, AIMessage

from bioplease.agent.state_manager import StateManager, State, AGEScores as StateAGEScores

# Optional imports for visualization
try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("Warning: matplotlib not available. Visualization features will be disabled.")


class DecisionType(Enum):
    """Types of decisions the agent can make"""
    PHASE_TRANSITION = "phase_transition"
    TOOL_SELECTION = "tool_selection"
    MODEL_CHOICE = "model_choice"
    STRATEGY_CHANGE = "strategy_change"
    RESOURCE_ALLOCATION = "resource_allocation"
    QUALITY_ASSESSMENT = "quality_assessment"
    OTHER = "other"


@dataclass
class DecisionLog:
    """Log entry for agent decisions and choices"""
    timestamp: datetime
    decision_type: DecisionType
    phase: str  # Which phase the decision was made in
    description: str  # What decision was made
    rationale: Optional[str] = None  # Why the decision was made
    alternatives_considered: List[str] = field(default_factory=list)  # Other options
    outcome: Optional[str] = None  # Result of the decision
    cost_impact: float = 0.0  # Cost impact of this decision
    metadata: Dict[str, Any] = field(default_factory=dict)  # Additional context


@dataclass
class ProgressSnapshot:
    """Snapshot of agent progress at a point in time"""
    timestamp: datetime
    phase: str  # PLAN, LEARN, EXECUTE, ASSESS, SHARE
    iteration: int
    total_cost: float
    time_elapsed: float
    status_message: str
    artifacts: List[str] = field(default_factory=list)
    current_task: Optional[str] = None


@dataclass
class AGEScores:
    """Achievement, Growth, Effort scores following PLEAS framework"""
    achievement: float  # Quality of output (1-9 NIH scale)
    growth: float  # Improvement over previous iterations
    effort: float  # Resource utilization efficiency


class ProductManager:
    """Product Manager Agent for monitoring and coordinating agent execution

    The PM agent follows the PLEAS framework principles:
    - Monitors all agents (Planner, Learner, Executor, Assessor, Sharer)
    - Tracks costs and time using CostManager
    - Provides progress reports to users
    - Coordinates state transitions
    - Evaluates agents using A.G.E. metrics

    Attributes:
        logs_dir: Directory containing agent logs
        data_path: Base path for agent data
        cost_manager: Reference to the agent's cost manager
        progress_history: List of progress snapshots
        start_time: Timestamp when agent execution started
    """

    def __init__(
        self,
        logs_dir: str = "/root/mywork/BioPLEASE/data/bioplease_data/logs",
        data_path: str = "./data",
        cost_manager=None,
        specific_log: Optional[str] = None,
        llm_model: str = "gpt-4o-mini",
        llm_source: str = "OpenAI"
    ):
        """Initialize the Product Manager agent

        Args:
            logs_dir: Directory to monitor for logs (or path to specific log file)
            data_path: Base path for data storage
            cost_manager: Reference to CostManager instance from A1 agent
            specific_log: Optional path to a specific log file to analyze
            llm_model: LLM model to use for interactive queries
            llm_source: LLM provider (OpenAI, Anthropic, Gemini)
        """
        # Handle specific log file vs directory
        if specific_log:
            self.specific_log_file = Path(specific_log)
            self.logs_dir = self.specific_log_file.parent
        else:
            logs_path = Path(logs_dir)
            if logs_path.is_file():
                # If logs_dir is actually a file, use it as specific log
                self.specific_log_file = logs_path
                self.logs_dir = logs_path.parent
            else:
                # It's a directory
                self.logs_dir = logs_path
                self.specific_log_file = None
        
        self.data_path = Path(data_path)
        self.cost_manager = cost_manager

        # Create logs directory if it doesn't exist
        os.makedirs(self.logs_dir, exist_ok=True)
        
        # Parse the log file immediately if specific log is provided
        self.log_content: Dict[str, Any] = {}
        if self.specific_log_file and self.specific_log_file.exists():
            self.log_content = self.parse_log_content(self.specific_log_file)

        # Initialize LLM for interactive queries
        self.llm_model = llm_model
        self.llm_source = llm_source
        self.llm = None
        self._initialize_llm()

        # Progress tracking
        self.progress_history: List[ProgressSnapshot] = []
        self.start_time = datetime.now()
        self.current_iteration = 0
        self.current_phase = "INIT"

        # A.G.E. tracking
        self.age_scores_history: List[AGEScores] = []

        # State management
        runs_dir = self.data_path / "runs"
        self.state_manager = StateManager(storage_dir=str(runs_dir))
        self.project_id = "bioplease_project" # Default project ID
        self.phase_counters: Dict[str, int] = {}
        # Configuration: keep only the N most recent iterations per phase in previous_states
        self.max_iterations_per_phase = 1  # Configurable: 1 or 2 recommended
        # Initialize with an INIT state
        self.current_state: Optional[State] = self.state_manager.create_new_state(self.project_id, "INIT", 1)
        self.state_id = self.current_state.state_id
        self.lessons_learned: List[str] = []

        # Decision tracking
        self.decision_history: List[DecisionLog] = []

        # Interactive query interface
        self.query_queue: queue.Queue = queue.Queue()
        self.response_queue: queue.Queue = queue.Queue()
        self._query_thread: Optional[threading.Thread] = None
        self._query_handlers: Dict[str, Callable] = {}
        self._setup_query_handlers()

    def _initialize_llm(self):
        """Initialize the LLM for interactive queries"""
        try:
            from bioplease.llm import get_llm
            self.llm = get_llm(
                model=self.llm_model,
                temperature=0.7,
                source=self.llm_source
            )
            print(f"✓ LLM initialized: {self.llm_model} ({self.llm_source})")
        except Exception as e:
            print(f"Warning: Could not initialize LLM: {e}")
            print("Interactive queries will use fallback pattern matching.")
            self.llm = None

    def set_run_directory(self, run_dir: str):
        """Update the storage directory for state files"""
        self.state_manager.set_storage_dir(run_dir)
        # Re-save current state to the new location
        if self.current_state:
            self.state_manager.save_state(self.current_state)

    def set_original_task(self, task: str):
        """Set the original task in the current state for context"""
        if self.current_state:
            self.current_state.original_task = task
            self.state_manager.save_state(self.current_state)
    
    def set_cost_budget(self, budget: float):
        """Set the cost budget in the current state"""
        if self.current_state:
            self.current_state.cost_budget = budget
            self.state_manager.save_state(self.current_state)

    def start_phase_state(self, phase: str, long_term_memory: str = None) -> State:
        """Start a new state for a specific phase"""
        # Increment counter for this phase
        count = self.phase_counters.get(phase, 0) + 1
        self.phase_counters[phase] = count
        
        # Create new state
        new_state = self.state_manager.create_new_state(self.project_id, phase, count)
        
        # Carry over previous phase outputs as a list
        if self.current_state:
            # Copy previous states list
            new_state.previous_states = list(self.current_state.previous_states)
            
            # Carry over original task
            if self.current_state.original_task:
                new_state.original_task = self.current_state.original_task
            
            # Carry over cost budget
            if self.current_state.cost_budget is not None:
                new_state.cost_budget = self.current_state.cost_budget
            
            # Add latest output from current state if available
            latest_output = self._extract_latest_output_from_state(self.current_state)
            if latest_output:
                new_state.previous_states.append(latest_output)
            
            # Deduplicate: keep only max_iterations_per_phase per phase
            new_state.previous_states = self._deduplicate_previous_states(new_state.previous_states)
        
        # Include long-term memory in the state
        if long_term_memory:
            new_state.long_term_memory = long_term_memory
        elif self.current_state and self.current_state.long_term_memory:
            # Carry over from previous state
            new_state.long_term_memory = self.current_state.long_term_memory
             
        self.current_state = new_state
        self.state_id = new_state.state_id
        self.state_manager.save_state(self.current_state)
        return self.current_state

    def end_phase_state(self, phase: str, output: str, long_term_memory: str = None):
        """Finalize the current state for a phase"""
        if self.current_state and self.current_state.phase == phase:
            self.current_state.timestamp_end = datetime.now().isoformat()
            
            # Store complete output based on phase
            if phase == "PLAN":
                self.current_state.plan_output = output
            elif phase == "LEARN":
                self.current_state.learn_output = output
            elif phase == "EXECUTE":
                self.current_state.execute_output = output
            elif phase == "MINI_SHARE":
                self.current_state.mini_share_output = output
            elif phase == "ASSESS":
                self.current_state.assess_output = output
            elif phase == "SHARE":
                self.current_state.share_output = output
            
            # Update long-term memory if provided
            if long_term_memory:
                self.current_state.long_term_memory = long_term_memory
            
            # Update expense summary with current cost data
            if self.cost_manager:
                self.current_state.expense_summary = self.cost_manager.get_report()
            
            try:
                self.state_manager.save_state(self.current_state)
            except Exception as e:
                print(f"[WARNING] Failed to save product manager state: {e}")
                # Continue execution even if state save fails
    
    def _extract_latest_output_from_state(self, state: State) -> Optional[Dict[str, Any]]:
        """Extract the latest non-None output from a state"""
        # Check outputs in reverse phase order (most recent first)
        if state.share_output:
            return {
                "phase": "SHARE",
                "iteration": state.iteration,
                "output": state.share_output,
                "timestamp": state.timestamp_end or state.timestamp_start
            }
        elif state.assess_output:
            return {
                "phase": "ASSESS",
                "iteration": state.iteration,
                "output": state.assess_output,
                "timestamp": state.timestamp_end or state.timestamp_start
            }
        elif state.mini_share_output:
            return {
                "phase": "MINI_SHARE",
                "iteration": state.iteration,
                "output": state.mini_share_output,
                "timestamp": state.timestamp_end or state.timestamp_start
            }
        elif state.execute_output:
            result = {
                "phase": "EXECUTE",
                "iteration": state.iteration,
                "output": state.execute_output,
                "timestamp": state.timestamp_end or state.timestamp_start
            }
            # Extract observation from execute output if present
            output_str = str(state.execute_output)
            if "<observation>" in output_str and "</observation>" in output_str:
                start_idx = output_str.find("<observation>") + len("<observation>")
                end_idx = output_str.find("</observation>")
                observation = output_str[start_idx:end_idx].strip()
                if observation:
                    result["observation"] = observation
            return result
        elif state.learn_output:
            return {
                "phase": "LEARN",
                "iteration": state.iteration,
                "output": state.learn_output,
                "timestamp": state.timestamp_end or state.timestamp_start
            }
        elif state.plan_output:
            return {
                "phase": "PLAN",
                "iteration": state.iteration,
                "output": state.plan_output,
                "timestamp": state.timestamp_end or state.timestamp_start
            }
        return None
    
    def _deduplicate_previous_states(self, previous_states: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Keep only the N most recent iterations per phase.
        
        Args:
            previous_states: List of previous state outputs
            
        Returns:
            Deduplicated list with max_iterations_per_phase per phase
        """
        from collections import defaultdict
        
        # Group by phase
        phase_outputs = defaultdict(list)
        for output in previous_states:
            phase = output.get('phase')
            if phase:
                phase_outputs[phase].append(output)
        
        # Keep only the latest N iterations per phase
        deduplicated = []
        for phase, outputs in phase_outputs.items():
            # Sort by iteration (descending) to get most recent first
            sorted_outputs = sorted(outputs, key=lambda x: x.get('iteration', 0), reverse=True)
            # Keep only the first max_iterations_per_phase
            deduplicated.extend(sorted_outputs[:self.max_iterations_per_phase])
        
        # Sort the final list by timestamp for chronological order
        deduplicated.sort(key=lambda x: x.get('timestamp', ''))
        
        return deduplicated

    def update_state_from_agent(self, agent_name: str, output: str):
        """Deprecated - use end_phase_state instead"""
        pass

    def transition_to_new_state(self) -> State:
        # Deprecated in favor of start_phase_state / end_phase_state
        return self.start_phase_state("UNKNOWN")

    def get_state_context(self, for_phase: str = None) -> Dict[str, Any]:
        """Get the context for agents based on current state
        
        Args:
            for_phase: If 'EXECUTE', returns a summarized version with only the last phase
                      (ASSESS or LEARN) limited to ~300 tokens. Otherwise returns full state.
        """
        if self.current_state:
            # Update expense summary with current costs before formatting
            if self.cost_manager:
                self.current_state.expense_summary = self.cost_manager.get_report()
            
            # For EXECUTE phase, provide summarized context to reduce context rot
            if for_phase == "EXECUTE":
                formatted_state = self._format_state_for_execute(self.current_state)
            else:
                # Format the full state as structured text
                formatted_state = self._format_state_as_text(self.current_state)
                
            return {
                "state_id": self.current_state.state_id,
                "formatted_state": formatted_state
            }
        return {}
    
    def _format_state_for_execute(self, state: State) -> str:
        """Format a minimal state context for EXECUTE phase to reduce context rot.
        
        Only includes:
        - Original task (truncated if needed)
        - Cost summary
        - Long-term memory (truncated if needed)
        - Last phase output (ASSESS or LEARN), summarized to ~300 tokens
        """
        lines = []
        lines.append(f"=== STATE: {state.phase}.{state.iteration} (EXECUTE Context) ===")
        
        # 1. Original task (truncated to ~100 tokens)
        if state.original_task:
            task = state.original_task
            max_task_chars = 400  # ~100 tokens
            if len(task) > max_task_chars:
                task = task[:max_task_chars] + " [...]"
            lines.append("")
            lines.append("--- Original Task ---")
            lines.append(task)
        
        # 2. Cost summary (compact)
        if state.expense_summary:
            lines.append("")
            lines.append("--- Cost Summary ---")
            total_cost = state.expense_summary.get('total_cost', 0.0)
            total_tokens = state.expense_summary.get('total_tokens', 0)
            
            if state.cost_budget is not None:
                lines.append(f"Budget: ${state.cost_budget:.2f} | Spent: ${total_cost:.4f} | Tokens: {total_tokens:,}")
                if total_cost > state.cost_budget:
                    lines.append("⚠️  WARNING: COST LIMIT EXCEEDED - ROUTE TO SHARE")
            else:
                lines.append(f"Cost: ${total_cost:.4f} | Tokens: {total_tokens:,}")
        
        # 3. Long-term memory (truncated to ~100 tokens)
        if state.long_term_memory:
            memory = state.long_term_memory
            max_memory_chars = 400  # ~100 tokens
            if len(memory) > max_memory_chars:
                memory = memory[:max_memory_chars] + " [...]"
            lines.append("")
            lines.append("--- Long-Term Memory ---")
            lines.append(memory)
        
        # 4. Last phase output (ASSESS or LEARN), summarized to ~300 tokens
        last_phase_output = None
        last_phase_name = None
        
        # Check previous_states for the most recent ASSESS or LEARN
        if state.previous_states:
            for prev_output in reversed(state.previous_states):
                phase = prev_output.get('phase', '')
                if phase in ['ASSESS', 'LEARN']:
                    last_phase_output = prev_output.get('output', '')
                    last_phase_name = phase
                    break
        
        # If not found in previous_states, check current state outputs
        if not last_phase_output:
            if state.assess_output:
                last_phase_output = state.assess_output
                last_phase_name = 'ASSESS'
            elif state.learn_output:
                last_phase_output = state.learn_output
                last_phase_name = 'LEARN'
        
        if last_phase_output and last_phase_name:
            # Truncate to ~300 tokens (1200 chars)
            max_phase_chars = 1200  # ~300 tokens
            if len(last_phase_output) > max_phase_chars:
                # Try to truncate at a sentence boundary
                truncated = last_phase_output[:max_phase_chars]
                last_period = max(truncated.rfind('.'), truncated.rfind('!'), truncated.rfind('?'))
                if last_period > max_phase_chars * 0.7:  # If we can find a sentence boundary in the last 30%
                    last_phase_output = last_phase_output[:last_period + 1] + "\n[... truncated for brevity ...]"
                else:
                    last_phase_output = truncated + "\n[... truncated for brevity ...]"
            
            lines.append("")
            lines.append(f"--- Last Phase Output ({last_phase_name}) ---")
            lines.append(last_phase_output)
        
        lines.append("")
        lines.append("=" * 50)
        return "\n".join(lines)
    
    def _format_state_as_text(self, state: State) -> str:
        """Format a State object as organized, readable text with complete outputs - fully dynamic"""
        from dataclasses import fields
        
        lines = []
        lines.append(f"=== STATE: {state.phase}.{state.iteration} ===")
        
        # Define field groups with custom formatting
        basic_fields = {'state_id', 'project_id', 'phase', 'iteration', 'timestamp_start', 'timestamp_end', 'original_task'}
        phase_output_fields = {'plan_output', 'learn_output', 'execute_output', 'mini_share_output', 'assess_output', 'share_output'}
        special_fields = {'long_term_memory', 'learning_resources', 'rescue_incidents', 'age_scores', 'previous_states', 'expense_summary'}
        
        # Get all fields from the State dataclass
        state_fields = {f.name for f in fields(state)}
        
        # 1. Display basic metadata
        for field_name in basic_fields:
            if field_name in state_fields:
                value = getattr(state, field_name, None)
                if value:
                    display_name = field_name.replace('_', ' ').title()
                    lines.append(f"{display_name}: {value}")
        
        # 1.5 Display original task prominently if it exists
        if state.original_task:
            lines.append("")
            lines.append("=" * 50)
            lines.append("ORIGINAL TASK")
            lines.append("=" * 50)
            lines.append(state.original_task)
            lines.append("=" * 50)
        
        # 1.6 Display cost summary prominently
        if state.expense_summary:
            lines.append("")
            lines.append("=" * 50)
            lines.append("COST SUMMARY")
            lines.append("=" * 50)
            total_cost = state.expense_summary.get('total_cost', 0.0)
            total_tokens = state.expense_summary.get('total_tokens', 0)
            
            # Show budget if available
            if state.cost_budget is not None:
                lines.append(f"Budget: ${state.cost_budget:.2f}")
            
            lines.append(f"Total Cost: ${total_cost:.4f}")
            lines.append(f"Total Tokens: {total_tokens:,}")
            
            # Check if exceeding budget and add warning
            if state.cost_budget is not None and total_cost > state.cost_budget:
                lines.append("")
                lines.append("⚠️  WARNING: COST LIMIT EXCEEDED ⚠️")
                lines.append("ROUTE TO SHARE IF YOU ARE EXCEEDING THE LIMIT")
                lines.append(f"Overage: ${total_cost - state.cost_budget:.4f}")
            
            # Show per-model breakdown if available
            by_model = state.expense_summary.get('by_model', {})
            if by_model:
                lines.append("\nCost by Model:")
                for model, data in by_model.items():
                    model_cost = data.get('cost', 0.0)
                    model_tokens = data.get('total_tokens', 0)
                    lines.append(f"  {model}: ${model_cost:.4f} ({model_tokens:,} tokens)")
            lines.append("=" * 50)
        
        # 2. Display long-term memory
        if state.long_term_memory:
            lines.append("")
            lines.append("--- Long-Term Memory ---")
            lines.append(state.long_term_memory)
        
        # 3. Display phase outputs
        lines.append("")
        lines.append("--- Phase Outputs ---")
        for field_name in phase_output_fields:
            if field_name in state_fields:
                value = getattr(state, field_name, None)
                if value:
                    phase_name = field_name.replace('_output', '').upper()
                    lines.append(f"\n[{phase_name} Output]:\n{value}")
        
        # 4. Display learning resources
        if state.learning_resources:
            lines.append("")
            lines.append("--- Learning Resources ---")
            for i, res in enumerate(state.learning_resources, 1):
                lines.append(f"{i}. {res.title} ({res.type}) - {res.relevance} relevance - {res.status}")
                if res.link:
                    lines.append(f"   Link: {res.link}")
        
        # 5. Display rescue incidents
        if state.rescue_incidents:
            lines.append("")
            lines.append("--- Rescue Incidents ---")
            for i, incident in enumerate(state.rescue_incidents, 1):
                lines.append(f"{i}. {incident.get('description', 'N/A')}")
                lines.append(f"   Action: {incident.get('action', 'N/A')}")
        
        # 6. Display AGE scores
        if state.age_scores:
            lines.append("")
            lines.append("--- AGE Scores ---")
            for agent_name, scores in state.age_scores.items():
                if hasattr(scores, 'achievement'):
                    lines.append(f"[{agent_name}] Achievement: {scores.achievement}, Growth: {scores.growth}, Effort: {scores.effort}")
                else:
                    lines.append(f"[{agent_name}] {scores}")
        
        # 7. Display previous states
        if state.previous_states:
            lines.append("")
            lines.append("--- Previous Phase Outputs ---")
            for i, prev_output in enumerate(state.previous_states, 1):
                phase = prev_output.get('phase', 'UNKNOWN')
                iteration = prev_output.get('iteration', '?')
                timestamp = prev_output.get('timestamp', 'N/A')
                lines.append(f"\n[{i}] {phase}.{iteration} ({timestamp})")
                
                # Dynamically display all fields except the standard ones
                for key, value in prev_output.items():
                    if key not in ['phase', 'iteration', 'timestamp', 'output'] and value:
                        lines.append(f"[{key}]: {value}")
                
                # Always display output last for readability
                output = prev_output.get('output', '')
                if output:
                    lines.append(f"{output}")
        
        # 8. Dynamically display any other fields not covered above
        all_handled = basic_fields | phase_output_fields | special_fields
        remaining_fields = state_fields - all_handled
        
        if remaining_fields:
            lines.append("")
            lines.append("--- Additional Fields ---")
            for field_name in sorted(remaining_fields):
                value = getattr(state, field_name, None)
                if value:
                    # Handle different value types
                    if isinstance(value, (list, tuple)):
                        if value:
                            lines.append(f"\n{field_name.replace('_', ' ').title()}:")
                            for item in value:
                                lines.append(f"  - {item}")
                    elif isinstance(value, dict):
                        if value:
                            lines.append(f"\n{field_name.replace('_', ' ').title()}:")
                            for k, v in value.items():
                                lines.append(f"  {k}: {v}")
                    else:
                        lines.append(f"{field_name.replace('_', ' ').title()}: {value}")
        
        lines.append("="* 50)
        return "\n".join(lines)

    def get_latest_log_file(self) -> Optional[Path]:
        """Get the most recent log file from the logs directory

        Returns:
            Path to the most recent log file, or None if no logs exist
        """
        try:
            log_files = list(self.logs_dir.glob("go_log_*.txt"))
            if not log_files:
                return None
            # Sort by modification time, most recent first
            log_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
            return log_files[0]
        except Exception as e:
            print(f"Error reading log files: {e}")
            return None

    def parse_log_content(self, log_path: Path) -> Dict[str, Any]:
        """Parse log file to extract agent progress information

        Args:
            log_path: Path to the log file

        Returns:
            Dictionary containing parsed log information
        """
        try:
            with open(log_path, 'r', encoding='utf-8') as f:
                content = f.read()

            # Extract phase information
            phases_found = []
            for phase in ["PLAN", "LEARN", "EXECUTE", "ASSESS", "SHARE"]:
                if phase in content:
                    phases_found.append(phase)

            current_phase = phases_found[-1] if phases_found else "UNKNOWN"

            # Extract task information
            task_lines = [line for line in content.split('\n') if 'task' in line.lower() or 'goal' in line.lower()]
            current_task = task_lines[-1] if task_lines else None

            # Extract artifacts (files, results, etc.)
            artifacts = []
            if 'saved' in content.lower() or 'generated' in content.lower():
                artifact_lines = [line for line in content.split('\n')
                                if 'saved' in line.lower() or 'generated' in line.lower()]
                artifacts = artifact_lines[:5]  # Limit to first 5

            return {
                'phase': current_phase,
                'phases_completed': phases_found,
                'current_task': current_task,
                'artifacts': artifacts,
                'log_length': len(content),
                'log_time': datetime.fromtimestamp(log_path.stat().st_mtime)
            }
        except Exception as e:
            print(f"Error parsing log file: {e}")
            return {}

    def update_cost_manager(self, cost_manager):
        """Update the cost manager reference

        This is useful when the A1 agent resets its cost manager
        and we need to update the PM's reference.

        Args:
            cost_manager: New CostManager instance to use
        """
        self.cost_manager = cost_manager

    def get_cost_summary(self) -> Dict[str, Any]:
        """Get current cost summary from the cost manager

        Returns:
            Dictionary with cost information
        """
        if self.cost_manager is None:
            return {
                'total_cost': 0.0,
                'total_tokens': 0,
                'total_time_seconds': 0.0,
                'by_model': {}
            }

        return self.cost_manager.get_report()

    def get_time_elapsed(self) -> float:
        """Get time elapsed since agent started

        Returns:
            Time elapsed in seconds
        """
        return (datetime.now() - self.start_time).total_seconds()

    def create_progress_snapshot(self) -> ProgressSnapshot:
        """Create a snapshot of current agent progress

        Returns:
            ProgressSnapshot object with current state
        """
        # Get latest log information
        latest_log = self.get_latest_log_file()
        log_info = self.parse_log_content(latest_log) if latest_log else {}

        # Get cost information
        cost_summary = self.get_cost_summary()

        # Create snapshot
        snapshot = ProgressSnapshot(
            timestamp=datetime.now(),
            phase=log_info.get('phase', self.current_phase),
            iteration=self.current_iteration,
            total_cost=cost_summary['total_cost'],
            time_elapsed=self.get_time_elapsed(),
            status_message=self._generate_status_message(log_info, cost_summary),
            artifacts=log_info.get('artifacts', []),
            current_task=log_info.get('current_task')
        )

        self.progress_history.append(snapshot)
        return snapshot

    def _generate_status_message(self, log_info: Dict, cost_summary: Dict) -> str:
        """Generate a human-readable status message

        Args:
            log_info: Parsed log information
            cost_summary: Cost summary from cost manager

        Returns:
            Status message string
        """
        phase = log_info.get('phase', 'UNKNOWN')
        phases_completed = log_info.get('phases_completed', [])

        msg = f"Agent is currently in {phase} phase. "
        msg += f"Completed phases: {', '.join(phases_completed)}. "

        if cost_summary['total_cost'] > 0:
            msg += f"Cost: ${cost_summary['total_cost']:.4f}. "

        time_elapsed = self.get_time_elapsed()
        if time_elapsed > 60:
            minutes = int(time_elapsed // 60)
            seconds = int(time_elapsed % 60)
            msg += f"Time elapsed: {minutes}m {seconds}s"
        else:
            msg += f"Time elapsed: {int(time_elapsed)}s"

        return msg

    def get_progress_report(self, detailed: bool = False) -> str:
        """Generate a progress report for the user

        Args:
            detailed: If True, include detailed information

        Returns:
            Formatted progress report string
        """
        snapshot = self.create_progress_snapshot()

        report_lines = [
            "=" * 60,
            "PRODUCT MANAGER PROGRESS REPORT",
            "=" * 60,
            f"Session ID: {self.state_id}",
            f"Timestamp: {snapshot.timestamp.strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "CURRENT STATUS:",
            f"  Phase: {snapshot.phase}",
            f"  Iteration: {snapshot.iteration}",
            "",
            "RESOURCE USAGE:",
            f"  Total Cost: ${snapshot.total_cost:.4f}",
            f"  Time Elapsed: {self._format_time(snapshot.time_elapsed)}",
            "",
            snapshot.status_message,
        ]

        if snapshot.current_task:
            report_lines.extend([
                "",
                "CURRENT TASK:",
                f"  {snapshot.current_task[:200]}..."  # Truncate long tasks
            ])

        if snapshot.artifacts:
            report_lines.extend([
                "",
                "RECENT ARTIFACTS:",
            ])
            for artifact in snapshot.artifacts[:5]:
                report_lines.append(f"  - {artifact[:100]}")  # Truncate long paths

        if detailed and self.cost_manager:
            cost_summary = self.get_cost_summary()
            report_lines.extend([
                "",
                "DETAILED COST BREAKDOWN:",
            ])
            for model, stats in cost_summary['by_model'].items():
                report_lines.append(
                    f"  {model}: ${stats['cost']:.4f} "
                    f"({stats['input_tokens']} in + {stats['output_tokens']} out tokens, "
                    f"{stats['calls']} calls)"
                )

        if self.lessons_learned:
            report_lines.extend([
                "",
                "LESSONS LEARNED:",
            ])
            for lesson in self.lessons_learned[-3:]:  # Show last 3 lessons
                report_lines.append(f"  - {lesson}")

        report_lines.append("=" * 60)

        return "\n".join(report_lines)

    def _format_time(self, seconds: float) -> str:
        """Format seconds into human-readable time

        Args:
            seconds: Time in seconds

        Returns:
            Formatted time string
        """
        if seconds < 60:
            return f"{int(seconds)}s"
        elif seconds < 3600:
            minutes = int(seconds // 60)
            secs = int(seconds % 60)
            return f"{minutes}m {secs}s"
        else:
            hours = int(seconds // 3600)
            minutes = int((seconds % 3600) // 60)
            return f"{hours}h {minutes}m"

    def communicate_with_user(self, message: str) -> HumanMessage:
        """Create a human message for communication with users

        Following the PLEAS framework, use human messages for agent communication.

        Args:
            message: Message content to send to user

        Returns:
            HumanMessage object
        """
        return HumanMessage(content=f"[PM Agent] {message}")

    def check_budget_status(self, budget_limit: Optional[float] = None) -> Dict[str, Any]:
        """Check if the agent is within budget constraints

        Args:
            budget_limit: Maximum allowed cost (optional)

        Returns:
            Dictionary with budget status information
        """
        cost_summary = self.get_cost_summary()
        total_cost = cost_summary['total_cost']

        status = {
            'total_cost': total_cost,
            'budget_limit': budget_limit,
            'within_budget': True,
            'budget_remaining': None,
            'budget_used_percent': None,
            'warning': None
        }

        if budget_limit is not None:
            status['within_budget'] = total_cost <= budget_limit
            status['budget_remaining'] = budget_limit - total_cost
            status['budget_used_percent'] = (total_cost / budget_limit) * 100

            if status['budget_used_percent'] >= 90:
                status['warning'] = "CRITICAL: Budget 90% exhausted!"
            elif status['budget_used_percent'] >= 75:
                status['warning'] = "WARNING: Budget 75% used"
            elif status['budget_used_percent'] >= 50:
                status['warning'] = "INFO: Budget 50% used"

        return status

    def add_lesson_learned(self, lesson: str) -> None:
        """Add a lesson learned for future reference

        Following PLEAS framework's emphasis on lessons learned.

        Args:
            lesson: Lesson description
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.lessons_learned.append(f"[{timestamp}] {lesson}")

    def calculate_age_score(
        self,
        quality: float,
        previous_quality: Optional[float] = None,
        effort_ratio: float = 1.0
    ) -> AGEScores:
        """Calculate A.G.E. (Achievement, Growth, Effort) scores

        Following the PLEAS framework's A.G.E. evaluation system:
        - Achievement: Quality + Quantity (NIH 1-9 scale)
        - Growth: Improvement over previous iterations
        - Effort: Resource utilization efficiency

        Args:
            quality: Current quality score (1-9)
            previous_quality: Previous iteration's quality score
            effort_ratio: Actual effort / expected effort

        Returns:
            AGEScores object
        """
        # Achievement score (quality-based)
        achievement = quality

        # Growth score (improvement)
        if previous_quality is not None:
            growth = ((quality - previous_quality) / 9.0) * 100  # Percentage improvement
        else:
            growth = 0.0

        # Effort score (efficiency)
        # Lower effort_ratio = more efficient (higher score)
        # effort_ratio of 1.0 = exactly as expected (score 5)
        # effort_ratio < 1.0 = more efficient (score > 5)
        # effort_ratio > 1.0 = less efficient (score < 5)
        if effort_ratio <= 1.0:
            effort = 5 + (1.0 - effort_ratio) * 4  # Scale from 5 to 9
        else:
            effort = max(1, 5 - (effort_ratio - 1.0) * 4)  # Scale from 5 to 1

        scores = AGEScores(
            achievement=achievement,
            growth=growth,
            effort=effort
        )

        self.age_scores_history.append(scores)
        return scores

    def generate_dashboard_data(self) -> Dict[str, Any]:
        """Generate data for dashboard visualization

        Returns:
            Dictionary with dashboard data
        """
        return {
            'session_id': self.state_id,
            'start_time': self.start_time.isoformat(),
            'current_phase': self.current_phase,
            'current_iteration': self.current_iteration,
            'progress_history': [
                {
                    'timestamp': s.timestamp.isoformat(),
                    'phase': s.phase,
                    'cost': s.total_cost,
                    'time': s.time_elapsed,
                    'status': s.status_message
                }
                for s in self.progress_history
            ],
            'cost_summary': self.get_cost_summary(),
            'lessons_learned': self.lessons_learned,
            'age_scores': [
                {
                    'achievement': s.achievement,
                    'growth': s.growth,
                    'effort': s.effort
                }
                for s in self.age_scores_history
            ]
        }

    def save_session_state(self, output_dir: Optional[Path] = None) -> Path:
        """Save current PM session state to file

        Args:
            output_dir: Directory to save state (defaults to logs_dir)

        Returns:
            Path to saved state file
        """
        if output_dir is None:
            output_dir = self.logs_dir

        output_dir = Path(output_dir)
        os.makedirs(output_dir, exist_ok=True)

        state_file = output_dir / f"pm_state_{self.state_id}.json"

        dashboard_data = self.generate_dashboard_data()

        with open(state_file, 'w', encoding='utf-8') as f:
            json.dump(dashboard_data, f, indent=2)

        return state_file

    # =========================================================================
    # DECISION TRACKING
    # =========================================================================

    def log_decision(
        self,
        decision_type: DecisionType,
        description: str,
        rationale: Optional[str] = None,
        alternatives: Optional[List[str]] = None,
        cost_impact: float = 0.0,
        **metadata
    ) -> DecisionLog:
        """Log an agent decision for tracking and reporting

        Args:
            decision_type: Type of decision made
            description: Description of the decision
            rationale: Why this decision was made
            alternatives: Other options that were considered
            cost_impact: Estimated cost impact
            **metadata: Additional context

        Returns:
            DecisionLog entry
        """
        decision = DecisionLog(
            timestamp=datetime.now(),
            decision_type=decision_type,
            phase=self.current_phase,
            description=description,
            rationale=rationale,
            alternatives_considered=alternatives or [],
            cost_impact=cost_impact,
            metadata=metadata
        )
        self.decision_history.append(decision)
        return decision

    def get_decisions_by_phase(self, phase: str) -> List[DecisionLog]:
        """Get all decisions made in a specific phase

        Args:
            phase: Phase name (PLAN, LEARN, EXECUTE, ASSESS, SHARE)

        Returns:
            List of DecisionLog entries for that phase
        """
        return [d for d in self.decision_history if d.phase == phase]

    def get_decisions_by_type(self, decision_type: DecisionType) -> List[DecisionLog]:
        """Get all decisions of a specific type

        Args:
            decision_type: Type of decision to filter by

        Returns:
            List of DecisionLog entries of that type
        """
        return [d for d in self.decision_history if d.decision_type == decision_type]

    def get_decision_summary(self) -> str:
        """Generate a summary of all decisions made

        Returns:
            Formatted decision summary string
        """
        if not self.decision_history:
            return "No decisions logged yet."

        lines = [
            "=" * 60,
            "DECISION SUMMARY",
            "=" * 60,
            f"Total Decisions: {len(self.decision_history)}",
            ""
        ]

        # Group by phase
        by_phase = {}
        for decision in self.decision_history:
            if decision.phase not in by_phase:
                by_phase[decision.phase] = []
            by_phase[decision.phase].append(decision)

        for phase, decisions in by_phase.items():
            lines.append(f"\n{phase} Phase ({len(decisions)} decisions):")
            for i, decision in enumerate(decisions[-5:], 1):  # Show last 5 per phase
                lines.append(f"  {i}. [{decision.decision_type.value}] {decision.description}")
                if decision.rationale:
                    lines.append(f"     Rationale: {decision.rationale[:100]}")
                if decision.alternatives_considered:
                    lines.append(f"     Alternatives: {', '.join(decision.alternatives_considered[:3])}")

        lines.append("=" * 60)
        return "\n".join(lines)

    # =========================================================================
    # VISUALIZATION & REPORTING
    # =========================================================================

    def generate_cost_graph(self, output_path: Optional[Path] = None) -> Optional[Path]:
        """Generate a graph showing cost over time

        Args:
            output_path: Where to save the graph (optional)

        Returns:
            Path to saved graph, or None if matplotlib unavailable
        """
        if not MATPLOTLIB_AVAILABLE:
            print("Matplotlib not available. Cannot generate graph.")
            return None

        if not self.progress_history:
            print("No progress data available for graphing.")
            return None

        # Extract data
        timestamps = [s.timestamp for s in self.progress_history]
        costs = [s.total_cost for s in self.progress_history]

        # Create figure
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(timestamps, costs, marker='o', linestyle='-', linewidth=2, markersize=6)

        ax.set_xlabel('Time', fontsize=12)
        ax.set_ylabel('Total Cost ($)', fontsize=12)
        ax.set_title('Agent Cost Over Time', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)

        # Format x-axis
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
        plt.xticks(rotation=45)

        plt.tight_layout()

        # Save figure
        if output_path is None:
            output_path = self.logs_dir / f"cost_graph_{self.state_id}.png"

        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

        return output_path

    def generate_phase_timeline_graph(self, output_path: Optional[Path] = None) -> Optional[Path]:
        """Generate a timeline graph showing phase progression

        Args:
            output_path: Where to save the graph (optional)

        Returns:
            Path to saved graph, or None if matplotlib unavailable
        """
        if not MATPLOTLIB_AVAILABLE:
            print("Matplotlib not available. Cannot generate graph.")
            return None

        if not self.progress_history:
            print("No progress data available for graphing.")
            return None

        # Extract phase data
        phase_colors = {
            'INIT': '#808080',
            'PLAN': '#4285F4',
            'LEARN': '#34A853',
            'EXECUTE': '#FBBC04',
            'ASSESS': '#EA4335',
            'SHARE': '#9C27B0'
        }

        timestamps = [s.timestamp for s in self.progress_history]
        phases = [s.phase for s in self.progress_history]
        colors = [phase_colors.get(p, '#808080') for p in phases]

        # Create figure
        fig, ax = plt.subplots(figsize=(12, 4))

        # Plot phase transitions
        for i, (time, phase, color) in enumerate(zip(timestamps, phases, colors)):
            ax.scatter(time, 0, s=200, c=color, marker='s', zorder=2, edgecolors='black', linewidth=1.5)
            if i == 0 or phases[i] != phases[i-1]:
                ax.text(time, 0.15, phase, ha='center', va='bottom', fontsize=9, fontweight='bold')

        ax.set_ylim(-0.5, 0.5)
        ax.set_xlabel('Time', fontsize=12)
        ax.set_title('Agent Phase Timeline', fontsize=14, fontweight='bold')
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
        ax.yaxis.set_visible(False)
        plt.xticks(rotation=45)

        # Add legend
        legend_elements = [plt.Line2D([0], [0], marker='s', color='w', label=phase,
                                     markerfacecolor=color, markersize=10, markeredgecolor='black')
                          for phase, color in phase_colors.items() if phase in phases]
        ax.legend(handles=legend_elements, loc='upper right', ncol=len(legend_elements))

        plt.tight_layout()

        # Save figure
        if output_path is None:
            output_path = self.logs_dir / f"phase_timeline_{self.state_id}.png"

        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

        return output_path

    def generate_decision_breakdown_graph(self, output_path: Optional[Path] = None) -> Optional[Path]:
        """Generate a pie chart showing decision type breakdown

        Args:
            output_path: Where to save the graph (optional)

        Returns:
            Path to saved graph, or None if matplotlib unavailable
        """
        if not MATPLOTLIB_AVAILABLE:
            print("Matplotlib not available. Cannot generate graph.")
            return None

        if not self.decision_history:
            print("No decision data available for graphing.")
            return None

        # Count decisions by type
        decision_counts = {}
        for decision in self.decision_history:
            dtype = decision.decision_type.value
            decision_counts[dtype] = decision_counts.get(dtype, 0) + 1

        # Create figure
        fig, ax = plt.subplots(figsize=(10, 8))

        labels = list(decision_counts.keys())
        sizes = list(decision_counts.values())
        colors = plt.cm.Set3(range(len(labels)))

        wedges, texts, autotexts = ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%',
                                           startangle=90, textprops={'fontsize': 10})

        # Make percentage text bold
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')

        ax.set_title('Decision Types Breakdown', fontsize=14, fontweight='bold')

        plt.tight_layout()

        # Save figure
        if output_path is None:
            output_path = self.logs_dir / f"decision_breakdown_{self.state_id}.png"

        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

        return output_path

    def generate_comprehensive_report(
        self,
        output_dir: Optional[Path] = None,
        include_graphs: bool = True
    ) -> Dict[str, Any]:
        """Generate a comprehensive PM report with text and visualizations

        Args:
            output_dir: Directory to save report files
            include_graphs: Whether to generate visualization graphs

        Returns:
            Dictionary containing report data and file paths
        """
        if output_dir is None:
            output_dir = self.logs_dir / f"report_{self.state_id}"

        output_dir = Path(output_dir)
        os.makedirs(output_dir, exist_ok=True)

        report_data = {
            'session_id': self.state_id,
            'generated_at': datetime.now().isoformat(),
            'text_report': self.get_progress_report(detailed=True),
            'decision_summary': self.get_decision_summary(),
            'graphs': {}
        }

        # Save text report
        text_report_path = output_dir / "progress_report.txt"
        with open(text_report_path, 'w', encoding='utf-8') as f:
            f.write(report_data['text_report'])
            f.write("\n\n")
            f.write(report_data['decision_summary'])

        report_data['text_report_path'] = str(text_report_path)

        # Generate graphs if requested
        if include_graphs and MATPLOTLIB_AVAILABLE:
            cost_graph = self.generate_cost_graph(output_dir / "cost_over_time.png")
            if cost_graph:
                report_data['graphs']['cost'] = str(cost_graph)

            timeline_graph = self.generate_phase_timeline_graph(output_dir / "phase_timeline.png")
            if timeline_graph:
                report_data['graphs']['timeline'] = str(timeline_graph)

            decision_graph = self.generate_decision_breakdown_graph(output_dir / "decision_breakdown.png")
            if decision_graph:
                report_data['graphs']['decisions'] = str(decision_graph)

        # Save report metadata
        metadata_path = output_dir / "report_metadata.json"
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(report_data, f, indent=2)

        report_data['metadata_path'] = str(metadata_path)

        return report_data

    # =========================================================================
    # INTERACTIVE QUERY INTERFACE
    # =========================================================================

    def _setup_query_handlers(self):
        """Setup handlers for different query types"""
        self._query_handlers = {
            'status': self._handle_status_query,
            'progress': self._handle_progress_query,
            'cost': self._handle_cost_query,
            'decisions': self._handle_decisions_query,
            'phase': self._handle_phase_query,
            'help': self._handle_help_query,
        }

    def _handle_status_query(self, query: str) -> str:
        """Handle status query"""
        snapshot = self.create_progress_snapshot()
        return snapshot.status_message

    def _handle_progress_query(self, query: str) -> str:
        """Handle progress query"""
        return self.get_progress_report(detailed=False)

    def _handle_cost_query(self, query: str) -> str:
        """Handle cost query"""
        cost_summary = self.get_cost_summary()
        lines = [
            f"Total Cost: ${cost_summary['total_cost']:.4f}",
            f"Total Tokens: {cost_summary['total_tokens']}",
            f"Time: {self._format_time(cost_summary['total_time_seconds'])}",
            ""
        ]
        if cost_summary['by_model']:
            lines.append("By Model:")
            for model, stats in cost_summary['by_model'].items():
                lines.append(f"  {model}: ${stats['cost']:.4f} ({stats['calls']} calls)")
        return "\n".join(lines)

    def _handle_decisions_query(self, query: str) -> str:
        """Handle decisions query"""
        if not self.decision_history:
            return "No decisions have been logged yet."

        # Get recent decisions
        recent = self.decision_history[-5:]
        lines = [f"Recent Decisions (last {len(recent)}):"]
        for i, decision in enumerate(recent, 1):
            lines.append(f"{i}. [{decision.phase}] {decision.description}")
            if decision.rationale:
                lines.append(f"   Rationale: {decision.rationale[:80]}...")

        return "\n".join(lines)

    def _handle_phase_query(self, query: str) -> str:
        """Handle phase query"""
        return f"Current Phase: {self.current_phase} (Iteration: {self.current_iteration})"

    def _handle_help_query(self, query: str) -> str:
        """Handle help query"""
        return """
Available PM Queries:
  - status: Get current status message
  - progress: Get progress report
  - cost: Get cost breakdown
  - decisions: See recent decisions
  - phase: Get current phase
  - help: Show this help message
"""

    def query(self, question: str) -> str:
        """Query the PM agent interactively (synchronous)

        This is a non-blocking query interface that doesn't interfere with
        the main agent process.

        Args:
            question: Query string (e.g., "status", "cost", "decisions")

        Returns:
            Response string
        """
        question_lower = question.lower().strip()

        # Check for registered handlers
        for keyword, handler in self._query_handlers.items():
            if keyword in question_lower:
                return handler(question)

        # Default: treat as general query
        return self._handle_status_query(question)

    def start_async_query_listener(self):
        """Start background thread for async query processing

        This allows queries to be processed without blocking the main agent.
        """
        if self._query_thread is not None and self._query_thread.is_alive():
            return  # Already running

        def query_worker():
            while True:
                try:
                    # Check for query with timeout
                    query = self.query_queue.get(timeout=1.0)
                    if query is None:  # Sentinel value to stop
                        break

                    # Process query
                    response = self.query(query)

                    # Put response in queue
                    self.response_queue.put({
                        'query': query,
                        'response': response,
                        'timestamp': datetime.now().isoformat()
                    })

                    self.query_queue.task_done()
                except queue.Empty:
                    continue
                except Exception as e:
                    self.response_queue.put({
                        'query': query if 'query' in locals() else 'unknown',
                        'response': f"Error processing query: {str(e)}",
                        'timestamp': datetime.now().isoformat()
                    })

        self._query_thread = threading.Thread(target=query_worker, daemon=True)
        self._query_thread.start()

    def ask_async(self, question: str):
        """Ask a question asynchronously (non-blocking)

        Args:
            question: Query string
        """
        if self._query_thread is None:
            self.start_async_query_listener()

        self.query_queue.put(question)

    def get_async_response(self, timeout: float = 0.1) -> Optional[Dict[str, Any]]:
        """Get response from async query queue

        Args:
            timeout: How long to wait for response

        Returns:
            Response dictionary or None if no response available
        """
        try:
            return self.response_queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def stop_async_query_listener(self):
        """Stop the async query listener thread"""
        if self._query_thread is not None and self._query_thread.is_alive():
            self.query_queue.put(None)  # Sentinel to stop
            self._query_thread.join(timeout=2.0)

    # =========================================================================
    # ENHANCED ANALYSIS & INTERACTIVE FEATURES
    # =========================================================================

    def set_log_file(self, log_path: str):
        """Set or change the specific log file to analyze
        
        Args:
            log_path: Path to the log file to analyze
        """
        self.specific_log_file = Path(log_path)
        if self.specific_log_file.exists():
            self.logs_dir = self.specific_log_file.parent
            self.log_content = self.parse_log_content(self.specific_log_file)
            print(f"✓ Now analyzing: {self.specific_log_file.name}")
        else:
            print(f"✗ Log file not found: {log_path}")

    def get_detailed_analysis(self) -> str:
        """Get comprehensive detailed analysis of the log file
        
        Returns:
            Detailed analysis report as formatted string
        """
        log_to_analyze = self.specific_log_file or self.get_latest_log_file()
        
        if not log_to_analyze or not log_to_analyze.exists():
            return "No log file available for analysis."
        
        # Parse log content
        log_info = self.parse_log_content(log_to_analyze)
        
        # Read full content for detailed analysis
        with open(log_to_analyze, 'r', encoding='utf-8') as f:
            full_content = f.read()
        
        lines = full_content.split('\n')
        
        # Extract detailed statistics
        error_lines = [l for l in lines if 'error' in l.lower() or 'exception' in l.lower()]
        warning_lines = [l for l in lines if 'warning' in l.lower() or 'warn' in l.lower()]
        tool_calls = [l for l in lines if 'tool' in l.lower() or 'function' in l.lower()]
        llm_calls = [l for l in lines if 'llm' in l.lower() or 'model' in l.lower()]
        
        # Phase timing analysis
        phase_mentions = {}
        for phase in ["PLAN", "LEARN", "EXECUTE", "ASSESS", "SHARE"]:
            count = full_content.count(phase)
            if count > 0:
                phase_mentions[phase] = count
        
        # Build detailed report
        report = [
            "=" * 80,
            "DETAILED LOG ANALYSIS",
            "=" * 80,
            f"Log File: {log_to_analyze.name}",
            f"Log Path: {log_to_analyze}",
            f"File Size: {log_to_analyze.stat().st_size / 1024:.2f} KB",
            f"Last Modified: {datetime.fromtimestamp(log_to_analyze.stat().st_mtime).strftime('%Y-%m-%d %H:%M:%S')}",
            f"Total Lines: {len(lines)}",
            "",
            "PHASE ANALYSIS:",
            f"  Phases Completed: {', '.join(log_info.get('phases_completed', ['None']))}",
            f"  Current Phase: {log_info.get('phase', 'UNKNOWN')}",
        ]
        
        if phase_mentions:
            report.append("\n  Phase Mentions:")
            for phase, count in sorted(phase_mentions.items(), key=lambda x: x[1], reverse=True):
                report.append(f"    {phase}: {count} times")
        
        report.extend([
            "",
            "ACTIVITY SUMMARY:",
            f"  Tool/Function Calls: {len(tool_calls)}",
            f"  LLM Interactions: {len(llm_calls)}",
            f"  Errors Found: {len(error_lines)}",
            f"  Warnings Found: {len(warning_lines)}",
        ])
        
        if error_lines:
            report.append("\n  Recent Errors:")
            for error in error_lines[-5:]:
                report.append(f"    • {error.strip()[:100]}")
        
        if warning_lines:
            report.append("\n  Recent Warnings:")
            for warning in warning_lines[-5:]:
                report.append(f"    • {warning.strip()[:100]}")
        
        report.extend([
            "",
            "TASK INFORMATION:",
        ])
        
        task = log_info.get('current_task')
        if task:
            report.append(f"  Current Task: {task.strip()[:200]}")
        else:
            report.append("  Current Task: Not detected")
        
        artifacts = log_info.get('artifacts', [])
        if artifacts:
            report.append("\n  Artifacts Generated:")
            for artifact in artifacts:
                report.append(f"    • {artifact.strip()[:100]}")
        
        # Search for key decisions
        decision_keywords = ['decided', 'choosing', 'selected', 'determined', 'concluded']
        decisions = []
        for line in lines:
            line_lower = line.lower()
            if any(keyword in line_lower for keyword in decision_keywords):
                decisions.append(line.strip())
        
        if decisions:
            report.append("\n  Key Decisions Detected:")
            for decision in decisions[-10:]:  # Last 10 decisions
                report.append(f"    • {decision[:150]}")
        
        report.append("=" * 80)
        
        return "\n".join(report)

    def search_log(self, search_term: str, context_lines: int = 2) -> str:
        """Search the log file for specific terms and show context
        
        Args:
            search_term: Term to search for (case-insensitive)
            context_lines: Number of lines to show before/after match
            
        Returns:
            Formatted search results
        """
        log_to_search = self.specific_log_file or self.get_latest_log_file()
        
        if not log_to_search or not log_to_search.exists():
            return "No log file available for search."
        
        with open(log_to_search, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        matches = []
        for i, line in enumerate(lines):
            if search_term.lower() in line.lower():
                start = max(0, i - context_lines)
                end = min(len(lines), i + context_lines + 1)
                context = {
                    'line_num': i + 1,
                    'match': line.strip(),
                    'context_before': [lines[j].rstrip() for j in range(start, i)],
                    'context_after': [lines[j].rstrip() for j in range(i + 1, end)]
                }
                matches.append(context)
        
        if not matches:
            return f"No matches found for '{search_term}'"
        
        result = [
            f"Found {len(matches)} match(es) for '{search_term}':",
            "=" * 80
        ]
        
        for idx, match in enumerate(matches[:20], 1):  # Limit to 20 matches
            result.append(f"\nMatch {idx} (Line {match['line_num']}):")
            
            if match['context_before']:
                for line in match['context_before'][-context_lines:]:
                    result.append(f"  {line}")
            
            result.append(f"→ {match['match']}")
            
            if match['context_after']:
                for line in match['context_after'][:context_lines]:
                    result.append(f"  {line}")
        
        if len(matches) > 20:
            result.append(f"\n... and {len(matches) - 20} more matches")
        
        return "\n".join(result)

    def get_phase_details(self, phase: str) -> str:
        """Get detailed information about a specific phase
        
        Args:
            phase: Phase name (PLAN, LEARN, EXECUTE, ASSESS, SHARE)
            
        Returns:
            Detailed phase analysis
        """
        log_to_analyze = self.specific_log_file or self.get_latest_log_file()
        
        if not log_to_analyze or not log_to_analyze.exists():
            return f"No log file available for {phase} analysis."
        
        phase = phase.upper()
        if phase not in ["PLAN", "LEARN", "EXECUTE", "ASSESS", "SHARE"]:
            return f"Invalid phase '{phase}'. Must be one of: PLAN, LEARN, EXECUTE, ASSESS, SHARE"
        
        with open(log_to_analyze, 'r', encoding='utf-8') as f:
            content = f.read()
        
        if phase not in content:
            return f"Phase '{phase}' not found in log file."
        
        lines = content.split('\n')
        phase_lines = []
        capturing = False
        
        for line in lines:
            if phase in line:
                capturing = True
            elif capturing and any(p in line for p in ["PLAN", "LEARN", "EXECUTE", "ASSESS", "SHARE"] if p != phase):
                break
            if capturing:
                phase_lines.append(line)
        
        result = [
            "=" * 80,
            f"{phase} PHASE DETAILS",
            "=" * 80,
            f"Total lines in phase: {len(phase_lines)}",
            ""
        ]
        
        # Extract key information
        errors = [l for l in phase_lines if 'error' in l.lower()]
        tools = [l for l in phase_lines if 'tool' in l.lower() or 'function' in l.lower()]
        decisions = [l for l in phase_lines if any(k in l.lower() for k in ['decided', 'choosing', 'selected'])]
        
        result.append(f"Errors: {len(errors)}")
        result.append(f"Tool calls: {len(tools)}")
        result.append(f"Decisions: {len(decisions)}")
        result.append("")
        
        if phase_lines:
            result.append("Phase Content Preview (first 50 lines):")
            result.append("-" * 80)
            for line in phase_lines[:50]:
                result.append(line.rstrip())
        
        return "\n".join(result)

    def list_available_logs(self) -> str:
        """List all available log files in the logs directory
        
        Returns:
            Formatted list of log files
        """
        try:
            log_files = list(self.logs_dir.glob("go_log_*.txt"))
            if not log_files:
                return f"No log files found in {self.logs_dir}"
            
            log_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
            
            result = [
                "=" * 80,
                "AVAILABLE LOG FILES",
                "=" * 80,
                f"Directory: {self.logs_dir}",
                f"Total Files: {len(log_files)}",
                ""
            ]
            
            for idx, log_file in enumerate(log_files, 1):
                stat = log_file.stat()
                size_kb = stat.st_size / 1024
                mod_time = datetime.fromtimestamp(stat.st_mtime).strftime('%Y-%m-%d %H:%M:%S')
                current = "←" if log_file == self.specific_log_file else " "
                
                result.append(f"{current}{idx:2}. {log_file.name}")
                result.append(f"    Size: {size_kb:.2f} KB | Modified: {mod_time}")
            
            result.append("")
            result.append("Use set_log_file(path) to analyze a specific log.")
            result.append("=" * 80)
            
            return "\n".join(result)
        except Exception as e:
            return f"Error listing log files: {e}"

    def ask(self, question: str) -> str:
        """Ask the PM agent a question about the logs (enhanced version)
        
        Args:
            question: Natural language question
            
        Returns:
            Answer based on log analysis
        """
        question_lower = question.lower().strip()
        
        # Enhanced query routing with priority order
        # NOTE: Use specific commands for structured data, LLM for natural questions
        
        # 1. List logs (exact match)
        if question_lower in ['list', 'show logs', 'available logs', 'all logs', 'list logs']:
            return self.list_available_logs()
        
        # 2. Detailed analysis (exact match)
        if question_lower in ['detailed', 'analysis', 'comprehensive', 'full report', 'detailed analysis']:
            return self.get_detailed_analysis()
        
        # 3. Search functionality (must have "search" or "grep" at start)
        if question_lower.startswith('search ') or question_lower.startswith('grep ') or question_lower.startswith('find '):
            # Extract search term - handle various formats
            search_term = None
            for word in ['search ', 'find ', 'grep ']:
                if question_lower.startswith(word):
                    search_term = question[len(word):].strip().strip('"\'')
                    break
            
            if search_term:
                return self.search_log(search_term)
            else:
                return "Please specify what to search for. Example: 'search error' or 'find CRISPR'"
        
        # 4. Phase-specific queries (exact match for phase names)
        for phase in ['plan', 'learn', 'execute', 'assess', 'share']:
            if question_lower == phase or question_lower == f"{phase} phase" or question_lower == f"show {phase}":
                return self.get_phase_details(phase)
        
        # 5. Progress report (exact match)
        if question_lower in ['progress', 'status']:
            return self.get_progress_report(detailed=False)
        
        # 6. Cost information (exact match)
        if question_lower in ['cost', 'budget', 'costs']:
            return self._handle_cost_query(question)
        
        # 7. Decisions (exact match)
        if question_lower in ['decisions', 'decision']:
            return self.get_decision_summary()
        
        # 8. Help (exact match)
        if question_lower in ['help', 'commands', '?']:
            return """
╔════════════════════════════════════════════════════════════════════════════════╗
║                    PRODUCT MANAGER - AVAILABLE COMMANDS                        ║
╚════════════════════════════════════════════════════════════════════════════════╝

📋 LOG MANAGEMENT:
  • list / show logs / available logs    → List all log files
  • switch <log_name>                    → Change to different log file
  
🔍 ANALYSIS:
  • detailed / analysis / full report    → Get comprehensive log analysis
  • progress / status                    → Get current progress report
  • search <term> / find <term>          → Search for specific term in log
  
📊 SPECIFIC QUERIES:
  • plan/learn/execute/assess/share      → Get details for specific phase
  • cost / budget / spent                → Get cost breakdown
  • errors / warnings / problems         → Show errors in log
  • decisions                            → Show agent decisions
  • artifacts / outputs / files          → Show generated files
  
❓ OTHER:
  • help                                 → Show this help message
  • exit / quit                          → Exit interactive mode

💡 TIP: You can ask questions naturally, like:
  - "What errors occurred?"
  - "Show me the PLAN phase"
  - "Search for CRISPR"
  - "What files were generated?"
"""
        
        # 11. Default fallback - use LLM for natural language query
        if self.llm:
            return self.ask_llm(question)
        else:
            return f"""I'm not sure how to answer "{question}".

Try asking:
  • 'detailed' - for comprehensive analysis
  • 'list' - to see all available logs
  • 'search <term>' - to find specific content
  • 'help' - to see all available commands"""

    def ask_llm(self, question: str) -> str:
        """Use LLM to answer questions about the log file
        
        Args:
            question: Natural language question
            
        Returns:
            LLM-generated answer based on log context
        """
        if not self.llm:
            return "LLM not available. Please use specific commands like 'detailed', 'list', 'search', etc."
        
        # Get log content
        log_to_analyze = self.specific_log_file or self.get_latest_log_file()
        
        if not log_to_analyze or not log_to_analyze.exists():
            return "No log file available for analysis."
        
        # Read log content (limit size for LLM context)
        try:
            with open(log_to_analyze, 'r', encoding='utf-8') as f:
                log_content = f.read()
            
            # Limit log content to avoid context overflow (last 50KB)
            if len(log_content) > 50000:
                log_content = "...[earlier content truncated]...\n\n" + log_content[-50000:]
            
            # Parse structured information
            log_info = self.parse_log_content(log_to_analyze)
            
            # Build context for LLM
            context = f"""You are analyzing a BioPLE agent execution log file.

LOG FILE: {log_to_analyze.name}
FILE SIZE: {log_to_analyze.stat().st_size / 1024:.2f} KB
LAST MODIFIED: {datetime.fromtimestamp(log_to_analyze.stat().st_mtime).strftime('%Y-%m-%d %H:%M:%S')}

STRUCTURED INFORMATION:
- Current Phase: {log_info.get('phase', 'UNKNOWN')}
- Phases Completed: {', '.join(log_info.get('phases_completed', ['None']))}
- Current Task: {log_info.get('current_task', 'Not detected')[:200]}
- Artifacts Generated: {len(log_info.get('artifacts', []))} files

LOG CONTENT (last 50KB):
{log_content}

User Question: {question}

Please provide a clear, concise answer based on the log content above. If the log doesn't contain relevant information, say so. Focus on the most recent activity when relevant."""

            # Query LLM
            response = self.llm.invoke(context)
            
            # Extract text from response
            if hasattr(response, 'content'):
                return response.content
            else:
                return str(response)
                
        except Exception as e:
            return f"Error querying LLM: {e}\n\nTry using specific commands like 'search', 'detailed', etc."

    def chat(self, message: str = None) -> str:
        """Interactive chat mode with the PM agent using LLM
        
        Args:
            message: Optional message to start chat, or None to enter interactive mode
            
        Returns:
            Response from the PM agent
        """
        if not self.llm:
            return "LLM not available. Please initialize with llm_model and llm_source parameters."
        
        if message:
            return self.ask_llm(message)
        
        # Interactive mode
        print("\n" + "="*80)
        print("PRODUCT MANAGER - INTERACTIVE CHAT MODE")
        print("="*80)
        print("Ask me anything about the log file. Type 'exit' or 'quit' to stop.")
        print("-"*80 + "\n")
        
        while True:
            try:
                user_input = input("You: ").strip()
                
                if not user_input:
                    continue
                
                if user_input.lower() in ['exit', 'quit', 'q']:
                    print("Chat session ended.")
                    break
                
                response = self.ask_llm(user_input)
                print(f"\nPM: {response}\n")
                
            except KeyboardInterrupt:
                print("\n\nChat session ended.")
                break
            except Exception as e:
                print(f"\nError: {e}\n")

        return "Chat session completed."

    # =========================================================================
    # MEETING COORDINATION
    # =========================================================================

    def conduct_meeting(
        self,
        phase_logger,
        comm_hub,
        topic: str = "Project Status and Direction",
        meeting_type: str = "standup"
    ) -> Dict[str, Any]:
        """Conduct a multi-agent meeting to discuss project direction

        This method uses the MeetingFacilitator to bring together virtual agents
        (LLM instances with different personas) to discuss project status and direction.

        Args:
            phase_logger: PhaseLogger instance with organized phase logs
            comm_hub: CommunicationHub for inter-agent messaging
            topic: Meeting topic
            meeting_type: Type of meeting - "standup", "planning", or "retrospective"

        Returns:
            Dictionary with meeting results and transcript
        """
        try:
            from bioplease.agent.meeting import MeetingFacilitator
            from bioplease.llm import get_llm

            # Detect meeting stage based on project status
            summary = phase_logger.get_all_phases_summary()
            has_final_paper = summary["phases"].get("SHARE", {}).get("step_count", 0) > 0
            meeting_stage = "final_paper" if has_final_paper else "mid_run"
            
            stage_label = "FINAL PAPER REVIEW" if meeting_stage == "final_paper" else "MID-RUN TECHNICAL REVIEW"
            print(f"\n[PM] Meeting Stage: {stage_label}")

            # Create meeting facilitator
            facilitator = MeetingFacilitator(
                phase_logger=phase_logger,
                comm_hub=comm_hub,
                llm_factory=get_llm,
                llm_model=self.llm_model
            )

            # Conduct appropriate meeting type with stage awareness
            if meeting_type == "standup":
                results = facilitator.quick_standup(
                    save_transcript=True,
                    meeting_stage=meeting_stage
                )
            elif meeting_type == "planning":
                results = facilitator.planning_session(
                    focus_area=topic,
                    save_transcript=True,
                    meeting_stage=meeting_stage
                )
            elif meeting_type == "retrospective":
                results = facilitator.retrospective(
                    save_transcript=True,
                    meeting_stage=meeting_stage
                )
            else:
                # Custom meeting with stage-appropriate prompts
                if meeting_stage == "mid_run":
                    discussion_prompts = [
                        "What specific technical issues or bugs do you identify in the current implementation?",
                        "What exact parameter settings, configurations, or code changes are needed?",
                        "What low-level diagnostic data or error details require attention?"
                    ]
                else:  # final_paper
                    discussion_prompts = [
                        "How does the scientific quality and rigor of this work compare to publication standards?",
                        "What are the key strengths and weaknesses of the methodology and results?",
                        "Is this work ready for publication, or what major revisions are needed?"
                    ]
                
                results = facilitator.conduct_meeting(
                    topic=topic,
                    discussion_prompts=discussion_prompts,
                    rounds=2,
                    save_transcript=True,
                    meeting_stage=meeting_stage
                )

            # Log decision from meeting
            if results.get("action_items"):
                self.log_decision(
                    decision_type=DecisionType.STRATEGY_CHANGE,
                    phase="MEETING",
                    description=f"Multi-agent meeting conducted: {topic}",
                    rationale=f"Meeting generated {len(results['action_items'])} action items",
                    metadata={
                        "meeting_id": results.get("meeting_id"),
                        "action_items": results.get("action_items", [])[:5],  # First 5
                        "participants": results.get("participants", [])
                    }
                )

            return results

        except ImportError as e:
            print(f"Error: Could not import meeting modules: {e}")
            return {"error": str(e)}
        except Exception as e:
            print(f"Error conducting meeting: {e}")
            return {"error": str(e)}

    def schedule_periodic_meetings(
        self,
        phase_logger,
        comm_hub,
        interval_phases: int = 3
    ):
        """Schedule periodic meetings after every N phases

        This can be called periodically by the A1 agent to trigger meetings.

        Args:
            phase_logger: PhaseLogger instance
            comm_hub: CommunicationHub instance
            interval_phases: How many phases between meetings

        Returns:
            Meeting results if conducted, None otherwise
        """
        # Count completed phases
        summary = phase_logger.get_all_phases_summary()
        completed_phases = sum(
            1 for phase_info in summary["phases"].values()
            if phase_info["exists"] and phase_info["step_count"] > 0
        )

        # Check if we should have a meeting
        if completed_phases % interval_phases == 0 and completed_phases > 0:
            print(f"\n[PM] Triggering periodic meeting after {completed_phases} phases")
            return self.conduct_meeting(
                phase_logger=phase_logger,
                comm_hub=comm_hub,
                topic=f"Status Review - After {completed_phases} Phases",
                meeting_type="standup"
            )

        return None

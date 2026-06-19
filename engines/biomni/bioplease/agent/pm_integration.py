"""Integration module for Product Manager with A1 Agent

This module provides helper functions to integrate the Product Manager
with the existing A1 agent architecture without modifying the core a1.py file.
"""

from typing import Optional
from pathlib import Path
from .product_manager import ProductManager


def add_product_manager_to_agent(agent, logs_dir: Optional[str] = None):
    """Add Product Manager capability to an existing A1 agent instance

    Args:
        agent: A1 agent instance
        logs_dir: Optional custom logs directory path

    Returns:
        The agent instance with PM attached
    """
    # Determine logs directory
    if logs_dir is None:
        # Use agent's data path
        logs_dir = Path(agent.path) / "logs"

    # Create Product Manager instance
    pm = ProductManager(
        logs_dir=str(logs_dir),
        data_path=str(agent.path),
        cost_manager=agent.cost_manager if hasattr(agent, 'cost_manager') else None
    )

    # Attach to agent
    agent.product_manager = pm
    # Note: agent.pm property is defined in A1 class as a shorthand

    # Add convenience methods for existing features
    def get_progress_report(detailed=False):
        return agent.product_manager.get_progress_report(detailed=detailed)

    def check_budget(budget_limit=None):
        if budget_limit is None and hasattr(agent, 'cost_budget'):
            budget_limit = agent.cost_budget
        return agent.product_manager.check_budget_status(budget_limit)

    def add_lesson(lesson):
        return agent.product_manager.add_lesson_learned(lesson)

    def pm_update():
        """Get a quick PM update message"""
        snapshot = agent.product_manager.create_progress_snapshot()
        return agent.product_manager.communicate_with_user(snapshot.status_message)

    # Add convenience methods for new features
    def log_decision(decision_type, description, rationale=None, alternatives=None, cost_impact=0.0, **metadata):
        """Log an agent decision"""
        return agent.product_manager.log_decision(
            decision_type=decision_type,
            description=description,
            rationale=rationale,
            alternatives=alternatives,
            cost_impact=cost_impact,
            **metadata
        )

    def get_decision_summary():
        """Get summary of agent decisions"""
        return agent.product_manager.get_decision_summary()

    def query_pm(question):
        """Ask PM agent a question (synchronous)"""
        return agent.product_manager.query(question)

    def ask_pm_async(question):
        """Ask PM agent a question asynchronously"""
        agent.product_manager.ask_async(question)

    def get_pm_response(timeout=0.1):
        """Get async response from PM agent"""
        return agent.product_manager.get_async_response(timeout=timeout)

    def generate_pm_report(output_dir=None, include_graphs=True):
        """Generate comprehensive PM report with graphs"""
        return agent.product_manager.generate_comprehensive_report(
            output_dir=output_dir,
            include_graphs=include_graphs
        )

    def update_state(agent_name, output):
        return agent.product_manager.update_state_from_agent(agent_name, output)

    def transition_state():
        return agent.product_manager.transition_to_new_state()

    def get_state_context():
        return agent.product_manager.get_state_context()

    # Attach methods to agent
    agent.get_progress_report = get_progress_report
    agent.check_budget = check_budget
    agent.add_lesson = add_lesson
    agent.pm_update = pm_update

    # Attach new methods
    agent.log_decision = log_decision
    agent.get_decision_summary = get_decision_summary
    agent.query_pm = query_pm
    agent.ask_pm_async = ask_pm_async
    agent.get_pm_response = get_pm_response
    agent.generate_pm_report = generate_pm_report
    
    def start_phase(phase):
        return agent.product_manager.start_phase_state(phase)

    def end_phase(phase, output):
        return agent.product_manager.end_phase_state(phase, output)

    # Attach state management methods
    agent.update_state = update_state
    agent.transition_state = transition_state
    agent.get_state_context = get_state_context
    agent.start_phase = start_phase
    agent.end_phase = end_phase

    return agent


def create_pm_enhanced_agent(path="./data", llm="gpt-4o-mini", cost_budget=5.0, **kwargs):
    """Create an A1 agent with Product Manager already integrated

    Args:
        path: Path to data directory
        llm: LLM model to use
        cost_budget: Cost budget limit
        **kwargs: Additional arguments for A1 initialization

    Returns:
        A1 agent instance with Product Manager integrated
    """
    from .a1 import A1

    # Create agent
    agent = A1(path=path, llm=llm, cost_budget=cost_budget, **kwargs)

    # Add Product Manager
    agent = add_product_manager_to_agent(agent)

    return agent

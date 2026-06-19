import unittest
from unittest.mock import MagicMock, patch
import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from bioplease.agent.a1 import A1, AgentState
from langchain_core.messages import HumanMessage, AIMessage

class TestA1StateIntegration(unittest.TestCase):
    @patch('bioplease.agent.a1.get_llm')
    @patch('bioplease.agent.pm_integration.add_product_manager_to_agent')
    def test_state_integration(self, mock_add_pm, mock_get_llm):
        # Mock LLM
        mock_llm_instance = MagicMock()
        mock_llm_instance.invoke.return_value = AIMessage(content="Mock response")
        mock_get_llm.return_value = mock_llm_instance

        # Initialize Agent
        agent = A1(path="./test_data", enable_pm=True)
        
        # Mock Product Manager
        agent.product_manager = MagicMock()
        agent.product_manager.get_state_context.return_value = {"previous_summary": "Prev Summary"}
        
        # Mock internal methods to avoid complex setup
        agent._ensure_mem = MagicMock()
        agent._prompt_for = MagicMock(return_value="Prompt")
        agent._llm_for = MagicMock(return_value=mock_llm_instance)
        agent.phase_logger = MagicMock()
        agent._roll_memory = MagicMock()
        agent._prune_history = MagicMock()
        
        # Create a dummy state
        state = {
            "messages": [HumanMessage(content="Start")],
            "artifacts": {"memory": {}},
            "phase": "INIT"
        }
        
        # Configure the agent to build the graph
        agent.configure()
        
        # We can't easily access the inner functions directly, but we can run the graph
        # and check if the PM methods are called.
        
        # Mock the LLM to return a valid response for PLAN
        mock_llm_instance.invoke.return_value = AIMessage(content="Plan output")
        
        # Run the graph for the 'plan' node
        # We can use invoke on the compiled app
        try:
            result = agent.app.invoke(state, config={"recursion_limit": 5})
        except Exception as e:
            # It might fail on subsequent steps if we don't mock everything, 
            # but we just want to see if 'plan' ran and called PM.
            print(f"Graph execution stopped: {e}")
            pass
            
        # Check if get_state_context was called (in PLAN)
        agent.product_manager.get_state_context.assert_called()
        
        # Check if update_state_from_agent was called (in PLAN)
        # Note: The graph execution might have proceeded to 'learn' etc.
        # So we check if it was called at least once with "PLAN"
        agent.product_manager.update_state_from_agent.assert_any_call("PLAN", "Plan output")

if __name__ == '__main__':
    unittest.main()

import json
import uuid
import os
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import List, Dict, Any, Optional

@dataclass
class Resource:
    title: str
    link: str
    type: str
    relevance: str
    status: str

@dataclass
class AGEScores:
    achievement: float
    growth: float
    effort: float

@dataclass
class State:
    state_id: str
    phase: str
    iteration: int
    project_id: str
    timestamp_start: str
    timestamp_end: Optional[str] = None
    
    # Original task for context
    original_task: Optional[str] = None
    
    # Complete phase outputs (not summaries)
    plan_output: Optional[str] = None
    learn_output: Optional[str] = None
    execute_output: Optional[str] = None
    mini_share_output: Optional[str] = None
    assess_output: Optional[str] = None
    share_output: Optional[str] = None
    
    # Long-term memory integrated into state
    long_term_memory: Optional[str] = None
    
    # Links and artifacts
    plan_document_link: Optional[str] = None
    learning_resources: List[Resource] = field(default_factory=list)
    execution_logs_links: List[str] = field(default_factory=list)
    execution_metrics: Dict[str, Any] = field(default_factory=dict)
    share_documents: List[str] = field(default_factory=list)
    share_notion_links: List[str] = field(default_factory=list)
    share_commit_hashes: List[str] = field(default_factory=list)
    
    # Scores and metrics
    assessment_scores: Dict[str, float] = field(default_factory=dict)
    assessment_recommendations: Optional[str] = None
    age_scores: Dict[str, AGEScores] = field(default_factory=dict)
    expense_summary: Dict[str, Any] = field(default_factory=dict)
    cost_budget: Optional[float] = None  # Cost budget for the task
    
    # Incidents
    rescue_incidents: List[Dict[str, Any]] = field(default_factory=list)
    
    # History of previous phase outputs (list of dicts with phase, output, timestamp)
    previous_states: List[Dict[str, Any]] = field(default_factory=list)

    def to_json(self) -> str:
        return json.dumps(asdict(self), default=str, indent=2)

    @classmethod
    def from_json(cls, json_str: str) -> 'State':
        data = json.loads(json_str)
        
        # Reconstruct nested objects
        if 'learning_resources' in data:
            data['learning_resources'] = [Resource(**r) for r in data['learning_resources']]
            
        if 'age_scores' in data and data['age_scores']:
            # Handle dictionary of AGEScores
            scores_dict = {}
            for agent_name, score_data in data['age_scores'].items():
                scores_dict[agent_name] = AGEScores(**score_data)
            data['age_scores'] = scores_dict
            
        return cls(**data)

    def save_to_file(self, filepath: str):
        try:
            with open(filepath, 'w') as f:
                f.write(self.to_json())
        except (IOError, OSError) as e:
            print(f"[WARNING] Failed to write state file {filepath}: {e}")
            raise

    @classmethod
    def load_from_file(cls, filepath: str) -> 'State':
        with open(filepath, 'r') as f:
            return cls.from_json(f.read())

class StateManager:
    def __init__(self, db_path: str = "state_db.json", storage_dir: str = "state_storage"):
        self.db_path = db_path
        self.storage_dir = storage_dir
        self.states: Dict[str, State] = {}
        self.load_db()

    def load_db(self):
        if os.path.exists(self.db_path):
            try:
                with open(self.db_path, 'r') as f:
                    data = json.load(f)
                    for state_id, state_data in data.items():
                        # We need to handle the reconstruction carefully if we just load the dict
                        # But for simplicity, let's assume we can re-instantiate or just keep as dict if needed
                        # Better to use the from_json logic if possible, but here we have a dict of dicts
                        # Let's just store the raw data or reconstruct if we want to be strict
                        # For now, let's just keep it simple and rely on explicit save/load of individual states
                        pass 
            except Exception as e:
                print(f"Error loading state DB: {e}")

    def set_storage_dir(self, new_dir: str):
        self.storage_dir = new_dir
        os.makedirs(self.storage_dir, exist_ok=True)

    def save_state(self, state: State):
        self.states[state.state_id] = state
        
        # Save to states/<PHASE>/<ID>.json with error handling
        try:
            states_dir = os.path.join(self.storage_dir, "states")
            phase_dir = os.path.join(states_dir, state.phase)
            os.makedirs(phase_dir, exist_ok=True)
            
            filename = os.path.join(phase_dir, f"{state.state_id}.json")
            state.save_to_file(filename)
        except Exception as e:
            print(f"[WARNING] Failed to save state to file: {e}")
            # Continue execution even if file save fails

    def get_state(self, state_id: str, phase: str) -> Optional[State]:
        filename = os.path.join(self.storage_dir, "states", phase, f"{state_id}.json")
        if os.path.exists(filename):
            return State.load_from_file(filename)
        return None

    def create_new_state(self, project_id: str, phase: str, iteration: int) -> State:
        state_id = str(iteration)
        timestamp_start = datetime.now().isoformat()
        new_state = State(
            state_id=state_id, 
            phase=phase,
            iteration=iteration,
            project_id=project_id, 
            timestamp_start=timestamp_start
        )
        return new_state

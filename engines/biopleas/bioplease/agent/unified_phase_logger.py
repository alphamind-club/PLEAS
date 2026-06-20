"""
Unified Phase Log System for BioPLEASE

This module provides a single, cumulative log file that records all phases, steps, and context in a single .txt file, updated after every phase.
"""
import os
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, Optional

class UnifiedPhaseLogger:
    def __init__(self, log_file: str):
        self.log_file = Path(log_file)
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        # Optionally clear the log at the start of a run
        self.log_file.write_text(f"# Unified Phase Log\n# Created: {datetime.now().isoformat()}\n\n")

    def log_phase(self, phase: str, step_type: str, content: Any, metadata: Optional[Dict[str, Any]] = None):
        timestamp = datetime.now().isoformat()
        meta_str = f" | Metadata: {metadata}" if metadata else ""
        entry = (
            f"\n{'='*80}\n"
            f"[{timestamp}] PHASE: {phase} | STEP: {step_type}{meta_str}\n"
            f"{'-'*80}\n"
            f"{content}\n"
        )
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(entry)

    def log_context(self, context: str):
        timestamp = datetime.now().isoformat()
        entry = (
            f"\n{'='*80}\n"
            f"[{timestamp}] CONTEXT UPDATE\n"
            f"{'-'*80}\n"
            f"{context}\n"
        )
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(entry)

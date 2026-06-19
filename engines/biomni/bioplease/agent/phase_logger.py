"""Phase-based Logging System for BioPLE

This module provides structured, phase-organized logging where each phase's
outputs are separated into individual folders, and each step (prompt/response)
is saved as a separate file for easy analysis and debugging.

Directory structure:
    logs/
        PLAN/
            01_prompt.txt
            02_response.txt
            03_prompt.txt (if multiple iterations)
            04_response.txt
        LEARN/
            01_prompt.txt
            02_response.txt
        EXECUTE/
            01_prompt.txt
            02_response.txt
            ...
        ASSESS/
            ...
        SHARE/
            ...
        OVERVIEW_ASSESS/
            ...
"""

import os
import json
import time
import sys
from pathlib import Path
from typing import Any, Dict, Optional, List
from datetime import datetime
from dataclasses import dataclass, asdict


@dataclass
class LogEntry:
    """Represents a single log entry"""
    timestamp: str
    phase: str
    step_number: int
    step_type: str  # "prompt" or "response"
    content: str
    metadata: Dict[str, Any]


class PhaseLogger:
    """Manages phase-organized logging for BioPLE agents

    Each phase gets its own directory, and each step (prompt/response pair)
    is saved as individual files for easy tracking and analysis.

    Attributes:
        base_logs_dir: Base directory for all logs (e.g., data/bioplease_data/runs/RUN_123/logs)
        enabled: Whether logging is enabled
        phase_counters: Track step numbers for each phase
        current_phase: Currently active phase
    """

    def __init__(self, base_logs_dir: str, enabled: bool = True):
        """Initialize the phase logger

        Args:
            base_logs_dir: Base directory where phase logs will be stored
            enabled: Whether to enable logging (default True)
        """
        self.base_logs_dir = Path(base_logs_dir)
        self.enabled = enabled
        self.phase_counters: Dict[str, int] = {}
        self.current_phase: Optional[str] = None
        self.log_entries: List[LogEntry] = []

        # stdout/stderr redirection tracking
        self.result_log_file = None
        self.original_stdout = None
        self.original_stderr = None
        self.tee_stdout = None
        self.tee_stderr = None

        # Valid phases
        self.valid_phases = [
            "PLAN", "LEARN", "EXECUTE", "MINI_SHARE", "ASSESS",
            "SHARE", "OVERVIEW_ASSESS", "MEETING"
        ]

        if self.enabled:
            self._setup_directories()

    def _setup_directories(self):
        """Create the base logs directory and phase subdirectories"""
        self.base_logs_dir.mkdir(parents=True, exist_ok=True)

        # Create phase directories
        for phase in self.valid_phases:
            phase_dir = self.base_logs_dir / phase
            phase_dir.mkdir(exist_ok=True)

            # Initialize counter for each phase
            if phase not in self.phase_counters:
                self.phase_counters[phase] = 0

    def _msgs_to_text(self, content) -> str:
        """Convert message content to text string

        Handles various message formats (string, list, objects with .content)
        """
        if isinstance(content, str):
            return content
        if isinstance(content, (list, tuple)):
            parts = []
            for m in content:
                # Try to take message.content; fall back to str(m)
                parts.append(getattr(m, "content", str(m)))
            return "\n".join(parts)
        return str(content)

    def set_phase(self, phase: str):
        """Set the current active phase

        Args:
            phase: Name of the phase (PLAN, LEARN, EXECUTE, ASSESS, SHARE, etc.)
        """
        if phase not in self.valid_phases:
            print(f"Warning: Unknown phase '{phase}'. Valid phases: {self.valid_phases}")

        self.current_phase = phase

        # Ensure phase directory exists and counter is initialized
        if self.enabled:
            phase_dir = self.base_logs_dir / phase
            phase_dir.mkdir(exist_ok=True)

            if phase not in self.phase_counters:
                self.phase_counters[phase] = 0

    def log_step(
        self,
        step_type: str,
        content: Any,
        phase: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Optional[Path]:
        """Log a single step (prompt or response) for a phase

        Args:
            step_type: Type of step - "prompt" or "response"
            content: Content to log (can be string, messages, etc.)
            phase: Phase name (uses current_phase if not specified)
            metadata: Additional metadata to include

        Returns:
            Path to the created log file, or None if logging disabled
        """
        if not self.enabled:
            return None

        # Determine phase
        phase = phase or self.current_phase
        if not phase:
            print("Warning: No phase set. Call set_phase() first.")
            return None

        # Increment counter only on prompts (so prompt and response share the same number)
        if step_type == "prompt":
            self.phase_counters[phase] += 1
        
        step_number = self.phase_counters[phase]

        # Convert content to text
        text_content = self._msgs_to_text(content).strip()

        # Create log entry
        timestamp = datetime.now().isoformat()
        entry = LogEntry(
            timestamp=timestamp,
            phase=phase,
            step_number=step_number,
            step_type=step_type,
            content=text_content,
            metadata=metadata or {}
        )
        self.log_entries.append(entry)

        # Determine filename
        phase_dir = self.base_logs_dir / phase
        filename = f"{step_number:02d}_{step_type}.txt"
        filepath = phase_dir / filename

        # Write to file with error handling
        try:
            phase_dir.mkdir(parents=True, exist_ok=True)
            with open(filepath, "w", encoding="utf-8") as f:
                # Write header with metadata
                f.write(f"# {step_type.upper()} - Step {step_number}\n")
                f.write(f"# Timestamp: {timestamp}\n")
                f.write(f"# Phase: {phase}\n")

                if metadata:
                    f.write(f"# Metadata: {json.dumps(metadata)}\n")

                f.write("\n" + "="*80 + "\n\n")
                f.write(text_content)
                f.write("\n")
        except (IOError, OSError) as e:
            print(f"[WARNING] Failed to write phase log to {filepath}: {e}")
            return None  # Return None to indicate failure but continue execution

        return filepath

    def log_prompt(
        self,
        content: Any,
        phase: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Optional[Path]:
        """Log a prompt

        Args:
            content: Prompt content
            phase: Phase name (optional, uses current phase)
            metadata: Additional metadata

        Returns:
            Path to created log file
        """
        return self.log_step("prompt", content, phase, metadata)

    def log_response(
        self,
        content: Any,
        phase: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Optional[Path]:
        """Log a response

        Args:
            content: Response content
            phase: Phase name (optional, uses current phase)
            metadata: Additional metadata

        Returns:
            Path to created log file
        """
        return self.log_step("response", content, phase, metadata)

    def log_observation(
        self,
        content: Any,
        phase: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Optional[Path]:
        """Log an observation (e.g., code execution output, tool results)

        Args:
            content: Observation content
            phase: Phase name (optional, uses current phase)
            metadata: Additional metadata

        Returns:
            Path to created log file
        """
        if not self.enabled:
            return None

        # Determine phase
        phase = phase or self.current_phase
        if not phase:
            print("Warning: No phase set. Call set_phase() first.")
            return None

        # Use current step number (don't increment for observations)
        step_number = self.phase_counters.get(phase, 0)
        if step_number == 0:
            # If no steps yet, start at 1
            step_number = 1
            self.phase_counters[phase] = 1

        # Convert content to text
        text_content = self._msgs_to_text(content).strip()

        # Create log entry
        timestamp = datetime.now().isoformat()
        entry = LogEntry(
            timestamp=timestamp,
            phase=phase,
            step_number=step_number,
            step_type="observation",
            content=text_content,
            metadata=metadata or {}
        )
        self.log_entries.append(entry)

        # Determine filename - observations use their own file
        phase_dir = self.base_logs_dir / phase
        filename = f"{step_number:02d}_observation.txt"
        filepath = phase_dir / filename

        # Write to file
        phase_dir.mkdir(parents=True, exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            # Write header with metadata
            f.write(f"# OBSERVATION - Step {step_number}\n")
            f.write(f"# Timestamp: {timestamp}\n")
            f.write(f"# Phase: {phase}\n")

            if metadata:
                f.write(f"# Metadata: {json.dumps(metadata)}\n")

            f.write("\n" + "="*80 + "\n\n")
            f.write(text_content)
            f.write("\n")

        return filepath

    def get_phase_summary(self, phase: str) -> Dict[str, Any]:
        """Get summary of logs for a specific phase

        Args:
            phase: Phase name

        Returns:
            Dictionary with phase summary information
        """
        phase_dir = self.base_logs_dir / phase

        if not phase_dir.exists():
            return {
                "phase": phase,
                "exists": False,
                "step_count": 0,
                "files": []
            }

        files = sorted(phase_dir.glob("*.txt"))

        return {
            "phase": phase,
            "exists": True,
            "step_count": len(files),
            "files": [str(f.name) for f in files],
            "directory": str(phase_dir)
        }

    def get_all_phases_summary(self) -> Dict[str, Any]:
        """Get summary of all phase logs

        Returns:
            Dictionary with summary of all phases
        """
        summaries = {}
        for phase in self.valid_phases:
            summaries[phase] = self.get_phase_summary(phase)

        return {
            "base_dir": str(self.base_logs_dir),
            "total_entries": len(self.log_entries),
            "phases": summaries
        }

    def export_phase_log(self, phase: str, output_file: Optional[str] = None) -> str:
        """Export all logs for a phase into a single consolidated file

        Args:
            phase: Phase name
            output_file: Output file path (optional, auto-generated if not provided)

        Returns:
            Path to the exported file
        """
        phase_dir = self.base_logs_dir / phase

        if not phase_dir.exists():
            raise ValueError(f"Phase directory does not exist: {phase_dir}")

        # Auto-generate output filename if not provided
        if not output_file:
            output_file = str(self.base_logs_dir / f"{phase}_consolidated.txt")

        files = sorted(phase_dir.glob("*.txt"))

        with open(output_file, "w", encoding="utf-8") as out:
            out.write(f"# Consolidated Log for Phase: {phase}\n")
            out.write(f"# Generated: {datetime.now().isoformat()}\n")
            out.write(f"# Total Steps: {len(files)}\n")
            out.write("\n" + "="*80 + "\n\n")

            for file in files:
                out.write(f"\n{'#'*80}\n")
                out.write(f"# FILE: {file.name}\n")
                out.write(f"{'#'*80}\n\n")

                with open(file, "r", encoding="utf-8") as f:
                    out.write(f.read())

                out.write("\n\n")

        return output_file

    def get_phase_entries(self, phase: str) -> List[LogEntry]:
        """Get all log entries for a specific phase

        Args:
            phase: Phase name

        Returns:
            List of LogEntry objects for the phase
        """
        return [entry for entry in self.log_entries if entry.phase == phase]

    def save_manifest(self, output_file: Optional[str] = None) -> str:
        """Save a manifest file with all log entries metadata

        Args:
            output_file: Output file path (optional)

        Returns:
            Path to the manifest file
        """
        if not output_file:
            output_file = str(self.base_logs_dir / "manifest.json")

        manifest = {
            "generated_at": datetime.now().isoformat(),
            "base_directory": str(self.base_logs_dir),
            "total_entries": len(self.log_entries),
            "phases": self.get_all_phases_summary(),
            "entries": [asdict(entry) for entry in self.log_entries]
        }

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)

        return output_file

    class TeeWriter:
        """Writer that outputs to both a file and the original stream"""
        def __init__(self, file, stream):
            self.file = file
            self.stream = stream
        
        def write(self, data):
            self.file.write(data)
            self.file.flush()  # Ensure immediate write
            self.stream.write(data)
            self.stream.flush()
        
        def flush(self):
            self.file.flush()
            self.stream.flush()

    def log_result_start(self, phase: Optional[str] = None):
        """Start redirecting stdout/stderr to result log file
        
        Creates logs/<PHASE>/<step_id>_result.txt and redirects all output there.
        
        Args:
            phase: Phase name (uses current_phase if not specified)
        """
        if not self.enabled:
            return
        
        # Determine phase
        phase = phase or self.current_phase
        if not phase:
            print("Warning: No phase set. Call set_phase() first.")
            return
        
        # Get current step number
        step_number = self.phase_counters.get(phase, 0)
        
        # Create result log file path
        phase_dir = self.base_logs_dir / phase
        phase_dir.mkdir(parents=True, exist_ok=True)
        
        result_filename = f"{step_number:02d}_result.txt"
        result_filepath = phase_dir / result_filename
        
        # Save original streams
        self.original_stdout = sys.stdout
        self.original_stderr = sys.stderr
        
        # Open result log file
        self.result_log_file = open(result_filepath, 'w', encoding='utf-8', buffering=1)
        
        # Write header
        timestamp = datetime.now().isoformat()
        self.result_log_file.write(f"# RESULT LOG - Step {step_number}\n")
        self.result_log_file.write(f"# Timestamp: {timestamp}\n")
        self.result_log_file.write(f"# Phase: {phase}\n")
        self.result_log_file.write("\n" + "="*80 + "\n\n")
        self.result_log_file.flush()
        
        # Create tee writers (output to both file and console)
        self.tee_stdout = self.TeeWriter(self.result_log_file, self.original_stdout)
        self.tee_stderr = self.TeeWriter(self.result_log_file, self.original_stderr)
        
        # Redirect stdout and stderr
        sys.stdout = self.tee_stdout
        sys.stderr = self.tee_stderr
        
        print(f"[PhaseLogger] Started capturing output to {result_filepath}")

    def log_result_end(self):
        """Stop redirecting stdout/stderr and close result log file"""
        if not self.enabled:
            return
        
        if self.result_log_file is None:
            return  # Nothing to close
        
        # Print footer before closing
        print("\n" + "="*80)
        print(f"# End of result log - {datetime.now().isoformat()}")
        
        # Restore original streams
        if self.original_stdout:
            sys.stdout = self.original_stdout
        if self.original_stderr:
            sys.stderr = self.original_stderr
        
        # Close the result log file
        if self.result_log_file:
            self.result_log_file.close()
            self.result_log_file = None
        
        # Reset tee writers
        self.tee_stdout = None
        self.tee_stderr = None
        self.original_stdout = None
        self.original_stderr = None
        
        print("[PhaseLogger] Stopped capturing output")

"""Meeting Feature for Product Manager

This module implements a multi-agent meeting system where different agents
(represented as LLM instances with specialized prompts) discuss project direction,
challenges, and next steps.

The meeting system uses:
- PhaseLogger for accessing organized phase logs
- CommunicationHub for inter-agent messaging
- LLM instances with different system prompts for each agent role
"""

import json
import time
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
from dataclasses import dataclass

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

from bioplease.agent.phase_logger import PhaseLogger
from bioplease.agent.communication_hub import (
    CommunicationHub,
    MessageType,
    MessagePriority
)


@dataclass
class AgentPersona:
    """Represents an agent persona for meetings"""
    agent_id: str
    name: str
    role: str
    system_prompt: str
    expertise: List[str]
    personality_traits: str = ""


class MeetingFacilitator:
    """Facilitates multi-agent meetings for project planning and coordination

    The facilitator creates virtual agents (LLM instances with specialized prompts)
    that discuss project direction based on the structured phase logs.

    Attributes:
        phase_logger: PhaseLogger for accessing organized logs
        comm_hub: CommunicationHub for agent communication
        llm_factory: Function to create LLM instances
        personas: Available agent personas
    """

    def __init__(
        self,
        phase_logger: PhaseLogger,
        comm_hub: CommunicationHub,
        llm_factory,
        llm_model: str = "gpt-4o"
    ):
        """Initialize the meeting facilitator

        Args:
            phase_logger: PhaseLogger instance with phase logs
            comm_hub: CommunicationHub for messaging
            llm_factory: Function to create LLM (e.g., get_llm from bioplease.llm)
            llm_model: Model to use for agent LLMs
        """
        self.phase_logger = phase_logger
        self.comm_hub = comm_hub
        self.llm_factory = llm_factory
        self.llm_model = llm_model

        # Define agent personas
        self.personas = self._create_personas()

        # Register personas with communication hub
        for persona in self.personas.values():
            self.comm_hub.register_agent(
                agent_id=persona.agent_id,
                agent_type=persona.role,
                description=f"{persona.name} - {persona.role}",
                capabilities=persona.expertise
            )

    def _create_personas(self) -> Dict[str, AgentPersona]:
        """Create agent personas for meetings

        Returns:
            Dictionary mapping agent_id to AgentPersona
        """
        personas = {
            "product_manager": AgentPersona(
                agent_id="product_manager",
                name="Product Manager",
                role="Project Coordinator",
                system_prompt="""You are a HIGHLY CRITICAL Product Manager for a scientific research project using the BioPLE framework.
Your role is to IDENTIFY PROBLEMS AND RISKS:
- Expose coordination failures and communication gaps
- Flag wasted resources and inefficiencies
- Identify blockers, delays, and bottlenecks
- Point out when project goals are not being met
- Challenge vague or unrealistic plans
- Call out when decisions lack data or reasoning

DO NOT PRAISE ACCOMPLISHMENTS. DO NOT BE DIPLOMATIC.
Be direct about failures, missed deadlines, scope creep, and poor resource allocation.
Focus exclusively on what's wrong and what needs to be fixed. No sugar-coating.""",
                expertise=["project_management", "resource_planning", "risk_assessment", "coordination"],
                personality_traits="Skeptical, demanding, brutally honest"
            ),

            "scientific_planner": AgentPersona(
                agent_id="scientific_planner",
                name="Scientific Planner",
                role="Research Strategy",
                system_prompt="""You are a HIGHLY CRITICAL Scientific Planner, responsible for finding flaws in research strategy.
Your role is to IDENTIFY SCIENTIFIC WEAKNESSES:
- Find gaps in experimental design and methodology
- Challenge weak hypotheses and research questions
- Expose lack of scientific rigor and invalid assumptions
- Identify missing controls and confounding variables
- Point out irreproducibility and poor statistical power
- Flag methodological shortcuts and sloppy reasoning

DO NOT PRAISE GOOD DESIGNS. DO NOT BE ENCOURAGING.
Be ruthless about scientific flaws, weak reasoning, and methodological problems.
Focus exclusively on what could fail, what's missing, and what's scientifically unsound.""",
                expertise=["research_design", "methodology", "hypothesis_generation", "scientific_planning"],
                personality_traits="Skeptical, uncompromising, ruthlessly rigorous"
            ),

            "technical_executor": AgentPersona(
                agent_id="technical_executor",
                name="Technical Executor",
                role="Implementation & Engineering",
                system_prompt="""You are a HIGHLY CRITICAL Technical Executor, responsible for identifying implementation failures.
Your role is to EXPOSE TECHNICAL PROBLEMS:
- Identify buggy, inefficient, or fragile code
- Point out technical debt and poor architecture
- Flag performance bottlenecks and resource waste
- Expose lack of error handling and edge cases
- Identify untested code and reproducibility issues
- Challenge overly complex or unmaintainable solutions

DO NOT PRAISE WORKING CODE. DO NOT BE SUPPORTIVE.
Be harsh about code quality, performance issues, and technical shortcuts.
Focus exclusively on what's broken, what will break, and what's poorly implemented.""",
                expertise=["coding", "debugging", "optimization", "implementation", "tooling"],
                personality_traits="Perfectionistic, impatient, unforgiving of bugs"
            ),

            "quality_assessor": AgentPersona(
                agent_id="quality_assessor",
                name="Quality Assessor",
                role="Quality & Validation",
                system_prompt="""You are a MERCILESS Quality Assessor, responsible for finding every flaw and weakness.
Your role is to IDENTIFY QUALITY FAILURES:
- Find errors, inconsistencies, and invalid conclusions
- Expose weak validation and insufficient testing
- Identify reproducibility failures and transparency gaps
- Challenge unsupported claims and overgeneralizations
- Point out cherry-picked results and confirmation bias
- Flag low-quality outputs and sloppy documentation

DO NOT ACKNOWLEDGE GOOD WORK. DO NOT BE CONSTRUCTIVE.
Be ruthless about quality failures, errors, and insufficient validation.
Focus exclusively on what's wrong, what's missing, and what doesn't meet standards.""",
                expertise=["validation", "quality_assurance", "peer_review", "error_detection"],
                personality_traits="Hyper-critical, skeptical, uncompromising"
            ),

            "knowledge_integrator": AgentPersona(
                agent_id="knowledge_integrator",
                name="Knowledge Integrator",
                role="Learning & Resources",
                system_prompt="""You are a HIGHLY CRITICAL Knowledge Integrator, responsible for exposing knowledge gaps and poor scholarship.
Your role is to IDENTIFY INTELLECTUAL FAILURES:
- Point out ignored literature and missed prior work
- Expose reinventing the wheel and wasted effort
- Identify inappropriate or outdated tools and methods
- Flag lack of integration with existing knowledge
- Challenge assumptions not grounded in evidence
- Point out when the team is ignorant of relevant work

DO NOT PRAISE GOOD RESEARCH PRACTICES. DO NOT BE HELPFUL.
Be harsh about knowledge gaps, missed literature, and poor tool choices.
Focus exclusively on what's been overlooked, what's wrong, and what shows lack of scholarship.""",
                expertise=["literature_review", "resource_discovery", "knowledge_synthesis", "tool_selection"],
                personality_traits="Pedantic, demanding, intolerant of ignorance"
            ),

            "communication_specialist": AgentPersona(
                agent_id="communication_specialist",
                name="Communication Specialist",
                role="Documentation & Sharing",
                system_prompt="""You are a HIGHLY CRITICAL Communication Specialist, responsible for exposing documentation failures.
Your role is to IDENTIFY COMMUNICATION PROBLEMS:
- Point out unclear, missing, or misleading documentation
- Expose poor organization and confusing presentation
- Identify irreproducible methods and inadequate detail
- Flag vague language and unsupported claims
- Challenge presentations that won't convince the audience
- Point out when work can't be understood or replicated

DO NOT PRAISE CLEAR WRITING. DO NOT BE ENCOURAGING.
Be harsh about documentation quality, clarity failures, and poor communication.
Focus exclusively on what's unclear, what's missing, and what will fail to communicate.""",
                expertise=["technical_writing", "documentation", "visualization", "presentation"],
                personality_traits="Demanding, perfectionist, intolerant of ambiguity"
            )
        }

        return personas

    def conduct_meeting(
        self,
        topic: str,
        discussion_prompts: List[str],
        participants: Optional[List[str]] = None,
        rounds: int = 2,
        save_transcript: bool = True,
        meeting_stage: str = "mid_run"  # NEW: "mid_run" or "final_paper"
    ) -> Dict[str, Any]:
        """Conduct a multi-agent meeting

        Args:
            topic: Meeting topic/purpose
            discussion_prompts: List of prompts to guide discussion
            participants: List of agent IDs to participate (None = all)
            rounds: Number of discussion rounds
            save_transcript: Whether to save meeting transcript
            meeting_stage: "mid_run" for granular technical details, "final_paper" for high-level review

        Returns:
            Dictionary with meeting results
        """
        # Determine participants
        if participants is None:
            participants = list(self.personas.keys())
        else:
            # Validate participants
            invalid = [p for p in participants if p not in self.personas]
            if invalid:
                raise ValueError(f"Invalid participants: {invalid}")

        # Prepare context from phase logs
        context = self._prepare_meeting_context()
        
        # Detect meeting stage automatically if not explicitly set
        if meeting_stage == "mid_run":
            # Check if final paper exists
            summary = self.phase_logger.get_all_phases_summary()
            has_final_paper = summary["phases"].get("SHARE", {}).get("step_count", 0) > 0
            if has_final_paper:
                meeting_stage = "final_paper"

        # Create meeting
        meeting_id = f"meeting_{int(time.time())}"
        meeting = self.comm_hub.create_meeting(
            meeting_id=meeting_id,
            organizer="product_manager",
            topic=topic,
            participants=participants,
            description=f"Multi-agent discussion: {topic}",
            metadata={
                "discussion_prompts": discussion_prompts,
                "rounds": rounds,
                "context": context
            }
        )

        print(f"\n{'='*80}")
        print(f"Starting Meeting: {topic}")
        print(f"Participants: {', '.join([self.personas[p].name for p in participants])}")
        print(f"Rounds: {rounds}")
        print(f"{'='*80}\n")

        # Initialize LLMs for each participant
        agent_llms = {}
        for participant_id in participants:
            persona = self.personas[participant_id]
            try:
                agent_llms[participant_id] = self.llm_factory(
                    model=self.llm_model,
                    temperature=0.7
                )
            except Exception as e:
                print(f"Warning: Could not initialize LLM for {persona.name}: {e}")
                return {"error": f"Failed to initialize LLM: {e}"}

        # Discussion history for each agent
        agent_histories: Dict[str, List] = {p: [] for p in participants}

        # Conduct discussion rounds
        all_responses = []
        transcript_path = None
        meeting_summary = None
        interrupted = False

        try:
            for round_num in range(rounds):
                print(f"\n--- Round {round_num + 1}/{rounds} ---\n")

                round_responses = []

                for prompt in discussion_prompts:
                    prompt_responses = {}

                    print(f"\nDiscussion Point: {prompt}\n")

                    # Each agent responds
                    for participant_id in participants:
                        persona = self.personas[participant_id]
                        llm = agent_llms[participant_id]

                        # Build messages for this agent with stage-specific guidance
                        if meeting_stage == "mid_run":
                            detail_guidance = """
FOCUS ON GRANULAR TECHNICAL DETAILS:
This is a MID-RUN checkpoint. Dive deep into LOW-LEVEL implementation specifics:
- Exact parameter values, hyperparameters, and configuration settings
- Specific error messages, stack traces, and debugging observations
- Precise numerical results, intermediate calculations, and diagnostic metrics
- Detailed code logic, algorithm steps, and data transformations
- Software versions, dependencies, environment specifics
- Performance bottlenecks, memory usage, computational resources
- Edge cases, boundary conditions, and failure modes

Be EXTREMELY DETAILED and TECHNICAL. Provide actionable insights at the implementation level.
Include specific values, file paths, function names, and exact observations.
Help identify what needs to be fixed NOW in the current execution cycle."""
                        else:  # final_paper
                            detail_guidance = """
FOCUS ON HIGH-LEVEL SCIENTIFIC QUALITY:
This is a FINAL PAPER review. Evaluate scientific soundness and presentation:
- Overall research design, methodology rigor, and reproducibility
- Scientific contribution, novelty, and significance
- Clarity of communication, paper structure, and argumentation
- Completeness of results, discussion, and limitations
- Statistical rigor, experimental validation, and interpretation
- Alignment with research questions and hypotheses
- Publication readiness and peer review standards

Be STRATEGIC and COMPREHENSIVE. Focus on scientific merit and presentation quality.
Evaluate whether this work is ready for publication or needs major revision.
Help identify gaps in scientific reasoning, writing, or validation."""

                        messages = [
                            SystemMessage(content=persona.system_prompt),
                            HumanMessage(content=f"""You are participating in a project meeting.

Topic: {topic}

Context from recent project phases:
{context}

Discussion point: {prompt}

{detail_guidance}

Please provide your perspective as the {persona.role}. Consider:
- What problems, flaws, or weaknesses do you identify?
- What risks or concerns do you see?
- What gaps or missing elements are there?
- What could go wrong or fail?
- What specific improvements are needed?

Be CRITICAL and DIRECT. Focus on identifying issues, not praising successes.
Be concise (2-3 paragraphs) and actionable.""")
                        ]

                        # Add previous responses from this round for context
                        if prompt_responses:
                            prev_summary = "\n\n".join([
                                f"{self.personas[pid].name}: {resp}"
                                for pid, resp in prompt_responses.items()
                            ])
                            messages.append(HumanMessage(content=f"Other participants have said:\n{prev_summary}"))

                        # Get response
                        try:
                            response = llm.invoke(messages)
                            response_text = response.content.strip()

                            # Store in history
                            agent_histories[participant_id].append({
                                "round": round_num + 1,
                                "prompt": prompt,
                                "response": response_text
                            })

                            prompt_responses[participant_id] = response_text

                            # Send message in meeting
                            meeting.send_message(
                                sender=participant_id,
                                content=f"Re: {prompt}\n\n{response_text}",
                                metadata={
                                    "round": round_num + 1,
                                    "prompt": prompt
                                }
                            )

                            # Print response
                            print(f"{persona.name}:")
                            print(f"{response_text}\n")

                        except Exception as e:
                            print(f"Error getting response from {persona.name}: {e}")
                            prompt_responses[participant_id] = f"[Error: {e}]"

                    round_responses.append({
                        "prompt": prompt,
                        "responses": prompt_responses
                    })

                all_responses.append({
                    "round": round_num + 1,
                    "discussions": round_responses
                })

            # End meeting and generate summary
            meeting_summary = meeting.end_meeting()

        except KeyboardInterrupt:
            print("\n\n⚠️  Meeting interrupted by user (Ctrl+C)")
            interrupted = True
            meeting_summary = "Meeting interrupted before completion"
        except Exception as e:
            print(f"\n\n⚠️  Meeting interrupted by error: {e}")
            interrupted = True
            meeting_summary = f"Meeting interrupted by error: {e}"
        finally:
            # Save transcript even if interrupted
            if save_transcript and self.phase_logger.enabled:
                transcript_path = str(
                    self.phase_logger.base_logs_dir / "MEETING" / f"transcript_{meeting_id}{'_PARTIAL' if interrupted else ''}.txt"
                )
                meeting.save_transcript(transcript_path)
                status = "PARTIAL" if interrupted else "COMPLETE"
                print(f"\nMeeting transcript ({status}) saved to: {transcript_path}")

        # Generate actionable summary
        action_items = self._generate_action_items(all_responses, participants)

        results = {
            "meeting_id": meeting_id,
            "topic": topic,
            "participants": participants,
            "rounds": rounds,
            "responses": all_responses,
            "action_items": action_items,
            "summary": meeting_summary if meeting_summary else "Meeting completed",
            "transcript_path": transcript_path,
            "interrupted": interrupted
        }

        print(f"\n{'='*80}")
        if interrupted:
            print("Meeting Interrupted (Partial Results Saved)")
        else:
            print("Meeting Complete")
        print(f"{'='*80}\n")

        return results

    def _prepare_meeting_context(self) -> str:
        """Prepare context from phase logs for the meeting

        Returns:
            Formatted context string
        """
        context_parts = []

        # Get summaries from each phase
        for phase in ["PLAN", "LEARN", "EXECUTE", "ASSESS", "SHARE"]:
            summary = self.phase_logger.get_phase_summary(phase)

            if summary["exists"] and summary["step_count"] > 0:
                # Read the latest response from this phase
                phase_dir = Path(summary["directory"])
                response_files = sorted(phase_dir.glob("*_response.txt"))

                if response_files:
                    latest_response = response_files[-1]

                    try:
                        with open(latest_response, "r", encoding="utf-8") as f:
                            content = f.read()

                        # Extract just the content (skip header)
                        lines = content.split("\n")
                        content_start = 0
                        for i, line in enumerate(lines):
                            if line.startswith("="*40):
                                content_start = i + 1
                                break

                        phase_content = "\n".join(lines[content_start:]).strip()

                        # Truncate if too long
                        if len(phase_content) > 1000:
                            phase_content = phase_content[:1000] + "\n... [truncated]"

                        context_parts.append(f"**{phase} Phase Summary:**\n{phase_content}")

                    except Exception as e:
                        context_parts.append(f"**{phase} Phase:** [Error reading: {e}]")

        if not context_parts:
            return "No phase logs available yet. This is an initial planning meeting."

        return "\n\n".join(context_parts)

    def _generate_action_items(
        self,
        all_responses: List[Dict[str, Any]],
        participants: List[str]
    ) -> List[str]:
        """Generate action items from meeting discussion

        Args:
            all_responses: All discussion responses
            participants: Participant IDs

        Returns:
            List of action items
        """
        action_items = []

        # Analyze responses for action-oriented language
        action_keywords = [
            "should", "need to", "must", "recommend", "suggest",
            "propose", "implement", "fix", "improve", "add",
            "create", "update", "revise", "review"
        ]

        for round_data in all_responses:
            for discussion in round_data["discussions"]:
                for participant_id, response in discussion["responses"].items():
                    # Look for sentences with action keywords
                    sentences = response.split(".")

                    for sentence in sentences:
                        sentence = sentence.strip()
                        if any(keyword in sentence.lower() for keyword in action_keywords):
                            if len(sentence) > 20 and len(sentence) < 200:
                                persona = self.personas[participant_id]
                                action_items.append(f"[{persona.name}] {sentence}")

        # Deduplicate and limit
        action_items = list(set(action_items))[:15]

        return action_items

    def quick_standup(self, save_transcript: bool = True, meeting_stage: str = "mid_run") -> Dict[str, Any]:
        """Conduct a quick standup meeting to assess current status

        Args:
            save_transcript: Whether to save transcript
            meeting_stage: "mid_run" or "final_paper"

        Returns:
            Meeting results
        """
        return self.conduct_meeting(
            topic="Project Status Standup",
            discussion_prompts=[
                "What is the current status from your perspective?",
                "What blockers or challenges do you see?",
                "What should we focus on next?"
            ],
            rounds=1,
            save_transcript=save_transcript,
            meeting_stage=meeting_stage
        )

    def planning_session(
        self,
        focus_area: str,
        save_transcript: bool = True,
        meeting_stage: str = "mid_run"
    ) -> Dict[str, Any]:
        """Conduct a planning session for a specific area

        Args:
            focus_area: Area to focus planning on
            save_transcript: Whether to save transcript
            meeting_stage: "mid_run" or "final_paper"

        Returns:
            Meeting results
        """
        return self.conduct_meeting(
            topic=f"Planning Session: {focus_area}",
            discussion_prompts=[
                f"What are the key challenges for {focus_area}?",
                f"What approach would you recommend for {focus_area}?",
                "What resources or tools do we need?",
                "What are the success criteria?"
            ],
            rounds=2,
            save_transcript=save_transcript,
            meeting_stage=meeting_stage
        )

    def retrospective(self, save_transcript: bool = True, meeting_stage: str = "mid_run") -> Dict[str, Any]:
        """Conduct a retrospective on recent work

        Args:
            save_transcript: Whether to save transcript
            meeting_stage: "mid_run" or "final_paper"

        Returns:
            Meeting results
        """
        return self.conduct_meeting(
            topic="Project Retrospective",
            discussion_prompts=[
                "What went well in the recent phases?",
                "What could be improved?",
                "What did we learn?",
                "What should we do differently going forward?"
            ],
            rounds=1,
            save_transcript=save_transcript,
            meeting_stage=meeting_stage
        )

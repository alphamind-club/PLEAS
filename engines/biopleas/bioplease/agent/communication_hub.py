"""Inter-Agent Communication Hub for BioPLE

This module provides a communication infrastructure for agents to exchange
messages, coordinate activities, and participate in multi-agent discussions.

Key features:
- Message bus for asynchronous agent communication
- Topic-based subscriptions
- Meeting coordination for multi-agent discussions
- Message history and replay capabilities
"""

import json
import threading
import queue
from pathlib import Path
from typing import Any, Dict, List, Optional, Callable, Set
from datetime import datetime
from dataclasses import dataclass, asdict, field
from enum import Enum


class MessagePriority(Enum):
    """Priority levels for messages"""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    URGENT = 4


class MessageType(Enum):
    """Types of inter-agent messages"""
    DIRECT = "direct"  # Direct message to specific agent
    BROADCAST = "broadcast"  # Broadcast to all agents
    TOPIC = "topic"  # Message on a specific topic
    MEETING_INVITE = "meeting_invite"  # Invitation to meeting
    MEETING_MESSAGE = "meeting_message"  # Message during meeting
    STATUS_UPDATE = "status_update"  # Agent status update
    REQUEST = "request"  # Request for action/information
    RESPONSE = "response"  # Response to a request


@dataclass
class AgentMessage:
    """Represents a message between agents"""
    message_id: str
    timestamp: str
    sender: str
    message_type: MessageType
    content: str
    recipients: List[str] = field(default_factory=list)
    topic: Optional[str] = None
    priority: MessagePriority = MessagePriority.NORMAL
    metadata: Dict[str, Any] = field(default_factory=dict)
    in_reply_to: Optional[str] = None  # ID of message this is replying to
    phase_context: Optional[str] = None  # Which phase this message relates to


@dataclass
class AgentProfile:
    """Profile information for an agent"""
    agent_id: str
    agent_type: str  # e.g., "ProductManager", "Planner", "Executor"
    description: str
    capabilities: List[str]
    subscribed_topics: Set[str] = field(default_factory=set)
    is_active: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)


class CommunicationHub:
    """Central communication hub for inter-agent messaging

    Provides a message bus for agents to communicate asynchronously,
    subscribe to topics, and participate in coordinated activities.

    Attributes:
        agents: Registry of connected agents
        messages: History of all messages
        topics: Map of topics to subscribed agents
        meetings: Active and past meetings
    """

    def __init__(self, logs_dir: Optional[str] = None):
        """Initialize the communication hub

        Args:
            logs_dir: Directory to save communication logs (optional)
        """
        self.agents: Dict[str, AgentProfile] = {}
        self.messages: List[AgentMessage] = []
        self.topics: Dict[str, Set[str]] = {}  # topic -> set of agent_ids
        self.meetings: Dict[str, 'Meeting'] = {}

        # Message queues for each agent
        self.agent_queues: Dict[str, queue.Queue] = {}

        # Callbacks for message handlers
        self.message_handlers: Dict[str, Callable] = {}

        # Thread safety
        self.lock = threading.Lock()

        # Logging
        self.logs_dir = Path(logs_dir) if logs_dir else None
        if self.logs_dir:
            self.logs_dir.mkdir(parents=True, exist_ok=True)

        # Message ID counter
        self._message_counter = 0

    def register_agent(
        self,
        agent_id: str,
        agent_type: str,
        description: str,
        capabilities: List[str],
        metadata: Optional[Dict[str, Any]] = None
    ) -> AgentProfile:
        """Register an agent with the communication hub

        Args:
            agent_id: Unique identifier for the agent
            agent_type: Type/role of the agent
            description: Human-readable description
            capabilities: List of agent capabilities
            metadata: Additional metadata

        Returns:
            AgentProfile for the registered agent
        """
        with self.lock:
            profile = AgentProfile(
                agent_id=agent_id,
                agent_type=agent_type,
                description=description,
                capabilities=capabilities,
                metadata=metadata or {}
            )

            self.agents[agent_id] = profile
            self.agent_queues[agent_id] = queue.Queue()

            return profile

    def unregister_agent(self, agent_id: str):
        """Unregister an agent from the hub

        Args:
            agent_id: ID of agent to unregister
        """
        with self.lock:
            if agent_id in self.agents:
                self.agents[agent_id].is_active = False

                # Remove from all topics
                for topic_agents in self.topics.values():
                    topic_agents.discard(agent_id)

    def subscribe_to_topic(self, agent_id: str, topic: str):
        """Subscribe an agent to a topic

        Args:
            agent_id: ID of the agent
            topic: Topic to subscribe to
        """
        with self.lock:
            if topic not in self.topics:
                self.topics[topic] = set()

            self.topics[topic].add(agent_id)

            if agent_id in self.agents:
                self.agents[agent_id].subscribed_topics.add(topic)

    def unsubscribe_from_topic(self, agent_id: str, topic: str):
        """Unsubscribe an agent from a topic

        Args:
            agent_id: ID of the agent
            topic: Topic to unsubscribe from
        """
        with self.lock:
            if topic in self.topics:
                self.topics[topic].discard(agent_id)

            if agent_id in self.agents:
                self.agents[agent_id].subscribed_topics.discard(topic)

    def send_message(
        self,
        sender: str,
        content: str,
        message_type: MessageType = MessageType.BROADCAST,
        recipients: Optional[List[str]] = None,
        topic: Optional[str] = None,
        priority: MessagePriority = MessagePriority.NORMAL,
        metadata: Optional[Dict[str, Any]] = None,
        in_reply_to: Optional[str] = None,
        phase_context: Optional[str] = None
    ) -> AgentMessage:
        """Send a message through the hub

        Args:
            sender: ID of sending agent
            content: Message content
            message_type: Type of message
            recipients: List of recipient agent IDs (for direct messages)
            topic: Topic for topic-based messages
            priority: Message priority
            metadata: Additional metadata
            in_reply_to: ID of message this is replying to
            phase_context: Phase context for the message

        Returns:
            The created AgentMessage
        """
        with self.lock:
            # Generate message ID
            self._message_counter += 1
            message_id = f"msg_{self._message_counter:06d}"

            # Determine recipients
            if message_type == MessageType.BROADCAST:
                recipients = list(self.agents.keys())
            elif message_type == MessageType.TOPIC and topic:
                recipients = list(self.topics.get(topic, set()))
            elif recipients is None:
                recipients = []

            # Create message
            message = AgentMessage(
                message_id=message_id,
                timestamp=datetime.now().isoformat(),
                sender=sender,
                message_type=message_type,
                content=content,
                recipients=recipients,
                topic=topic,
                priority=priority,
                metadata=metadata or {},
                in_reply_to=in_reply_to,
                phase_context=phase_context
            )

            # Store message
            self.messages.append(message)

            # Deliver to recipient queues
            for recipient in recipients:
                if recipient in self.agent_queues and recipient != sender:
                    self.agent_queues[recipient].put(message)

            # Log message if logging enabled
            if self.logs_dir:
                self._log_message(message)

            return message

    def get_messages(
        self,
        agent_id: str,
        block: bool = False,
        timeout: Optional[float] = None
    ) -> Optional[AgentMessage]:
        """Get next message for an agent

        Args:
            agent_id: ID of the agent
            block: Whether to block waiting for message
            timeout: Timeout in seconds (if blocking)

        Returns:
            Next AgentMessage or None
        """
        if agent_id not in self.agent_queues:
            return None

        try:
            return self.agent_queues[agent_id].get(block=block, timeout=timeout)
        except queue.Empty:
            return None

    def get_message_history(
        self,
        agent_id: Optional[str] = None,
        topic: Optional[str] = None,
        message_type: Optional[MessageType] = None,
        phase_context: Optional[str] = None,
        limit: Optional[int] = None
    ) -> List[AgentMessage]:
        """Get message history with optional filtering

        Args:
            agent_id: Filter by sender or recipient
            topic: Filter by topic
            message_type: Filter by message type
            phase_context: Filter by phase
            limit: Maximum number of messages to return

        Returns:
            List of matching messages
        """
        filtered = self.messages

        if agent_id:
            filtered = [
                m for m in filtered
                if m.sender == agent_id or agent_id in m.recipients
            ]

        if topic:
            filtered = [m for m in filtered if m.topic == topic]

        if message_type:
            filtered = [m for m in filtered if m.message_type == message_type]

        if phase_context:
            filtered = [m for m in filtered if m.phase_context == phase_context]

        if limit:
            filtered = filtered[-limit:]

        return filtered

    def _log_message(self, message: AgentMessage):
        """Log a message to file

        Args:
            message: Message to log
        """
        if not self.logs_dir:
            return

        # Create daily log file
        date_str = datetime.now().strftime("%Y%m%d")
        log_file = self.logs_dir / f"communication_{date_str}.jsonl"

        with open(log_file, "a", encoding="utf-8") as f:
            # Convert message to dict, handling Enum types
            msg_dict = asdict(message)
            msg_dict["message_type"] = message.message_type.value
            msg_dict["priority"] = message.priority.value

            f.write(json.dumps(msg_dict, ensure_ascii=False) + "\n")

    def create_meeting(
        self,
        meeting_id: str,
        organizer: str,
        topic: str,
        participants: List[str],
        description: str = "",
        metadata: Optional[Dict[str, Any]] = None
    ) -> 'Meeting':
        """Create a new meeting

        Args:
            meeting_id: Unique meeting identifier
            organizer: Agent ID of meeting organizer
            topic: Meeting topic
            participants: List of participant agent IDs
            description: Meeting description
            metadata: Additional metadata

        Returns:
            Created Meeting object
        """
        meeting = Meeting(
            meeting_id=meeting_id,
            organizer=organizer,
            topic=topic,
            participants=participants,
            description=description,
            hub=self,
            metadata=metadata or {}
        )

        self.meetings[meeting_id] = meeting

        # Send invitations
        for participant in participants:
            self.send_message(
                sender=organizer,
                content=f"Meeting invitation: {topic}\n\n{description}",
                message_type=MessageType.MEETING_INVITE,
                recipients=[participant],
                metadata={
                    "meeting_id": meeting_id,
                    "topic": topic
                }
            )

        return meeting

    def get_active_agents(self) -> List[AgentProfile]:
        """Get list of active agents

        Returns:
            List of active AgentProfile objects
        """
        return [agent for agent in self.agents.values() if agent.is_active]

    def export_communication_log(self, output_file: str):
        """Export all communications to a file

        Args:
            output_file: Path to output file
        """
        export_data = {
            "exported_at": datetime.now().isoformat(),
            "total_messages": len(self.messages),
            "agents": {
                agent_id: {
                    "agent_type": agent.agent_type,
                    "description": agent.description,
                    "capabilities": agent.capabilities,
                    "is_active": agent.is_active
                }
                for agent_id, agent in self.agents.items()
            },
            "messages": [
                {
                    **asdict(msg),
                    "message_type": msg.message_type.value,
                    "priority": msg.priority.value
                }
                for msg in self.messages
            ]
        }

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)


@dataclass
class Meeting:
    """Represents a multi-agent meeting

    Meetings provide a structured way for multiple agents to have
    coordinated discussions on specific topics.
    """
    meeting_id: str
    organizer: str
    topic: str
    participants: List[str]
    description: str
    hub: CommunicationHub
    metadata: Dict[str, Any] = field(default_factory=dict)
    start_time: str = field(default_factory=lambda: datetime.now().isoformat())
    end_time: Optional[str] = None
    is_active: bool = True
    transcript: List[AgentMessage] = field(default_factory=list)

    def send_message(
        self,
        sender: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> AgentMessage:
        """Send a message in the meeting

        Args:
            sender: ID of sending agent
            content: Message content
            metadata: Additional metadata

        Returns:
            The sent AgentMessage
        """
        if not self.is_active:
            raise ValueError("Meeting is not active")

        if sender not in self.participants and sender != self.organizer:
            raise ValueError(f"Agent {sender} is not a participant")

        # Send message through hub
        message = self.hub.send_message(
            sender=sender,
            content=content,
            message_type=MessageType.MEETING_MESSAGE,
            recipients=self.participants,
            topic=self.topic,
            metadata={
                **(metadata or {}),
                "meeting_id": self.meeting_id
            }
        )

        # Add to transcript
        self.transcript.append(message)

        return message

    def end_meeting(self) -> Dict[str, Any]:
        """End the meeting and generate summary

        Returns:
            Meeting summary dictionary
        """
        self.is_active = False
        self.end_time = datetime.now().isoformat()

        summary = {
            "meeting_id": self.meeting_id,
            "topic": self.topic,
            "organizer": self.organizer,
            "participants": self.participants,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "message_count": len(self.transcript),
            "duration_seconds": self._calculate_duration()
        }

        return summary

    def _calculate_duration(self) -> float:
        """Calculate meeting duration in seconds"""
        if not self.end_time:
            return 0.0

        start = datetime.fromisoformat(self.start_time)
        end = datetime.fromisoformat(self.end_time)
        return (end - start).total_seconds()

    def get_transcript(self) -> str:
        """Get formatted meeting transcript

        Returns:
            Formatted transcript string
        """
        lines = [
            f"Meeting Transcript: {self.topic}",
            f"Meeting ID: {self.meeting_id}",
            f"Organizer: {self.organizer}",
            f"Participants: {', '.join(self.participants)}",
            f"Start: {self.start_time}",
            ""
        ]

        if self.end_time:
            lines.append(f"End: {self.end_time}")
            lines.append("")

        lines.append("="*80)
        lines.append("")

        for msg in self.transcript:
            lines.append(f"[{msg.timestamp}] {msg.sender}:")
            lines.append(msg.content)
            lines.append("")

        return "\n".join(lines)

    def save_transcript(self, output_file: str):
        """Save meeting transcript to file

        Args:
            output_file: Path to output file
        """
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(self.get_transcript())

"""
Pro Se Divorce Document Automation System - Core Architecture

This module defines the complete architecture for automating pro se divorce
document generation, including agent interfaces, workflow orchestration,
message passing, and state management.

Architecture Pattern: Multi-Agent Pipeline with Shared State
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import (
    Any,
    Callable,
    Dict,
    Generic,
    List,
    Optional,
    Protocol,
    TypeVar,
    Union,
)
from contextlib import asynccontextmanager
import traceback

# Configure module logger
logger = logging.getLogger(__name__)


# =============================================================================
# Enums and Constants
# =============================================================================

class AgentType(Enum):
    """Enumeration of all agent types in the system."""
    DATA_INGESTION = auto()
    DOCUMENT_BUILDER = auto()
    FORMATTING_EDITOR = auto()
    LEGAL_PROOFER = auto()
    VISUAL_FORMATTER = auto()
    LEARNING_COORDINATOR = auto()
    HUMAN_INTERFACE = auto()


class MessagePriority(Enum):
    """Priority levels for inter-agent messages."""
    LOW = 1
    NORMAL = 5
    HIGH = 8
    CRITICAL = 10


class WorkflowStage(Enum):
    """Stages in the document automation workflow."""
    INTAKE = "intake"
    DATA_INGESTION = "data_ingestion"
    DOCUMENT_BUILDING = "document_building"
    FORMAT_CHECK = "format_check"
    LEGAL_PROOFING = "legal_proofing"
    VISUAL_FORMATTING = "visual_formatting"
    DRAFT_OUTPUT = "draft_output"
    HUMAN_REVIEW = "human_review"
    LEARNING_INCORPORATION = "learning_incorporation"
    FINALIZATION = "finalization"
    COMPLETED = "completed"
    FAILED = "failed"


class DocumentStatus(Enum):
    """Status of a document in the pipeline."""
    PENDING = "pending"
    PROCESSING = "processing"
    AWAITING_REVIEW = "awaiting_review"
    APPROVED = "approved"
    REJECTED = "rejected"
    FINALIZED = "finalized"
    ERROR = "error"


# =============================================================================
# Data Classes and Models
# =============================================================================

@dataclass
class AgentMessage:
    """Message passed between agents in the system."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    sender: AgentType = AgentType.DATA_INGESTION
    recipient: AgentType = AgentType.DOCUMENT_BUILDER
    payload: Dict[str, Any] = field(default_factory=dict)
    priority: MessagePriority = MessagePriority.NORMAL
    timestamp: datetime = field(default_factory=datetime.utcnow)
    correlation_id: Optional[str] = None
    reply_to: Optional[str] = None
    ttl_seconds: int = 3600
    metadata: Dict[str, Any] = field(default_factory=dict)

    def is_expired(self) -> bool:
        """Check if message has exceeded its TTL."""
        elapsed = (datetime.utcnow() - self.timestamp).total_seconds()
        return elapsed > self.ttl_seconds


@dataclass
class DocumentContext:
    """Context for a document being processed through the pipeline."""
    document_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    case_number: Optional[str] = None
    jurisdiction: str = ""
    document_type: str = ""
    petitioner_data: Dict[str, Any] = field(default_factory=dict)
    respondent_data: Dict[str, Any] = field(default_factory=dict)
    case_data: Dict[str, Any] = field(default_factory=dict)
    generated_content: Dict[str, Any] = field(default_factory=dict)
    validation_results: List[Dict[str, Any]] = field(default_factory=list)
    stage: WorkflowStage = WorkflowStage.INTAKE
    status: DocumentStatus = DocumentStatus.PENDING
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    error_log: List[str] = field(default_factory=list)
    retry_count: int = 0
    max_retries: int = 3
    metadata: Dict[str, Any] = field(default_factory=dict)

    def update_stage(self, new_stage: WorkflowStage) -> None:
        """Update the workflow stage and timestamp."""
        self.stage = new_stage
        self.updated_at = datetime.utcnow()

    def log_error(self, error: str) -> None:
        """Log an error to the document context."""
        self.error_log.append(f"[{datetime.utcnow().isoformat()}] {error}")
        self.updated_at = datetime.utcnow()


@dataclass
class BatchContext:
    """Context for batch processing multiple documents."""
    batch_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    documents: List[DocumentContext] = field(default_factory=list)
    total_count: int = 0
    processed_count: int = 0
    success_count: int = 0
    failure_count: int = 0
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    is_cancelled: bool = False

    @property
    def progress_percentage(self) -> float:
        """Calculate batch progress as percentage."""
        if self.total_count == 0:
            return 0.0
        return (self.processed_count / self.total_count) * 100


@dataclass
class AgentResult:
    """Result returned by an agent after processing."""
    success: bool
    data: Dict[str, Any] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    processing_time_ms: float = 0.0
    agent_type: Optional[AgentType] = None
    context_updates: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LearningFeedback:
    """Feedback from human review for learning system."""
    document_id: str
    reviewer_id: str
    feedback_type: str  # 'correction', 'approval', 'rejection'
    original_content: Dict[str, Any] = field(default_factory=dict)
    corrected_content: Dict[str, Any] = field(default_factory=dict)
    comments: str = ""
    timestamp: datetime = field(default_factory=datetime.utcnow)
    severity: str = "low"  # 'low', 'medium', 'high', 'critical'


# =============================================================================
# Protocols and Abstract Base Classes
# =============================================================================

T = TypeVar('T')


class MessageHandler(Protocol):
    """Protocol for message handling capability."""

    async def handle_message(self, message: AgentMessage) -> AgentResult:
        """Handle an incoming message."""
        ...


class StateObserver(Protocol):
    """Protocol for observing state changes."""

    def on_state_change(
        self,
        document_id: str,
        old_stage: WorkflowStage,
        new_stage: WorkflowStage
    ) -> None:
        """Called when document stage changes."""
        ...


class BaseAgent(ABC):
    """
    Abstract base class for all agents in the document automation system.

    All agents must implement the process method and can optionally
    override initialization and cleanup hooks.
    """

    def __init__(
        self,
        agent_id: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        self.agent_id = agent_id or str(uuid.uuid4())
        self.agent_type: AgentType = self._get_agent_type()
        self.config = config or {}
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self._is_initialized = False
        self._message_queue: asyncio.Queue[AgentMessage] = asyncio.Queue()
        self._observers: List[StateObserver] = []

    @abstractmethod
    def _get_agent_type(self) -> AgentType:
        """Return the type of this agent."""
        pass

    @abstractmethod
    async def process(self, context: DocumentContext) -> AgentResult:
        """
        Process a document context and return the result.

        Args:
            context: The document context to process

        Returns:
            AgentResult containing success status and any data/errors
        """
        pass

    async def initialize(self) -> None:
        """Initialize agent resources. Override in subclasses."""
        self.logger.info(f"Initializing agent {self.agent_id}")
        self._is_initialized = True

    async def cleanup(self) -> None:
        """Cleanup agent resources. Override in subclasses."""
        self.logger.info(f"Cleaning up agent {self.agent_id}")
        self._is_initialized = False

    async def handle_message(self, message: AgentMessage) -> AgentResult:
        """Handle an incoming message from another agent."""
        if message.is_expired():
            self.logger.warning(f"Received expired message: {message.id}")
            return AgentResult(
                success=False,
                errors=["Message expired before processing"],
                agent_type=self.agent_type
            )

        self.logger.debug(f"Processing message {message.id} from {message.sender}")
        # Default implementation - subclasses can override
        return AgentResult(success=True, agent_type=self.agent_type)

    def add_observer(self, observer: StateObserver) -> None:
        """Add a state observer."""
        self._observers.append(observer)

    def notify_observers(
        self,
        document_id: str,
        old_stage: WorkflowStage,
        new_stage: WorkflowStage
    ) -> None:
        """Notify all observers of a state change."""
        for observer in self._observers:
            try:
                observer.on_state_change(document_id, old_stage, new_stage)
            except Exception as e:
                self.logger.error(f"Observer notification failed: {e}")

    @asynccontextmanager
    async def processing_context(self, context: DocumentContext):
        """Context manager for processing with proper error handling."""
        start_time = datetime.utcnow()
        try:
            yield
        except Exception as e:
            context.log_error(f"{self.agent_type.name}: {str(e)}")
            self.logger.exception(f"Error in {self.agent_type.name}")
            raise
        finally:
            elapsed = (datetime.utcnow() - start_time).total_seconds() * 1000
            self.logger.debug(f"Processing took {elapsed:.2f}ms")


# =============================================================================
# Agent Interfaces
# =============================================================================

class DataIngestionAgent(BaseAgent):
    """
    Agent responsible for ingesting and validating input data.

    Handles:
    - Form data extraction
    - Document parsing (PDFs, images, etc.)
    - Data validation and normalization
    - Missing data detection
    """

    def _get_agent_type(self) -> AgentType:
        return AgentType.DATA_INGESTION

    @abstractmethod
    async def ingest_form_data(
        self,
        form_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Ingest and validate form data."""
        pass

    @abstractmethod
    async def parse_document(
        self,
        document_bytes: bytes,
        document_type: str
    ) -> Dict[str, Any]:
        """Parse a document and extract structured data."""
        pass

    @abstractmethod
    async def validate_completeness(
        self,
        context: DocumentContext
    ) -> List[str]:
        """
        Validate that all required data is present.

        Returns list of missing required fields.
        """
        pass

    async def process(self, context: DocumentContext) -> AgentResult:
        """Process ingestion for a document context."""
        async with self.processing_context(context):
            missing_fields = await self.validate_completeness(context)

            if missing_fields:
                return AgentResult(
                    success=False,
                    errors=[f"Missing required fields: {', '.join(missing_fields)}"],
                    agent_type=self.agent_type
                )

            return AgentResult(
                success=True,
                data={"ingestion_complete": True},
                agent_type=self.agent_type
            )


class DocumentBuilderAgent(BaseAgent):
    """
    Agent responsible for building and filling document templates.

    Handles:
    - Template selection based on jurisdiction and case type
    - Field mapping and population
    - Dynamic content generation
    - Multi-document package assembly
    """

    def _get_agent_type(self) -> AgentType:
        return AgentType.DOCUMENT_BUILDER

    @abstractmethod
    async def select_template(
        self,
        jurisdiction: str,
        document_type: str
    ) -> str:
        """Select appropriate template for jurisdiction and type."""
        pass

    @abstractmethod
    async def populate_template(
        self,
        template_id: str,
        context: DocumentContext
    ) -> Dict[str, Any]:
        """Populate template with context data."""
        pass

    @abstractmethod
    async def generate_dynamic_content(
        self,
        context: DocumentContext,
        content_type: str
    ) -> str:
        """Generate dynamic content (e.g., declarations, statements)."""
        pass

    @abstractmethod
    async def assemble_package(
        self,
        context: DocumentContext
    ) -> List[Dict[str, Any]]:
        """Assemble complete document package."""
        pass

    async def process(self, context: DocumentContext) -> AgentResult:
        """Build documents from context."""
        async with self.processing_context(context):
            template_id = await self.select_template(
                context.jurisdiction,
                context.document_type
            )

            populated = await self.populate_template(template_id, context)
            package = await self.assemble_package(context)

            return AgentResult(
                success=True,
                data={
                    "template_id": template_id,
                    "populated_content": populated,
                    "package": package
                },
                agent_type=self.agent_type
            )


class FormattingEditorAgent(BaseAgent):
    """
    Agent responsible for format validation and correction.

    Handles:
    - Court formatting requirements validation
    - Margin, font, spacing checks
    - Page numbering and headers
    - Cross-reference validation
    """

    def _get_agent_type(self) -> AgentType:
        return AgentType.FORMATTING_EDITOR

    @abstractmethod
    async def validate_formatting(
        self,
        document: Dict[str, Any],
        jurisdiction_rules: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Validate document formatting against rules.

        Returns list of formatting violations.
        """
        pass

    @abstractmethod
    async def apply_corrections(
        self,
        document: Dict[str, Any],
        violations: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Apply formatting corrections to document."""
        pass

    @abstractmethod
    async def validate_cross_references(
        self,
        document: Dict[str, Any]
    ) -> List[str]:
        """Validate all cross-references are valid."""
        pass

    async def process(self, context: DocumentContext) -> AgentResult:
        """Process formatting validation and correction."""
        async with self.processing_context(context):
            jurisdiction_rules = context.metadata.get("formatting_rules", {})
            document = context.generated_content

            violations = await self.validate_formatting(document, jurisdiction_rules)

            if violations:
                corrected = await self.apply_corrections(document, violations)
                context.generated_content = corrected

            ref_errors = await self.validate_cross_references(
                context.generated_content
            )

            return AgentResult(
                success=len(ref_errors) == 0,
                data={
                    "violations_found": len(violations),
                    "violations_corrected": len(violations),
                    "reference_errors": ref_errors
                },
                errors=ref_errors,
                agent_type=self.agent_type
            )


class LegalProoferAgent(BaseAgent):
    """
    Agent responsible for legal accuracy verification.

    Handles:
    - Legal terminology validation
    - Statutory compliance checking
    - Jurisdiction-specific requirements
    - Legal citation verification
    """

    def _get_agent_type(self) -> AgentType:
        return AgentType.LEGAL_PROOFER

    @abstractmethod
    async def validate_legal_terms(
        self,
        content: str,
        jurisdiction: str
    ) -> List[Dict[str, Any]]:
        """Validate legal terminology usage."""
        pass

    @abstractmethod
    async def check_statutory_compliance(
        self,
        context: DocumentContext
    ) -> List[Dict[str, Any]]:
        """Check compliance with relevant statutes."""
        pass

    @abstractmethod
    async def verify_citations(
        self,
        content: str
    ) -> List[Dict[str, Any]]:
        """Verify all legal citations are valid and current."""
        pass

    @abstractmethod
    async def check_jurisdiction_requirements(
        self,
        context: DocumentContext
    ) -> List[str]:
        """Check jurisdiction-specific filing requirements."""
        pass

    async def process(self, context: DocumentContext) -> AgentResult:
        """Process legal proofing."""
        async with self.processing_context(context):
            content = str(context.generated_content)

            term_issues = await self.validate_legal_terms(
                content,
                context.jurisdiction
            )
            compliance_issues = await self.check_statutory_compliance(context)
            citation_issues = await self.verify_citations(content)
            jurisdiction_issues = await self.check_jurisdiction_requirements(context)

            all_issues = term_issues + compliance_issues + citation_issues
            all_warnings = jurisdiction_issues

            context.validation_results.append({
                "agent": "legal_proofer",
                "timestamp": datetime.utcnow().isoformat(),
                "issues": all_issues,
                "warnings": all_warnings
            })

            return AgentResult(
                success=len(all_issues) == 0,
                data={
                    "term_issues": term_issues,
                    "compliance_issues": compliance_issues,
                    "citation_issues": citation_issues
                },
                warnings=all_warnings,
                agent_type=self.agent_type
            )


class VisualFormatterAgent(BaseAgent):
    """
    Agent responsible for visual presentation and PDF generation.

    Handles:
    - Final visual layout
    - PDF generation
    - Signature placeholders
    - Filing-ready output
    """

    def _get_agent_type(self) -> AgentType:
        return AgentType.VISUAL_FORMATTER

    @abstractmethod
    async def apply_visual_layout(
        self,
        document: Dict[str, Any],
        layout_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Apply visual layout to document."""
        pass

    @abstractmethod
    async def generate_pdf(
        self,
        document: Dict[str, Any]
    ) -> bytes:
        """Generate PDF from document."""
        pass

    @abstractmethod
    async def add_signature_placeholders(
        self,
        document: Dict[str, Any],
        signature_fields: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Add signature placeholders to document."""
        pass

    @abstractmethod
    async def prepare_for_filing(
        self,
        document: Dict[str, Any],
        filing_requirements: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Prepare document for court filing."""
        pass

    async def process(self, context: DocumentContext) -> AgentResult:
        """Process visual formatting."""
        async with self.processing_context(context):
            layout_config = context.metadata.get("layout_config", {})
            filing_requirements = context.metadata.get("filing_requirements", {})
            signature_fields = context.metadata.get("signature_fields", [])

            document = context.generated_content
            document = await self.apply_visual_layout(document, layout_config)
            document = await self.add_signature_placeholders(document, signature_fields)
            document = await self.prepare_for_filing(document, filing_requirements)

            pdf_bytes = await self.generate_pdf(document)

            return AgentResult(
                success=True,
                data={
                    "formatted_document": document,
                    "pdf_size_bytes": len(pdf_bytes),
                    "filing_ready": True
                },
                context_updates={"pdf_output": pdf_bytes},
                agent_type=self.agent_type
            )


class LearningCoordinatorAgent(BaseAgent):
    """
    Agent responsible for incorporating feedback and improving the system.

    Handles:
    - Feedback collection and analysis
    - Pattern recognition from corrections
    - Model fine-tuning coordination
    - Performance metrics tracking
    """

    def _get_agent_type(self) -> AgentType:
        return AgentType.LEARNING_COORDINATOR

    @abstractmethod
    async def record_feedback(
        self,
        feedback: LearningFeedback
    ) -> None:
        """Record feedback for learning."""
        pass

    @abstractmethod
    async def analyze_patterns(
        self,
        feedback_batch: List[LearningFeedback]
    ) -> Dict[str, Any]:
        """Analyze patterns in feedback."""
        pass

    @abstractmethod
    async def update_models(
        self,
        patterns: Dict[str, Any]
    ) -> bool:
        """Update models based on learned patterns."""
        pass

    @abstractmethod
    async def get_performance_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics."""
        pass

    async def process(self, context: DocumentContext) -> AgentResult:
        """Process learning coordination."""
        async with self.processing_context(context):
            # Extract feedback from context if available
            feedback_data = context.metadata.get("human_feedback", [])

            for fb in feedback_data:
                await self.record_feedback(LearningFeedback(**fb))

            metrics = await self.get_performance_metrics()

            return AgentResult(
                success=True,
                data={
                    "feedback_processed": len(feedback_data),
                    "metrics": metrics
                },
                agent_type=self.agent_type
            )


class HumanInterfaceAgent(BaseAgent):
    """
    Agent responsible for human interaction and review coordination.

    Handles:
    - Presenting documents for review
    - Collecting human corrections
    - Managing review workflows
    - Approval/rejection handling
    """

    def _get_agent_type(self) -> AgentType:
        return AgentType.HUMAN_INTERFACE

    @abstractmethod
    async def present_for_review(
        self,
        context: DocumentContext
    ) -> str:
        """Present document for human review. Returns review session ID."""
        pass

    @abstractmethod
    async def collect_feedback(
        self,
        review_session_id: str
    ) -> Optional[LearningFeedback]:
        """Collect feedback from review session."""
        pass

    @abstractmethod
    async def handle_approval(
        self,
        context: DocumentContext
    ) -> bool:
        """Handle document approval."""
        pass

    @abstractmethod
    async def handle_rejection(
        self,
        context: DocumentContext,
        rejection_reason: str
    ) -> Dict[str, Any]:
        """Handle document rejection with reason."""
        pass

    async def process(self, context: DocumentContext) -> AgentResult:
        """Process human interface interaction."""
        async with self.processing_context(context):
            review_session_id = await self.present_for_review(context)

            return AgentResult(
                success=True,
                data={
                    "review_session_id": review_session_id,
                    "status": "awaiting_review"
                },
                context_updates={"review_session_id": review_session_id},
                agent_type=self.agent_type
            )


# =============================================================================
# Shared State Management
# =============================================================================

class SharedStateManager:
    """
    Manages shared state across all agents in the system.

    Provides thread-safe access to document contexts and
    coordination data.
    """

    def __init__(self):
        self._contexts: Dict[str, DocumentContext] = {}
        self._locks: Dict[str, asyncio.Lock] = {}
        self._global_lock = asyncio.Lock()
        self._observers: List[StateObserver] = []
        self.logger = logging.getLogger(f"{__name__}.SharedStateManager")

    async def get_context(self, document_id: str) -> Optional[DocumentContext]:
        """Get a document context by ID."""
        return self._contexts.get(document_id)

    async def set_context(self, context: DocumentContext) -> None:
        """Store or update a document context."""
        async with self._global_lock:
            if context.document_id not in self._locks:
                self._locks[context.document_id] = asyncio.Lock()

        async with self._locks[context.document_id]:
            old_context = self._contexts.get(context.document_id)
            self._contexts[context.document_id] = context

            if old_context and old_context.stage != context.stage:
                self._notify_observers(
                    context.document_id,
                    old_context.stage,
                    context.stage
                )

    async def update_context(
        self,
        document_id: str,
        updates: Dict[str, Any]
    ) -> Optional[DocumentContext]:
        """Update specific fields of a document context."""
        async with self._global_lock:
            if document_id not in self._locks:
                return None

        async with self._locks[document_id]:
            context = self._contexts.get(document_id)
            if not context:
                return None

            old_stage = context.stage

            for key, value in updates.items():
                if hasattr(context, key):
                    setattr(context, key, value)

            context.updated_at = datetime.utcnow()

            if old_stage != context.stage:
                self._notify_observers(document_id, old_stage, context.stage)

            return context

    async def delete_context(self, document_id: str) -> bool:
        """Delete a document context."""
        async with self._global_lock:
            if document_id in self._contexts:
                del self._contexts[document_id]
                if document_id in self._locks:
                    del self._locks[document_id]
                return True
            return False

    async def get_contexts_by_stage(
        self,
        stage: WorkflowStage
    ) -> List[DocumentContext]:
        """Get all contexts in a specific stage."""
        return [c for c in self._contexts.values() if c.stage == stage]

    async def get_contexts_by_status(
        self,
        status: DocumentStatus
    ) -> List[DocumentContext]:
        """Get all contexts with a specific status."""
        return [c for c in self._contexts.values() if c.status == status]

    def add_observer(self, observer: StateObserver) -> None:
        """Add a state observer."""
        self._observers.append(observer)

    def _notify_observers(
        self,
        document_id: str,
        old_stage: WorkflowStage,
        new_stage: WorkflowStage
    ) -> None:
        """Notify all observers of state change."""
        for observer in self._observers:
            try:
                observer.on_state_change(document_id, old_stage, new_stage)
            except Exception as e:
                self.logger.error(f"Observer notification failed: {e}")


# =============================================================================
# Message Bus
# =============================================================================

class MessageBus:
    """
    Central message bus for inter-agent communication.

    Provides publish-subscribe and direct messaging capabilities.
    """

    def __init__(self):
        self._queues: Dict[AgentType, asyncio.Queue[AgentMessage]] = {}
        self._subscribers: Dict[str, List[Callable]] = {}
        self._message_history: List[AgentMessage] = []
        self._max_history = 1000
        self.logger = logging.getLogger(f"{__name__}.MessageBus")

        # Initialize queues for all agent types
        for agent_type in AgentType:
            self._queues[agent_type] = asyncio.Queue()

    async def send(self, message: AgentMessage) -> None:
        """Send a message to a specific agent."""
        if message.recipient in self._queues:
            await self._queues[message.recipient].put(message)
            self._record_message(message)
            self.logger.debug(
                f"Message {message.id} sent from {message.sender} to {message.recipient}"
            )
        else:
            self.logger.error(f"Unknown recipient: {message.recipient}")

    async def receive(
        self,
        agent_type: AgentType,
        timeout: Optional[float] = None
    ) -> Optional[AgentMessage]:
        """Receive a message for a specific agent type."""
        try:
            if timeout:
                return await asyncio.wait_for(
                    self._queues[agent_type].get(),
                    timeout=timeout
                )
            return await self._queues[agent_type].get()
        except asyncio.TimeoutError:
            return None

    async def broadcast(
        self,
        sender: AgentType,
        payload: Dict[str, Any],
        priority: MessagePriority = MessagePriority.NORMAL
    ) -> None:
        """Broadcast a message to all agents."""
        for agent_type in AgentType:
            if agent_type != sender:
                message = AgentMessage(
                    sender=sender,
                    recipient=agent_type,
                    payload=payload,
                    priority=priority
                )
                await self.send(message)

    def subscribe(self, topic: str, callback: Callable) -> None:
        """Subscribe to a topic."""
        if topic not in self._subscribers:
            self._subscribers[topic] = []
        self._subscribers[topic].append(callback)

    async def publish(self, topic: str, data: Any) -> None:
        """Publish to a topic."""
        if topic in self._subscribers:
            for callback in self._subscribers[topic]:
                try:
                    if asyncio.iscoroutinefunction(callback):
                        await callback(data)
                    else:
                        callback(data)
                except Exception as e:
                    self.logger.error(f"Subscriber callback failed: {e}")

    def _record_message(self, message: AgentMessage) -> None:
        """Record message in history."""
        self._message_history.append(message)
        if len(self._message_history) > self._max_history:
            self._message_history = self._message_history[-self._max_history:]

    def get_message_history(
        self,
        correlation_id: Optional[str] = None
    ) -> List[AgentMessage]:
        """Get message history, optionally filtered by correlation ID."""
        if correlation_id:
            return [m for m in self._message_history if m.correlation_id == correlation_id]
        return self._message_history.copy()


# =============================================================================
# Retry and Error Handling
# =============================================================================

class RetryPolicy:
    """Configuration for retry behavior."""

    def __init__(
        self,
        max_retries: int = 3,
        base_delay_seconds: float = 1.0,
        max_delay_seconds: float = 60.0,
        exponential_backoff: bool = True,
        retry_on_exceptions: Optional[List[type]] = None
    ):
        self.max_retries = max_retries
        self.base_delay_seconds = base_delay_seconds
        self.max_delay_seconds = max_delay_seconds
        self.exponential_backoff = exponential_backoff
        self.retry_on_exceptions = retry_on_exceptions or [Exception]

    def get_delay(self, attempt: int) -> float:
        """Calculate delay for a given attempt number."""
        if self.exponential_backoff:
            delay = self.base_delay_seconds * (2 ** attempt)
        else:
            delay = self.base_delay_seconds
        return min(delay, self.max_delay_seconds)

    def should_retry(self, exception: Exception, attempt: int) -> bool:
        """Determine if should retry based on exception and attempt count."""
        if attempt >= self.max_retries:
            return False
        return any(isinstance(exception, exc_type) for exc_type in self.retry_on_exceptions)


class RetryableError(Exception):
    """Exception that indicates operation can be retried."""
    pass


class NonRetryableError(Exception):
    """Exception that indicates operation should not be retried."""
    pass


async def with_retry(
    operation: Callable,
    policy: RetryPolicy,
    *args,
    **kwargs
) -> Any:
    """
    Execute an operation with retry logic.

    Args:
        operation: Async function to execute
        policy: Retry policy configuration
        *args, **kwargs: Arguments to pass to operation

    Returns:
        Result of successful operation

    Raises:
        Last exception if all retries exhausted
    """
    last_exception = None

    for attempt in range(policy.max_retries + 1):
        try:
            return await operation(*args, **kwargs)
        except NonRetryableError:
            raise
        except Exception as e:
            last_exception = e

            if not policy.should_retry(e, attempt):
                raise

            delay = policy.get_delay(attempt)
            logger.warning(
                f"Attempt {attempt + 1} failed: {e}. Retrying in {delay:.1f}s"
            )
            await asyncio.sleep(delay)

    raise last_exception


# =============================================================================
# Workflow Pipeline
# =============================================================================

class WorkflowStep:
    """Represents a single step in the workflow pipeline."""

    def __init__(
        self,
        name: str,
        agent_type: AgentType,
        stage: WorkflowStage,
        next_stage_on_success: WorkflowStage,
        next_stage_on_failure: WorkflowStage = WorkflowStage.FAILED,
        retry_policy: Optional[RetryPolicy] = None,
        required: bool = True
    ):
        self.name = name
        self.agent_type = agent_type
        self.stage = stage
        self.next_stage_on_success = next_stage_on_success
        self.next_stage_on_failure = next_stage_on_failure
        self.retry_policy = retry_policy or RetryPolicy()
        self.required = required


class WorkflowPipeline:
    """
    Orchestrates the document processing workflow.

    Defines the sequence of agent processing steps and
    manages transitions between stages.
    """

    def __init__(self):
        self.steps: List[WorkflowStep] = []
        self._step_map: Dict[WorkflowStage, WorkflowStep] = {}
        self.logger = logging.getLogger(f"{__name__}.WorkflowPipeline")
        self._build_default_pipeline()

    def _build_default_pipeline(self) -> None:
        """Build the default document processing pipeline."""
        steps = [
            WorkflowStep(
                name="Data Ingestion",
                agent_type=AgentType.DATA_INGESTION,
                stage=WorkflowStage.DATA_INGESTION,
                next_stage_on_success=WorkflowStage.DOCUMENT_BUILDING
            ),
            WorkflowStep(
                name="Document Building",
                agent_type=AgentType.DOCUMENT_BUILDER,
                stage=WorkflowStage.DOCUMENT_BUILDING,
                next_stage_on_success=WorkflowStage.FORMAT_CHECK
            ),
            WorkflowStep(
                name="Format Check",
                agent_type=AgentType.FORMATTING_EDITOR,
                stage=WorkflowStage.FORMAT_CHECK,
                next_stage_on_success=WorkflowStage.LEGAL_PROOFING
            ),
            WorkflowStep(
                name="Legal Proofing",
                agent_type=AgentType.LEGAL_PROOFER,
                stage=WorkflowStage.LEGAL_PROOFING,
                next_stage_on_success=WorkflowStage.VISUAL_FORMATTING
            ),
            WorkflowStep(
                name="Visual Formatting",
                agent_type=AgentType.VISUAL_FORMATTER,
                stage=WorkflowStage.VISUAL_FORMATTING,
                next_stage_on_success=WorkflowStage.DRAFT_OUTPUT
            ),
            WorkflowStep(
                name="Draft Output",
                agent_type=AgentType.HUMAN_INTERFACE,
                stage=WorkflowStage.DRAFT_OUTPUT,
                next_stage_on_success=WorkflowStage.HUMAN_REVIEW
            ),
            WorkflowStep(
                name="Human Review",
                agent_type=AgentType.HUMAN_INTERFACE,
                stage=WorkflowStage.HUMAN_REVIEW,
                next_stage_on_success=WorkflowStage.LEARNING_INCORPORATION
            ),
            WorkflowStep(
                name="Learning Incorporation",
                agent_type=AgentType.LEARNING_COORDINATOR,
                stage=WorkflowStage.LEARNING_INCORPORATION,
                next_stage_on_success=WorkflowStage.FINALIZATION
            ),
            WorkflowStep(
                name="Finalization",
                agent_type=AgentType.VISUAL_FORMATTER,
                stage=WorkflowStage.FINALIZATION,
                next_stage_on_success=WorkflowStage.COMPLETED
            ),
        ]

        for step in steps:
            self.add_step(step)

    def add_step(self, step: WorkflowStep) -> None:
        """Add a step to the pipeline."""
        self.steps.append(step)
        self._step_map[step.stage] = step

    def get_step(self, stage: WorkflowStage) -> Optional[WorkflowStep]:
        """Get step for a given stage."""
        return self._step_map.get(stage)

    def get_next_stage(
        self,
        current_stage: WorkflowStage,
        success: bool
    ) -> WorkflowStage:
        """Get the next stage based on current stage and result."""
        step = self._step_map.get(current_stage)
        if not step:
            return WorkflowStage.FAILED
        return step.next_stage_on_success if success else step.next_stage_on_failure


# =============================================================================
# Document Automation System (Main Orchestrator)
# =============================================================================

class DocumentAutomationSystem:
    """
    Main orchestrator for the Pro Se Divorce Document Automation System.

    Coordinates all agents, manages workflow execution, and provides
    the primary interface for document processing.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.logger = logging.getLogger(f"{__name__}.DocumentAutomationSystem")

        # Core components
        self.state_manager = SharedStateManager()
        self.message_bus = MessageBus()
        self.pipeline = WorkflowPipeline()

        # Agent registry
        self._agents: Dict[AgentType, BaseAgent] = {}

        # Batch processing
        self._active_batches: Dict[str, BatchContext] = {}

        # System state
        self._is_running = False
        self._shutdown_event = asyncio.Event()

    def register_agent(self, agent: BaseAgent) -> None:
        """Register an agent with the system."""
        self._agents[agent.agent_type] = agent
        agent.add_observer(self.state_manager)
        self.logger.info(f"Registered agent: {agent.agent_type.name}")

    def get_agent(self, agent_type: AgentType) -> Optional[BaseAgent]:
        """Get a registered agent by type."""
        return self._agents.get(agent_type)

    async def initialize(self) -> None:
        """Initialize the system and all registered agents."""
        self.logger.info("Initializing Document Automation System")

        for agent in self._agents.values():
            await agent.initialize()

        self._is_running = True
        self.logger.info("System initialization complete")

    async def shutdown(self) -> None:
        """Shutdown the system gracefully."""
        self.logger.info("Shutting down Document Automation System")
        self._shutdown_event.set()

        for agent in self._agents.values():
            await agent.cleanup()

        self._is_running = False
        self.logger.info("System shutdown complete")

    async def process_document(
        self,
        context: DocumentContext
    ) -> DocumentContext:
        """
        Process a single document through the entire pipeline.

        Args:
            context: Document context to process

        Returns:
            Updated document context after processing
        """
        if not self._is_running:
            raise RuntimeError("System is not running")

        self.logger.info(f"Starting document processing: {context.document_id}")
        await self.state_manager.set_context(context)

        # Start from intake
        context.update_stage(WorkflowStage.DATA_INGESTION)

        while context.stage not in [WorkflowStage.COMPLETED, WorkflowStage.FAILED]:
            if self._shutdown_event.is_set():
                context.log_error("Processing interrupted by shutdown")
                context.update_stage(WorkflowStage.FAILED)
                break

            step = self.pipeline.get_step(context.stage)
            if not step:
                self.logger.error(f"No step defined for stage: {context.stage}")
                context.update_stage(WorkflowStage.FAILED)
                break

            agent = self._agents.get(step.agent_type)
            if not agent:
                self.logger.error(f"No agent registered for type: {step.agent_type}")
                context.update_stage(WorkflowStage.FAILED)
                break

            try:
                result = await with_retry(
                    agent.process,
                    step.retry_policy,
                    context
                )

                # Apply context updates from result
                if result.context_updates:
                    await self.state_manager.update_context(
                        context.document_id,
                        result.context_updates
                    )

                next_stage = self.pipeline.get_next_stage(
                    context.stage,
                    result.success
                )
                context.update_stage(next_stage)

                if not result.success and step.required:
                    context.log_error(f"Required step failed: {step.name}")
                    if result.errors:
                        for error in result.errors:
                            context.log_error(error)

            except Exception as e:
                self.logger.exception(f"Error in step {step.name}")
                context.log_error(f"{step.name}: {str(e)}")
                context.retry_count += 1

                if context.retry_count >= context.max_retries:
                    context.update_stage(WorkflowStage.FAILED)
                    context.status = DocumentStatus.ERROR

        # Update final state
        await self.state_manager.set_context(context)

        self.logger.info(
            f"Document processing complete: {context.document_id} "
            f"- Stage: {context.stage.value}"
        )

        return context

    async def process_batch(
        self,
        documents: List[DocumentContext],
        max_concurrent: int = 5
    ) -> BatchContext:
        """
        Process multiple documents in batch.

        Args:
            documents: List of document contexts to process
            max_concurrent: Maximum concurrent document processing

        Returns:
            Batch context with results
        """
        batch = BatchContext(
            documents=documents,
            total_count=len(documents),
            started_at=datetime.utcnow()
        )
        self._active_batches[batch.batch_id] = batch

        self.logger.info(
            f"Starting batch processing: {batch.batch_id} "
            f"({batch.total_count} documents)"
        )

        semaphore = asyncio.Semaphore(max_concurrent)

        async def process_with_semaphore(doc: DocumentContext):
            async with semaphore:
                if batch.is_cancelled:
                    return
                try:
                    result = await self.process_document(doc)
                    batch.processed_count += 1
                    if result.stage == WorkflowStage.COMPLETED:
                        batch.success_count += 1
                    else:
                        batch.failure_count += 1
                except Exception as e:
                    batch.processed_count += 1
                    batch.failure_count += 1
                    self.logger.error(f"Batch document failed: {e}")

        await asyncio.gather(
            *[process_with_semaphore(doc) for doc in documents],
            return_exceptions=True
        )

        batch.completed_at = datetime.utcnow()

        self.logger.info(
            f"Batch complete: {batch.batch_id} "
            f"- Success: {batch.success_count}/{batch.total_count}"
        )

        return batch

    async def cancel_batch(self, batch_id: str) -> bool:
        """Cancel a running batch."""
        if batch_id in self._active_batches:
            self._active_batches[batch_id].is_cancelled = True
            return True
        return False

    async def get_document_status(
        self,
        document_id: str
    ) -> Optional[Dict[str, Any]]:
        """Get current status of a document."""
        context = await self.state_manager.get_context(document_id)
        if not context:
            return None

        return {
            "document_id": context.document_id,
            "stage": context.stage.value,
            "status": context.status.value,
            "created_at": context.created_at.isoformat(),
            "updated_at": context.updated_at.isoformat(),
            "error_count": len(context.error_log),
            "retry_count": context.retry_count
        }

    async def submit_human_feedback(
        self,
        document_id: str,
        feedback: LearningFeedback
    ) -> bool:
        """Submit human feedback for a document."""
        context = await self.state_manager.get_context(document_id)
        if not context:
            return False

        if "human_feedback" not in context.metadata:
            context.metadata["human_feedback"] = []

        context.metadata["human_feedback"].append({
            "document_id": feedback.document_id,
            "reviewer_id": feedback.reviewer_id,
            "feedback_type": feedback.feedback_type,
            "original_content": feedback.original_content,
            "corrected_content": feedback.corrected_content,
            "comments": feedback.comments,
            "timestamp": feedback.timestamp.isoformat(),
            "severity": feedback.severity
        })

        await self.state_manager.set_context(context)

        # If approved, move to next stage
        if feedback.feedback_type == "approval":
            context.status = DocumentStatus.APPROVED
            context.update_stage(WorkflowStage.LEARNING_INCORPORATION)
        elif feedback.feedback_type == "rejection":
            context.status = DocumentStatus.REJECTED

        await self.state_manager.set_context(context)
        return True

    def get_system_status(self) -> Dict[str, Any]:
        """Get overall system status."""
        return {
            "is_running": self._is_running,
            "registered_agents": [at.name for at in self._agents.keys()],
            "active_batches": len(self._active_batches),
            "pipeline_steps": len(self.pipeline.steps)
        }


# =============================================================================
# Factory Functions
# =============================================================================

def create_default_system(config: Optional[Dict[str, Any]] = None) -> DocumentAutomationSystem:
    """
    Factory function to create a DocumentAutomationSystem with default configuration.

    Note: This returns a system without agents registered.
    Concrete agent implementations must be registered before use.
    """
    system = DocumentAutomationSystem(config)

    # Configure logging
    logging.basicConfig(
        level=config.get("log_level", logging.INFO) if config else logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    return system


def create_document_context(
    jurisdiction: str,
    document_type: str,
    petitioner_data: Dict[str, Any],
    respondent_data: Optional[Dict[str, Any]] = None,
    case_data: Optional[Dict[str, Any]] = None,
    **kwargs
) -> DocumentContext:
    """
    Factory function to create a properly configured DocumentContext.
    """
    return DocumentContext(
        jurisdiction=jurisdiction,
        document_type=document_type,
        petitioner_data=petitioner_data,
        respondent_data=respondent_data or {},
        case_data=case_data or {},
        metadata=kwargs.get("metadata", {}),
    )


# =============================================================================
# Module Exports
# =============================================================================

__all__ = [
    # Enums
    "AgentType",
    "MessagePriority",
    "WorkflowStage",
    "DocumentStatus",

    # Data Classes
    "AgentMessage",
    "DocumentContext",
    "BatchContext",
    "AgentResult",
    "LearningFeedback",

    # Base Classes
    "BaseAgent",
    "MessageHandler",
    "StateObserver",

    # Agent Interfaces
    "DataIngestionAgent",
    "DocumentBuilderAgent",
    "FormattingEditorAgent",
    "LegalProoferAgent",
    "VisualFormatterAgent",
    "LearningCoordinatorAgent",
    "HumanInterfaceAgent",

    # Core Components
    "SharedStateManager",
    "MessageBus",
    "WorkflowPipeline",
    "WorkflowStep",
    "DocumentAutomationSystem",

    # Error Handling
    "RetryPolicy",
    "RetryableError",
    "NonRetryableError",
    "with_retry",

    # Factory Functions
    "create_default_system",
    "create_document_context",
]

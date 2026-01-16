"""
Database models for the learning and self-improvement system.

Defines data structures for rules, feedback, patterns, and knowledge storage.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Any, Optional, List
from enum import Enum
import uuid


class RuleCategory(Enum):
    """Categories of learned rules."""
    FORMATTING = "formatting"
    LEGAL_CLAUSE = "legal_clause"
    SCENARIO = "scenario"
    STYLE = "style"
    STRUCTURE = "structure"
    CONTENT = "content"
    COMPLIANCE = "compliance"


class FeedbackType(Enum):
    """Types of feedback received."""
    CORRECTION = "correction"
    ADDITION = "addition"
    DELETION = "deletion"
    RESTRUCTURE = "restructure"
    STYLE_CHANGE = "style_change"
    CLARIFICATION = "clarification"
    APPROVAL = "approval"


class PatternStatus(Enum):
    """Status of identified patterns."""
    DETECTED = "detected"
    VALIDATED = "validated"
    ACTIVE = "active"
    DEPRECATED = "deprecated"
    SUPERSEDED = "superseded"


class RuleStatus(Enum):
    """Status of learning rules."""
    DRAFT = "draft"
    TESTING = "testing"
    ACTIVE = "active"
    DISABLED = "disabled"
    ARCHIVED = "archived"


@dataclass
class EditRecord:
    """Records a human edit made to a document."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    document_id: str = ""
    user_id: Optional[str] = None
    original_text: str = ""
    edited_text: str = ""
    edit_type: FeedbackType = FeedbackType.CORRECTION
    section: Optional[str] = None
    context: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Pattern:
    """Represents an identified pattern from edit analysis."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    description: str = ""
    category: RuleCategory = RuleCategory.FORMATTING
    trigger_conditions: List[str] = field(default_factory=list)
    occurrence_count: int = 0
    confidence_score: float = 0.0
    status: PatternStatus = PatternStatus.DETECTED
    examples: List[Dict[str, str]] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    last_seen: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Rule:
    """A rule generated from patterns or explicit configuration."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    description: str = ""
    category: RuleCategory = RuleCategory.FORMATTING
    condition: str = ""  # Condition that triggers the rule
    action: str = ""  # Action to take when triggered
    priority: int = 5  # 1-10, higher is more important
    status: RuleStatus = RuleStatus.DRAFT
    success_count: int = 0
    failure_count: int = 0
    success_rate: float = 1.0
    source_pattern_id: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def update_success_metrics(self, success: bool) -> None:
        """Update success metrics for this rule."""
        if success:
            self.success_count += 1
        else:
            self.failure_count += 1

        total = self.success_count + self.failure_count
        if total > 0:
            self.success_rate = self.success_count / total
        self.updated_at = datetime.now()


@dataclass
class KnowledgeEntry:
    """An entry in the knowledge base."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    entry_type: str = ""  # scenario, formatting, legal_clause, etc.
    trigger: str = ""  # What triggers this knowledge
    response: str = ""  # The knowledge content/response
    context_tags: List[str] = field(default_factory=list)
    embedding: Optional[List[float]] = None  # Vector embedding for search
    usage_count: int = 0
    effectiveness_score: float = 1.0
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Feedback:
    """Human feedback on a document or system output."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    document_id: Optional[str] = None
    rule_id: Optional[str] = None
    user_id: Optional[str] = None
    feedback_type: FeedbackType = FeedbackType.CORRECTION
    original_content: str = ""
    feedback_content: str = ""
    natural_language_notes: Optional[str] = None
    is_processed: bool = False
    extracted_rules: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ScenarioRule:
    """A scenario-based rule for document generation."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    scenario_name: str = ""  # e.g., "contested_custody"
    scenario_description: str = ""
    detection_criteria: Dict[str, Any] = field(default_factory=dict)  # How to detect this scenario
    actions: List[Dict[str, Any]] = field(default_factory=list)  # Actions to take
    document_types: List[str] = field(default_factory=list)  # Applicable document types
    priority: int = 5
    is_active: bool = True
    created_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AccuracyMetric:
    """Tracks accuracy metrics over time."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    metric_name: str = ""
    value: float = 0.0
    period_start: datetime = field(default_factory=datetime.now)
    period_end: datetime = field(default_factory=datetime.now)
    sample_size: int = 0
    category: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TrainingData:
    """Generated training data for model improvement."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    input_text: str = ""
    expected_output: str = ""
    context: Dict[str, Any] = field(default_factory=dict)
    category: str = ""
    quality_score: float = 1.0
    is_validated: bool = False
    created_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DocumentReview:
    """Post-document review analysis."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    document_id: str = ""
    total_corrections: int = 0
    correction_categories: Dict[str, int] = field(default_factory=dict)
    rules_applied: List[str] = field(default_factory=list)
    rules_failed: List[str] = field(default_factory=list)
    suggestions_generated: List[str] = field(default_factory=list)
    gaps_identified: List[str] = field(default_factory=list)
    overall_quality_score: float = 0.0
    reviewed_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

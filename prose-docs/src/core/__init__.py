"""
Core module for prose-docs document automation.

Contains the document processing engine, learning system, knowledge base,
and self-improvement components.
"""

# Document processing engine
from .document_engine import (
    DocumentEngine,
    TemplateProcessor,
    FormattingUtils,
    Margins,
    PlaceholderInfo,
    ConditionalOperator,
    DocumentEngineError,
    TemplateError,
    PlaceholderError,
    ExportError,
    create_document_from_template,
    get_template_fields,
)

# Learning components (lazy import to avoid errors if not yet created)
try:
    from .learning import (
        LearningCoordinator,
        KnowledgeBase,
        FeedbackProcessor,
        SelfReflection,
        ScenarioEngine,
    )
    _LEARNING_AVAILABLE = True
except ImportError:
    _LEARNING_AVAILABLE = False
    LearningCoordinator = None
    KnowledgeBase = None
    FeedbackProcessor = None
    SelfReflection = None
    ScenarioEngine = None

__all__ = [
    # Document Engine
    "DocumentEngine",
    "TemplateProcessor",
    "FormattingUtils",
    "Margins",
    "PlaceholderInfo",
    "ConditionalOperator",
    "DocumentEngineError",
    "TemplateError",
    "PlaceholderError",
    "ExportError",
    "create_document_from_template",
    "get_template_fields",
    # Learning (conditionally available)
    "LearningCoordinator",
    "KnowledgeBase",
    "FeedbackProcessor",
    "SelfReflection",
    "ScenarioEngine",
]

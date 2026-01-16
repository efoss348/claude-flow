"""
Core module for prose-docs document automation.

Contains the document processing engine, learning system, knowledge base,
workflow orchestration, and self-improvement components.
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

# Template analyzer
try:
    from .template_analyzer import (
        TemplateAnalyzer,
        TemplateAnalysis,
        TemplateFieldDefinition,
        DetectedPlaceholder,
        BracketStyle,
    )
    _ANALYZER_AVAILABLE = True
except ImportError:
    _ANALYZER_AVAILABLE = False
    TemplateAnalyzer = None
    TemplateAnalysis = None
    TemplateFieldDefinition = None
    DetectedPlaceholder = None
    BracketStyle = None

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

# Workflow orchestration
try:
    from .workflow import (
        DocumentWorkflow,
        BatchProcessor,
        WorkflowConfig,
        WorkflowResult,
        WorkflowStep,
        BatchConfig,
        BatchResult,
        OutputFormat,
        ValidationIssue,
        ValidationSeverity,
        StepResult,
        generate_document,
        batch_generate,
    )
    _WORKFLOW_AVAILABLE = True
except ImportError:
    _WORKFLOW_AVAILABLE = False
    DocumentWorkflow = None
    BatchProcessor = None
    WorkflowConfig = None
    WorkflowResult = None
    WorkflowStep = None
    BatchConfig = None
    BatchResult = None
    OutputFormat = None
    ValidationIssue = None
    ValidationSeverity = None
    StepResult = None
    generate_document = None
    batch_generate = None

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
    # Template Analyzer
    "TemplateAnalyzer",
    "TemplateAnalysis",
    "TemplateFieldDefinition",
    "DetectedPlaceholder",
    "BracketStyle",
    # Learning (conditionally available)
    "LearningCoordinator",
    "KnowledgeBase",
    "FeedbackProcessor",
    "SelfReflection",
    "ScenarioEngine",
    # Workflow Orchestration
    "DocumentWorkflow",
    "BatchProcessor",
    "WorkflowConfig",
    "WorkflowResult",
    "WorkflowStep",
    "BatchConfig",
    "BatchResult",
    "OutputFormat",
    "ValidationIssue",
    "ValidationSeverity",
    "StepResult",
    "generate_document",
    "batch_generate",
]

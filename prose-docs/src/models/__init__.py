"""
Models module for prose-docs document automation.

Contains data models for learning, knowledge storage, document processing,
and database persistence with SQLAlchemy.
"""

from .learning_models import (
    RuleCategory,
    FeedbackType,
    PatternStatus,
    RuleStatus,
    EditRecord,
    Pattern,
    Rule,
    KnowledgeEntry,
    Feedback,
    ScenarioRule,
    AccuracyMetric,
    TrainingData,
    DocumentReview,
)

from .database import (
    # Base
    Base,

    # Enums
    TemplateCategory,
    DocumentFormat,
    DocumentStatus,
    CaseStatus,
    RuleType,
    RuleSource,
    EditType,
    EditMadeBy,

    # Models
    Template,
    Customer,
    GeneratedDocument,
    LearningRule,
    EditHistory,

    # Database initialization
    init_database,
    get_engine,
    get_session,
    session_scope,

    # Template CRUD
    create_template,
    get_template_by_id,
    get_template_by_name,
    get_templates_by_category,
    get_all_templates,
    update_template,
    delete_template,

    # Customer CRUD
    create_customer,
    get_customer_by_id,
    get_customer_by_case_number,
    get_customers_by_status,
    get_all_customers,
    search_customers,
    update_customer,
    delete_customer,

    # Generated Document CRUD
    create_generated_document,
    get_document_by_id,
    get_documents_by_customer,
    get_documents_by_template,
    get_documents_pending_review,
    get_documents_by_status,
    update_document,
    mark_document_reviewed,

    # Learning Rule CRUD
    create_learning_rule,
    get_learning_rule_by_id,
    get_learning_rules_by_type,
    get_active_learning_rules,
    get_top_performing_rules,
    update_learning_rule,
    record_rule_application,
    deactivate_low_performing_rules,

    # Edit History CRUD
    create_edit_history,
    get_edit_history_by_id,
    get_edit_history_by_document,
    get_unlearned_edits,
    get_edits_by_type,
    get_human_edits,
    mark_edit_as_learned,

    # Query Helpers
    get_edit_patterns,
    get_template_usage_stats,
    get_rule_effectiveness_summary,
    find_similar_edits,
    get_customer_document_summary,

    # Migration
    get_schema_version,
    set_schema_version,
    run_migrations,

    # Utilities
    export_data,
    get_database_stats,
)

__all__ = [
    # Learning Models (existing)
    "RuleCategory",
    "FeedbackType",
    "PatternStatus",
    "RuleStatus",
    "EditRecord",
    "Pattern",
    "Rule",
    "KnowledgeEntry",
    "Feedback",
    "ScenarioRule",
    "AccuracyMetric",
    "TrainingData",
    "DocumentReview",

    # Database Base
    "Base",

    # Database Enums
    "TemplateCategory",
    "DocumentFormat",
    "DocumentStatus",
    "CaseStatus",
    "RuleType",
    "RuleSource",
    "EditType",
    "EditMadeBy",

    # Database Models
    "Template",
    "Customer",
    "GeneratedDocument",
    "LearningRule",
    "EditHistory",

    # Database initialization
    "init_database",
    "get_engine",
    "get_session",
    "session_scope",

    # Template CRUD
    "create_template",
    "get_template_by_id",
    "get_template_by_name",
    "get_templates_by_category",
    "get_all_templates",
    "update_template",
    "delete_template",

    # Customer CRUD
    "create_customer",
    "get_customer_by_id",
    "get_customer_by_case_number",
    "get_customers_by_status",
    "get_all_customers",
    "search_customers",
    "update_customer",
    "delete_customer",

    # Generated Document CRUD
    "create_generated_document",
    "get_document_by_id",
    "get_documents_by_customer",
    "get_documents_by_template",
    "get_documents_pending_review",
    "get_documents_by_status",
    "update_document",
    "mark_document_reviewed",

    # Learning Rule CRUD
    "create_learning_rule",
    "get_learning_rule_by_id",
    "get_learning_rules_by_type",
    "get_active_learning_rules",
    "get_top_performing_rules",
    "update_learning_rule",
    "record_rule_application",
    "deactivate_low_performing_rules",

    # Edit History CRUD
    "create_edit_history",
    "get_edit_history_by_id",
    "get_edit_history_by_document",
    "get_unlearned_edits",
    "get_edits_by_type",
    "get_human_edits",
    "mark_edit_as_learned",

    # Query Helpers
    "get_edit_patterns",
    "get_template_usage_stats",
    "get_rule_effectiveness_summary",
    "find_similar_edits",
    "get_customer_document_summary",

    # Migration
    "get_schema_version",
    "set_schema_version",
    "run_migrations",

    # Utilities
    "export_data",
    "get_database_stats",
]

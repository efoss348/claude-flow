"""
Pro Se Document Automation System - Database Models

SQLAlchemy models for template management, customer data, document generation,
learning rules, and edit history tracking.
"""

import json
from datetime import datetime
from typing import Any, Dict, List, Optional
from contextlib import contextmanager

from sqlalchemy import (
    create_engine,
    Column,
    Integer,
    String,
    Text,
    LargeBinary,
    DateTime,
    Boolean,
    Float,
    ForeignKey,
    Index,
    JSON,
    Enum as SQLEnum,
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship, Session
from sqlalchemy.engine import Engine
import enum


Base = declarative_base()


# ============================================================================
# Enums
# ============================================================================

class TemplateCategory(enum.Enum):
    """Categories for legal document templates."""
    PETITION = "petition"
    RESPONSE = "response"
    DECREE = "decree"
    MOTION = "motion"
    AGREEMENT = "agreement"
    AFFIDAVIT = "affidavit"
    NOTICE = "notice"
    ORDER = "order"
    SUMMONS = "summons"
    STIPULATION = "stipulation"
    OTHER = "other"


class DocumentFormat(enum.Enum):
    """Output formats for generated documents."""
    DOCX = "docx"
    PDF = "pdf"


class DocumentStatus(enum.Enum):
    """Status of generated documents."""
    DRAFT = "draft"
    PENDING_REVIEW = "pending_review"
    APPROVED = "approved"
    REJECTED = "rejected"
    FINALIZED = "finalized"


class CaseStatus(enum.Enum):
    """Status of customer cases."""
    INTAKE = "intake"
    IN_PROGRESS = "in_progress"
    PENDING_DOCUMENTS = "pending_documents"
    PENDING_FILING = "pending_filing"
    FILED = "filed"
    COMPLETED = "completed"
    CLOSED = "closed"


class RuleType(enum.Enum):
    """Types of learning rules."""
    FORMATTING = "formatting"
    LEGAL = "legal"
    SCENARIO = "scenario"
    VALIDATION = "validation"
    STYLE = "style"


class RuleSource(enum.Enum):
    """Source of learning rules."""
    HUMAN_FEEDBACK = "human_feedback"
    AUTO_DETECTED = "auto_detected"
    IMPORTED = "imported"
    SYSTEM_DEFAULT = "system_default"


class EditType(enum.Enum):
    """Types of document edits."""
    INSERTION = "insertion"
    DELETION = "deletion"
    REPLACEMENT = "replacement"
    FORMATTING = "formatting"
    REORDERING = "reordering"


class EditMadeBy(enum.Enum):
    """Who made the edit."""
    HUMAN = "human"
    AGENT = "agent"
    SYSTEM = "system"


# ============================================================================
# Models
# ============================================================================

class Template(Base):
    """
    Document template model.

    Stores Word document templates with placeholder tags like [PETITIONER_NAME].
    """
    __tablename__ = "templates"

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(255), nullable=False, unique=True)
    version = Column(String(50), nullable=False, default="1.0.0")
    content = Column(LargeBinary, nullable=False)  # Binary Word doc content
    placeholders = Column(JSON, nullable=False, default=list)  # List of bracket tags
    category = Column(SQLEnum(TemplateCategory), nullable=False, default=TemplateCategory.OTHER)
    description = Column(Text, nullable=True)
    is_active = Column(Boolean, nullable=False, default=True)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    updated_at = Column(DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    documents = relationship("GeneratedDocument", back_populates="template", cascade="all, delete-orphan")

    __table_args__ = (
        Index("idx_template_category", "category"),
        Index("idx_template_active", "is_active"),
        Index("idx_template_name", "name"),
    )

    def __repr__(self) -> str:
        return f"<Template(id={self.id}, name='{self.name}', category={self.category.value})>"

    def to_dict(self) -> Dict[str, Any]:
        """Convert template to dictionary (excluding binary content)."""
        return {
            "id": self.id,
            "name": self.name,
            "version": self.version,
            "placeholders": self.placeholders,
            "category": self.category.value,
            "description": self.description,
            "is_active": self.is_active,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }


class Customer(Base):
    """
    Customer/case model.

    Stores all case information for pro se divorce proceedings.
    """
    __tablename__ = "customers"

    id = Column(Integer, primary_key=True, autoincrement=True)
    case_number = Column(String(100), nullable=True, unique=True)

    # Parties
    petitioner_name = Column(String(255), nullable=False)
    petitioner_address = Column(Text, nullable=True)
    petitioner_phone = Column(String(50), nullable=True)
    petitioner_email = Column(String(255), nullable=True)

    respondent_name = Column(String(255), nullable=True)
    respondent_address = Column(Text, nullable=True)
    respondent_phone = Column(String(50), nullable=True)
    respondent_email = Column(String(255), nullable=True)

    # Key dates
    marriage_date = Column(DateTime, nullable=True)
    separation_date = Column(DateTime, nullable=True)
    filing_date = Column(DateTime, nullable=True)

    # Case details (JSON for flexibility)
    children = Column(JSON, nullable=False, default=list)  # List of child details
    assets = Column(JSON, nullable=False, default=list)  # List of assets
    debts = Column(JSON, nullable=False, default=list)  # List of debts

    # Arrangements
    custody_arrangement = Column(Text, nullable=True)
    support_details = Column(Text, nullable=True)
    property_division = Column(Text, nullable=True)

    # Additional case info
    jurisdiction = Column(String(100), nullable=True)
    county = Column(String(100), nullable=True)
    grounds_for_divorce = Column(Text, nullable=True)

    # Status tracking
    status = Column(SQLEnum(CaseStatus), nullable=False, default=CaseStatus.INTAKE)
    notes = Column(Text, nullable=True)

    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    updated_at = Column(DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    documents = relationship("GeneratedDocument", back_populates="customer", cascade="all, delete-orphan")

    __table_args__ = (
        Index("idx_customer_case_number", "case_number"),
        Index("idx_customer_status", "status"),
        Index("idx_customer_petitioner", "petitioner_name"),
        Index("idx_customer_created", "created_at"),
    )

    def __repr__(self) -> str:
        return f"<Customer(id={self.id}, case='{self.case_number}', petitioner='{self.petitioner_name}')>"

    def to_dict(self) -> Dict[str, Any]:
        """Convert customer to dictionary."""
        return {
            "id": self.id,
            "case_number": self.case_number,
            "petitioner_name": self.petitioner_name,
            "petitioner_address": self.petitioner_address,
            "petitioner_phone": self.petitioner_phone,
            "petitioner_email": self.petitioner_email,
            "respondent_name": self.respondent_name,
            "respondent_address": self.respondent_address,
            "respondent_phone": self.respondent_phone,
            "respondent_email": self.respondent_email,
            "marriage_date": self.marriage_date.isoformat() if self.marriage_date else None,
            "separation_date": self.separation_date.isoformat() if self.separation_date else None,
            "filing_date": self.filing_date.isoformat() if self.filing_date else None,
            "children": self.children,
            "assets": self.assets,
            "debts": self.debts,
            "custody_arrangement": self.custody_arrangement,
            "support_details": self.support_details,
            "property_division": self.property_division,
            "jurisdiction": self.jurisdiction,
            "county": self.county,
            "grounds_for_divorce": self.grounds_for_divorce,
            "status": self.status.value,
            "notes": self.notes,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }

    def get_placeholder_values(self) -> Dict[str, str]:
        """Get dictionary of placeholder tag values for document generation."""
        values = {
            "PETITIONER_NAME": self.petitioner_name or "",
            "PETITIONER_ADDRESS": self.petitioner_address or "",
            "PETITIONER_PHONE": self.petitioner_phone or "",
            "PETITIONER_EMAIL": self.petitioner_email or "",
            "RESPONDENT_NAME": self.respondent_name or "",
            "RESPONDENT_ADDRESS": self.respondent_address or "",
            "RESPONDENT_PHONE": self.respondent_phone or "",
            "RESPONDENT_EMAIL": self.respondent_email or "",
            "CASE_NUMBER": self.case_number or "",
            "MARRIAGE_DATE": self.marriage_date.strftime("%B %d, %Y") if self.marriage_date else "",
            "SEPARATION_DATE": self.separation_date.strftime("%B %d, %Y") if self.separation_date else "",
            "FILING_DATE": self.filing_date.strftime("%B %d, %Y") if self.filing_date else "",
            "CUSTODY_ARRANGEMENT": self.custody_arrangement or "",
            "SUPPORT_DETAILS": self.support_details or "",
            "PROPERTY_DIVISION": self.property_division or "",
            "JURISDICTION": self.jurisdiction or "",
            "COUNTY": self.county or "",
            "GROUNDS_FOR_DIVORCE": self.grounds_for_divorce or "",
            "CURRENT_DATE": datetime.now().strftime("%B %d, %Y"),
        }

        # Add children count
        values["NUM_CHILDREN"] = str(len(self.children)) if self.children else "0"

        return values


class GeneratedDocument(Base):
    """
    Generated document model.

    Tracks documents generated from templates for customers.
    """
    __tablename__ = "generated_documents"

    id = Column(Integer, primary_key=True, autoincrement=True)
    template_id = Column(Integer, ForeignKey("templates.id"), nullable=False)
    customer_id = Column(Integer, ForeignKey("customers.id"), nullable=False)

    output_path = Column(String(500), nullable=False)
    format = Column(SQLEnum(DocumentFormat), nullable=False, default=DocumentFormat.DOCX)
    version = Column(String(50), nullable=False, default="1")
    status = Column(SQLEnum(DocumentStatus), nullable=False, default=DocumentStatus.DRAFT)

    # Review tracking
    human_reviewed = Column(Boolean, nullable=False, default=False)
    review_notes = Column(Text, nullable=True)
    reviewed_by = Column(String(255), nullable=True)
    reviewed_at = Column(DateTime, nullable=True)

    # Generation metadata
    generation_params = Column(JSON, nullable=True)  # Parameters used for generation
    error_log = Column(Text, nullable=True)

    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    updated_at = Column(DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    template = relationship("Template", back_populates="documents")
    customer = relationship("Customer", back_populates="documents")
    edit_history = relationship("EditHistory", back_populates="document", cascade="all, delete-orphan")

    __table_args__ = (
        Index("idx_gendoc_template", "template_id"),
        Index("idx_gendoc_customer", "customer_id"),
        Index("idx_gendoc_status", "status"),
        Index("idx_gendoc_reviewed", "human_reviewed"),
        Index("idx_gendoc_created", "created_at"),
    )

    def __repr__(self) -> str:
        return f"<GeneratedDocument(id={self.id}, template_id={self.template_id}, status={self.status.value})>"

    def to_dict(self) -> Dict[str, Any]:
        """Convert generated document to dictionary."""
        return {
            "id": self.id,
            "template_id": self.template_id,
            "customer_id": self.customer_id,
            "output_path": self.output_path,
            "format": self.format.value,
            "version": self.version,
            "status": self.status.value,
            "human_reviewed": self.human_reviewed,
            "review_notes": self.review_notes,
            "reviewed_by": self.reviewed_by,
            "reviewed_at": self.reviewed_at.isoformat() if self.reviewed_at else None,
            "generation_params": self.generation_params,
            "error_log": self.error_log,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }


class LearningRule(Base):
    """
    Learning rule model.

    Stores rules learned from human feedback and auto-detection.
    """
    __tablename__ = "learning_rules"

    id = Column(Integer, primary_key=True, autoincrement=True)
    rule_type = Column(SQLEnum(RuleType), nullable=False)
    name = Column(String(255), nullable=False)
    description = Column(Text, nullable=True)

    # Rule definition
    trigger_condition = Column(JSON, nullable=False)  # Conditions that trigger this rule
    action = Column(Text, nullable=False)  # What action to take
    action_params = Column(JSON, nullable=True)  # Additional parameters for the action

    # Source and context
    source = Column(SQLEnum(RuleSource), nullable=False, default=RuleSource.AUTO_DETECTED)
    context = Column(JSON, nullable=True)  # Additional context about when rule applies

    # Effectiveness tracking
    times_applied = Column(Integer, nullable=False, default=0)
    times_successful = Column(Integer, nullable=False, default=0)
    times_rejected = Column(Integer, nullable=False, default=0)
    success_rate = Column(Float, nullable=False, default=0.0)

    # Status
    is_active = Column(Boolean, nullable=False, default=True)
    priority = Column(Integer, nullable=False, default=0)  # Higher = more priority

    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    updated_at = Column(DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow)

    __table_args__ = (
        Index("idx_rule_type", "rule_type"),
        Index("idx_rule_source", "source"),
        Index("idx_rule_active", "is_active"),
        Index("idx_rule_success_rate", "success_rate"),
        Index("idx_rule_priority", "priority"),
    )

    def __repr__(self) -> str:
        return f"<LearningRule(id={self.id}, type={self.rule_type.value}, success_rate={self.success_rate:.2f})>"

    def to_dict(self) -> Dict[str, Any]:
        """Convert learning rule to dictionary."""
        return {
            "id": self.id,
            "rule_type": self.rule_type.value,
            "name": self.name,
            "description": self.description,
            "trigger_condition": self.trigger_condition,
            "action": self.action,
            "action_params": self.action_params,
            "source": self.source.value,
            "context": self.context,
            "times_applied": self.times_applied,
            "times_successful": self.times_successful,
            "times_rejected": self.times_rejected,
            "success_rate": self.success_rate,
            "is_active": self.is_active,
            "priority": self.priority,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }

    def update_success_rate(self) -> None:
        """Recalculate success rate based on application history."""
        if self.times_applied > 0:
            self.success_rate = self.times_successful / self.times_applied
        else:
            self.success_rate = 0.0

    def record_application(self, successful: bool) -> None:
        """Record a rule application and update statistics."""
        self.times_applied += 1
        if successful:
            self.times_successful += 1
        else:
            self.times_rejected += 1
        self.update_success_rate()


class EditHistory(Base):
    """
    Edit history model.

    Tracks all edits made to generated documents for learning.
    """
    __tablename__ = "edit_history"

    id = Column(Integer, primary_key=True, autoincrement=True)
    document_id = Column(Integer, ForeignKey("generated_documents.id"), nullable=False)

    edit_type = Column(SQLEnum(EditType), nullable=False)
    original_content = Column(Text, nullable=True)
    new_content = Column(Text, nullable=True)

    # Location in document
    location_section = Column(String(255), nullable=True)
    location_paragraph = Column(Integer, nullable=True)
    location_bullet = Column(Integer, nullable=True)
    location_detail = Column(JSON, nullable=True)  # Additional location info

    # Who made the edit
    made_by = Column(SQLEnum(EditMadeBy), nullable=False)
    made_by_name = Column(String(255), nullable=True)

    # Learning integration
    learned = Column(Boolean, nullable=False, default=False)
    learning_rule_id = Column(Integer, ForeignKey("learning_rules.id"), nullable=True)
    learning_notes = Column(Text, nullable=True)

    # Context
    reason = Column(Text, nullable=True)
    context = Column(JSON, nullable=True)

    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)

    # Relationships
    document = relationship("GeneratedDocument", back_populates="edit_history")
    learning_rule = relationship("LearningRule")

    __table_args__ = (
        Index("idx_edit_document", "document_id"),
        Index("idx_edit_type", "edit_type"),
        Index("idx_edit_made_by", "made_by"),
        Index("idx_edit_learned", "learned"),
        Index("idx_edit_created", "created_at"),
    )

    def __repr__(self) -> str:
        return f"<EditHistory(id={self.id}, doc_id={self.document_id}, type={self.edit_type.value})>"

    def to_dict(self) -> Dict[str, Any]:
        """Convert edit history to dictionary."""
        return {
            "id": self.id,
            "document_id": self.document_id,
            "edit_type": self.edit_type.value,
            "original_content": self.original_content,
            "new_content": self.new_content,
            "location_section": self.location_section,
            "location_paragraph": self.location_paragraph,
            "location_bullet": self.location_bullet,
            "location_detail": self.location_detail,
            "made_by": self.made_by.value,
            "made_by_name": self.made_by_name,
            "learned": self.learned,
            "learning_rule_id": self.learning_rule_id,
            "learning_notes": self.learning_notes,
            "reason": self.reason,
            "context": self.context,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


# ============================================================================
# Database Initialization
# ============================================================================

_engine: Optional[Engine] = None
_SessionLocal: Optional[sessionmaker] = None


def init_database(database_url: str = "sqlite:///prose_docs.db", echo: bool = False) -> Engine:
    """
    Initialize the database engine and create all tables.

    Args:
        database_url: SQLAlchemy database URL
        echo: Whether to echo SQL statements

    Returns:
        SQLAlchemy Engine instance
    """
    global _engine, _SessionLocal

    _engine = create_engine(
        database_url,
        echo=echo,
        connect_args={"check_same_thread": False} if "sqlite" in database_url else {},
    )

    # Create all tables
    Base.metadata.create_all(bind=_engine)

    _SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=_engine)

    return _engine


def get_engine() -> Engine:
    """Get the database engine, initializing if necessary."""
    global _engine
    if _engine is None:
        init_database()
    return _engine


def get_session() -> Session:
    """Get a new database session."""
    global _SessionLocal
    if _SessionLocal is None:
        init_database()
    return _SessionLocal()


@contextmanager
def session_scope():
    """Context manager for database sessions with automatic commit/rollback."""
    session = get_session()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


# ============================================================================
# CRUD Operations - Templates
# ============================================================================

def create_template(
    session: Session,
    name: str,
    content: bytes,
    placeholders: List[str],
    category: TemplateCategory,
    version: str = "1.0.0",
    description: Optional[str] = None,
) -> Template:
    """Create a new template."""
    template = Template(
        name=name,
        content=content,
        placeholders=placeholders,
        category=category,
        version=version,
        description=description,
    )
    session.add(template)
    session.flush()
    return template


def get_template_by_id(session: Session, template_id: int) -> Optional[Template]:
    """Get a template by ID."""
    return session.query(Template).filter(Template.id == template_id).first()


def get_template_by_name(session: Session, name: str) -> Optional[Template]:
    """Get a template by name."""
    return session.query(Template).filter(Template.name == name).first()


def get_templates_by_category(
    session: Session,
    category: TemplateCategory,
    active_only: bool = True,
) -> List[Template]:
    """Get all templates in a category."""
    query = session.query(Template).filter(Template.category == category)
    if active_only:
        query = query.filter(Template.is_active == True)
    return query.order_by(Template.name).all()


def get_all_templates(session: Session, active_only: bool = True) -> List[Template]:
    """Get all templates."""
    query = session.query(Template)
    if active_only:
        query = query.filter(Template.is_active == True)
    return query.order_by(Template.category, Template.name).all()


def update_template(
    session: Session,
    template_id: int,
    **kwargs,
) -> Optional[Template]:
    """Update a template."""
    template = get_template_by_id(session, template_id)
    if template:
        for key, value in kwargs.items():
            if hasattr(template, key):
                setattr(template, key, value)
        session.flush()
    return template


def delete_template(session: Session, template_id: int) -> bool:
    """Delete a template (soft delete by setting is_active=False)."""
    template = get_template_by_id(session, template_id)
    if template:
        template.is_active = False
        session.flush()
        return True
    return False


# ============================================================================
# CRUD Operations - Customers
# ============================================================================

def create_customer(
    session: Session,
    petitioner_name: str,
    case_number: Optional[str] = None,
    **kwargs,
) -> Customer:
    """Create a new customer."""
    customer = Customer(
        petitioner_name=petitioner_name,
        case_number=case_number,
        **kwargs,
    )
    session.add(customer)
    session.flush()
    return customer


def get_customer_by_id(session: Session, customer_id: int) -> Optional[Customer]:
    """Get a customer by ID."""
    return session.query(Customer).filter(Customer.id == customer_id).first()


def get_customer_by_case_number(session: Session, case_number: str) -> Optional[Customer]:
    """Get a customer by case number."""
    return session.query(Customer).filter(Customer.case_number == case_number).first()


def get_customers_by_status(session: Session, status: CaseStatus) -> List[Customer]:
    """Get all customers with a specific status."""
    return session.query(Customer).filter(Customer.status == status).order_by(Customer.created_at.desc()).all()


def get_all_customers(session: Session, limit: int = 100) -> List[Customer]:
    """Get all customers."""
    return session.query(Customer).order_by(Customer.created_at.desc()).limit(limit).all()


def search_customers(session: Session, search_term: str) -> List[Customer]:
    """Search customers by name or case number."""
    search_pattern = f"%{search_term}%"
    return session.query(Customer).filter(
        (Customer.petitioner_name.ilike(search_pattern)) |
        (Customer.respondent_name.ilike(search_pattern)) |
        (Customer.case_number.ilike(search_pattern))
    ).order_by(Customer.created_at.desc()).all()


def update_customer(
    session: Session,
    customer_id: int,
    **kwargs,
) -> Optional[Customer]:
    """Update a customer."""
    customer = get_customer_by_id(session, customer_id)
    if customer:
        for key, value in kwargs.items():
            if hasattr(customer, key):
                setattr(customer, key, value)
        session.flush()
    return customer


def delete_customer(session: Session, customer_id: int) -> bool:
    """Delete a customer."""
    customer = get_customer_by_id(session, customer_id)
    if customer:
        session.delete(customer)
        session.flush()
        return True
    return False


# ============================================================================
# CRUD Operations - Generated Documents
# ============================================================================

def create_generated_document(
    session: Session,
    template_id: int,
    customer_id: int,
    output_path: str,
    format: DocumentFormat = DocumentFormat.DOCX,
    **kwargs,
) -> GeneratedDocument:
    """Create a new generated document record."""
    document = GeneratedDocument(
        template_id=template_id,
        customer_id=customer_id,
        output_path=output_path,
        format=format,
        **kwargs,
    )
    session.add(document)
    session.flush()
    return document


def get_document_by_id(session: Session, document_id: int) -> Optional[GeneratedDocument]:
    """Get a generated document by ID."""
    return session.query(GeneratedDocument).filter(GeneratedDocument.id == document_id).first()


def get_documents_by_customer(session: Session, customer_id: int) -> List[GeneratedDocument]:
    """Get all documents for a customer."""
    return session.query(GeneratedDocument).filter(
        GeneratedDocument.customer_id == customer_id
    ).order_by(GeneratedDocument.created_at.desc()).all()


def get_documents_by_template(session: Session, template_id: int) -> List[GeneratedDocument]:
    """Get all documents generated from a template."""
    return session.query(GeneratedDocument).filter(
        GeneratedDocument.template_id == template_id
    ).order_by(GeneratedDocument.created_at.desc()).all()


def get_documents_pending_review(session: Session) -> List[GeneratedDocument]:
    """Get all documents pending review."""
    return session.query(GeneratedDocument).filter(
        GeneratedDocument.status == DocumentStatus.PENDING_REVIEW,
        GeneratedDocument.human_reviewed == False,
    ).order_by(GeneratedDocument.created_at).all()


def get_documents_by_status(session: Session, status: DocumentStatus) -> List[GeneratedDocument]:
    """Get all documents with a specific status."""
    return session.query(GeneratedDocument).filter(
        GeneratedDocument.status == status
    ).order_by(GeneratedDocument.created_at.desc()).all()


def update_document(
    session: Session,
    document_id: int,
    **kwargs,
) -> Optional[GeneratedDocument]:
    """Update a generated document."""
    document = get_document_by_id(session, document_id)
    if document:
        for key, value in kwargs.items():
            if hasattr(document, key):
                setattr(document, key, value)
        session.flush()
    return document


def mark_document_reviewed(
    session: Session,
    document_id: int,
    reviewed_by: str,
    review_notes: Optional[str] = None,
    approved: bool = True,
) -> Optional[GeneratedDocument]:
    """Mark a document as reviewed."""
    document = get_document_by_id(session, document_id)
    if document:
        document.human_reviewed = True
        document.reviewed_by = reviewed_by
        document.reviewed_at = datetime.utcnow()
        document.review_notes = review_notes
        document.status = DocumentStatus.APPROVED if approved else DocumentStatus.REJECTED
        session.flush()
    return document


# ============================================================================
# CRUD Operations - Learning Rules
# ============================================================================

def create_learning_rule(
    session: Session,
    rule_type: RuleType,
    name: str,
    trigger_condition: Dict[str, Any],
    action: str,
    source: RuleSource = RuleSource.AUTO_DETECTED,
    **kwargs,
) -> LearningRule:
    """Create a new learning rule."""
    rule = LearningRule(
        rule_type=rule_type,
        name=name,
        trigger_condition=trigger_condition,
        action=action,
        source=source,
        **kwargs,
    )
    session.add(rule)
    session.flush()
    return rule


def get_learning_rule_by_id(session: Session, rule_id: int) -> Optional[LearningRule]:
    """Get a learning rule by ID."""
    return session.query(LearningRule).filter(LearningRule.id == rule_id).first()


def get_learning_rules_by_type(
    session: Session,
    rule_type: RuleType,
    active_only: bool = True,
) -> List[LearningRule]:
    """Get all learning rules of a specific type."""
    query = session.query(LearningRule).filter(LearningRule.rule_type == rule_type)
    if active_only:
        query = query.filter(LearningRule.is_active == True)
    return query.order_by(LearningRule.priority.desc(), LearningRule.success_rate.desc()).all()


def get_active_learning_rules(
    session: Session,
    min_success_rate: float = 0.0,
) -> List[LearningRule]:
    """Get all active learning rules above a minimum success rate."""
    return session.query(LearningRule).filter(
        LearningRule.is_active == True,
        LearningRule.success_rate >= min_success_rate,
    ).order_by(LearningRule.priority.desc(), LearningRule.success_rate.desc()).all()


def get_top_performing_rules(
    session: Session,
    limit: int = 10,
    min_applications: int = 5,
) -> List[LearningRule]:
    """Get top performing learning rules."""
    return session.query(LearningRule).filter(
        LearningRule.is_active == True,
        LearningRule.times_applied >= min_applications,
    ).order_by(LearningRule.success_rate.desc()).limit(limit).all()


def update_learning_rule(
    session: Session,
    rule_id: int,
    **kwargs,
) -> Optional[LearningRule]:
    """Update a learning rule."""
    rule = get_learning_rule_by_id(session, rule_id)
    if rule:
        for key, value in kwargs.items():
            if hasattr(rule, key):
                setattr(rule, key, value)
        session.flush()
    return rule


def record_rule_application(
    session: Session,
    rule_id: int,
    successful: bool,
) -> Optional[LearningRule]:
    """Record that a rule was applied and whether it was successful."""
    rule = get_learning_rule_by_id(session, rule_id)
    if rule:
        rule.record_application(successful)
        session.flush()
    return rule


def deactivate_low_performing_rules(
    session: Session,
    max_success_rate: float = 0.3,
    min_applications: int = 10,
) -> int:
    """Deactivate rules with low success rates after sufficient applications."""
    rules = session.query(LearningRule).filter(
        LearningRule.is_active == True,
        LearningRule.success_rate < max_success_rate,
        LearningRule.times_applied >= min_applications,
    ).all()

    count = 0
    for rule in rules:
        rule.is_active = False
        count += 1

    session.flush()
    return count


# ============================================================================
# CRUD Operations - Edit History
# ============================================================================

def create_edit_history(
    session: Session,
    document_id: int,
    edit_type: EditType,
    made_by: EditMadeBy,
    original_content: Optional[str] = None,
    new_content: Optional[str] = None,
    **kwargs,
) -> EditHistory:
    """Create a new edit history record."""
    edit = EditHistory(
        document_id=document_id,
        edit_type=edit_type,
        made_by=made_by,
        original_content=original_content,
        new_content=new_content,
        **kwargs,
    )
    session.add(edit)
    session.flush()
    return edit


def get_edit_history_by_id(session: Session, edit_id: int) -> Optional[EditHistory]:
    """Get an edit history record by ID."""
    return session.query(EditHistory).filter(EditHistory.id == edit_id).first()


def get_edit_history_by_document(session: Session, document_id: int) -> List[EditHistory]:
    """Get all edit history for a document."""
    return session.query(EditHistory).filter(
        EditHistory.document_id == document_id
    ).order_by(EditHistory.created_at).all()


def get_unlearned_edits(session: Session, limit: int = 100) -> List[EditHistory]:
    """Get edits that haven't been processed for learning."""
    return session.query(EditHistory).filter(
        EditHistory.learned == False,
        EditHistory.made_by == EditMadeBy.HUMAN,
    ).order_by(EditHistory.created_at).limit(limit).all()


def get_edits_by_type(session: Session, edit_type: EditType) -> List[EditHistory]:
    """Get all edits of a specific type."""
    return session.query(EditHistory).filter(
        EditHistory.edit_type == edit_type
    ).order_by(EditHistory.created_at.desc()).all()


def get_human_edits(session: Session, limit: int = 100) -> List[EditHistory]:
    """Get recent human edits for learning analysis."""
    return session.query(EditHistory).filter(
        EditHistory.made_by == EditMadeBy.HUMAN
    ).order_by(EditHistory.created_at.desc()).limit(limit).all()


def mark_edit_as_learned(
    session: Session,
    edit_id: int,
    learning_rule_id: Optional[int] = None,
    learning_notes: Optional[str] = None,
) -> Optional[EditHistory]:
    """Mark an edit as processed for learning."""
    edit = get_edit_history_by_id(session, edit_id)
    if edit:
        edit.learned = True
        edit.learning_rule_id = learning_rule_id
        edit.learning_notes = learning_notes
        session.flush()
    return edit


# ============================================================================
# Query Helpers for Learning System
# ============================================================================

def get_edit_patterns(
    session: Session,
    min_occurrences: int = 3,
) -> List[Dict[str, Any]]:
    """
    Analyze edit history to find common patterns.

    Returns patterns that occur multiple times, indicating potential learning opportunities.
    """
    from sqlalchemy import func

    # Group by edit type, section, and content similarity
    patterns = session.query(
        EditHistory.edit_type,
        EditHistory.location_section,
        func.count(EditHistory.id).label("count"),
    ).filter(
        EditHistory.made_by == EditMadeBy.HUMAN,
    ).group_by(
        EditHistory.edit_type,
        EditHistory.location_section,
    ).having(
        func.count(EditHistory.id) >= min_occurrences
    ).all()

    return [
        {
            "edit_type": p[0].value if p[0] else None,
            "section": p[1],
            "occurrences": p[2],
        }
        for p in patterns
    ]


def get_template_usage_stats(session: Session) -> List[Dict[str, Any]]:
    """Get usage statistics for templates."""
    from sqlalchemy import func

    stats = session.query(
        Template.id,
        Template.name,
        Template.category,
        func.count(GeneratedDocument.id).label("doc_count"),
    ).outerjoin(
        GeneratedDocument, Template.id == GeneratedDocument.template_id
    ).group_by(
        Template.id,
        Template.name,
        Template.category,
    ).all()

    return [
        {
            "template_id": s[0],
            "template_name": s[1],
            "category": s[2].value if s[2] else None,
            "documents_generated": s[3],
        }
        for s in stats
    ]


def get_rule_effectiveness_summary(session: Session) -> Dict[str, Any]:
    """Get a summary of learning rule effectiveness."""
    from sqlalchemy import func

    total_rules = session.query(func.count(LearningRule.id)).scalar()
    active_rules = session.query(func.count(LearningRule.id)).filter(
        LearningRule.is_active == True
    ).scalar()

    avg_success = session.query(func.avg(LearningRule.success_rate)).filter(
        LearningRule.is_active == True,
        LearningRule.times_applied > 0,
    ).scalar()

    total_applications = session.query(func.sum(LearningRule.times_applied)).scalar()

    by_type = session.query(
        LearningRule.rule_type,
        func.count(LearningRule.id),
        func.avg(LearningRule.success_rate),
    ).filter(
        LearningRule.is_active == True,
    ).group_by(
        LearningRule.rule_type
    ).all()

    return {
        "total_rules": total_rules or 0,
        "active_rules": active_rules or 0,
        "average_success_rate": float(avg_success or 0),
        "total_applications": total_applications or 0,
        "by_type": {
            t[0].value: {"count": t[1], "avg_success_rate": float(t[2] or 0)}
            for t in by_type
        },
    }


def find_similar_edits(
    session: Session,
    edit_type: EditType,
    section: Optional[str] = None,
    limit: int = 10,
) -> List[EditHistory]:
    """Find similar edits for pattern analysis."""
    query = session.query(EditHistory).filter(
        EditHistory.edit_type == edit_type,
        EditHistory.made_by == EditMadeBy.HUMAN,
    )

    if section:
        query = query.filter(EditHistory.location_section == section)

    return query.order_by(EditHistory.created_at.desc()).limit(limit).all()


def get_customer_document_summary(
    session: Session,
    customer_id: int,
) -> Dict[str, Any]:
    """Get a summary of documents for a customer."""
    from sqlalchemy import func

    customer = get_customer_by_id(session, customer_id)
    if not customer:
        return {}

    doc_stats = session.query(
        GeneratedDocument.status,
        func.count(GeneratedDocument.id),
    ).filter(
        GeneratedDocument.customer_id == customer_id
    ).group_by(
        GeneratedDocument.status
    ).all()

    return {
        "customer_id": customer_id,
        "case_number": customer.case_number,
        "petitioner_name": customer.petitioner_name,
        "case_status": customer.status.value,
        "documents": {
            s[0].value: s[1] for s in doc_stats
        },
        "total_documents": sum(s[1] for s in doc_stats),
    }


# ============================================================================
# Migration Support
# ============================================================================

def get_schema_version(session: Session) -> Optional[str]:
    """Get the current schema version from the database."""
    from sqlalchemy import text

    try:
        result = session.execute(
            text("SELECT value FROM _schema_version ORDER BY applied_at DESC LIMIT 1")
        ).fetchone()
        return result[0] if result else None
    except Exception:
        return None


def set_schema_version(session: Session, version: str) -> None:
    """Set the schema version in the database."""
    from sqlalchemy import text

    # Create version table if not exists
    session.execute(text("""
        CREATE TABLE IF NOT EXISTS _schema_version (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            version VARCHAR(50) NOT NULL,
            applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """))

    session.execute(
        text("INSERT INTO _schema_version (version) VALUES (:version)"),
        {"version": version}
    )
    session.commit()


def run_migrations(session: Session, target_version: str = "1.0.0") -> List[str]:
    """
    Run database migrations to bring schema up to date.

    Returns list of migrations applied.
    """
    current_version = get_schema_version(session) or "0.0.0"
    applied = []

    migrations = {
        "1.0.0": [
            # Initial schema - tables created by Base.metadata.create_all()
        ],
        # Add future migrations here
        # "1.0.1": [
        #     "ALTER TABLE templates ADD COLUMN new_field VARCHAR(100)",
        # ],
    }

    from packaging.version import Version

    for version, statements in sorted(migrations.items(), key=lambda x: Version(x[0])):
        if Version(version) > Version(current_version) and Version(version) <= Version(target_version):
            for statement in statements:
                from sqlalchemy import text
                session.execute(text(statement))
            set_schema_version(session, version)
            applied.append(version)

    return applied


# ============================================================================
# Utility Functions
# ============================================================================

def export_data(session: Session, include_content: bool = False) -> Dict[str, Any]:
    """Export all database data to a dictionary."""
    data = {
        "templates": [t.to_dict() for t in get_all_templates(session, active_only=False)],
        "customers": [c.to_dict() for c in get_all_customers(session, limit=10000)],
        "documents": [d.to_dict() for d in session.query(GeneratedDocument).all()],
        "learning_rules": [r.to_dict() for r in session.query(LearningRule).all()],
        "edit_history": [e.to_dict() for e in session.query(EditHistory).all()],
        "exported_at": datetime.utcnow().isoformat(),
    }

    return data


def get_database_stats(session: Session) -> Dict[str, int]:
    """Get counts of records in each table."""
    from sqlalchemy import func

    return {
        "templates": session.query(func.count(Template.id)).scalar() or 0,
        "customers": session.query(func.count(Customer.id)).scalar() or 0,
        "generated_documents": session.query(func.count(GeneratedDocument.id)).scalar() or 0,
        "learning_rules": session.query(func.count(LearningRule.id)).scalar() or 0,
        "edit_history": session.query(func.count(EditHistory.id)).scalar() or 0,
    }

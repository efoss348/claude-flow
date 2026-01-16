"""
Learning and Self-Improvement System for Document Automation.

Provides adaptive learning from human edits, pattern recognition,
knowledge base management, and continuous self-improvement capabilities.
"""

from __future__ import annotations

import json
import subprocess
import hashlib
import re
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Tuple, Callable
from collections import defaultdict
from difflib import SequenceMatcher
import uuid

from ..models.learning_models import (
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


class MemoryClient:
    """Client for interacting with claude-flow memory system."""

    def __init__(self, cli_path: str = "npx"):
        self.cli_path = cli_path
        self.cli_package = "@claude-flow/cli@latest"

    def store(
        self,
        key: str,
        value: Any,
        namespace: str = "learning",
        tags: Optional[List[str]] = None,
    ) -> bool:
        """Store a value in the memory system."""
        try:
            value_str = json.dumps(value) if not isinstance(value, str) else value
            cmd = [
                self.cli_path,
                self.cli_package,
                "memory",
                "store",
                "--key",
                key,
                "--value",
                value_str,
                "--namespace",
                namespace,
            ]
            if tags:
                cmd.extend(["--tags", ",".join(tags)])

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=30,
            )
            return result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return False

    def search(
        self,
        query: str,
        namespace: str = "learning",
        limit: int = 5,
    ) -> List[Dict[str, Any]]:
        """Search the memory system using semantic search."""
        try:
            cmd = [
                self.cli_path,
                self.cli_package,
                "memory",
                "search",
                "--query",
                query,
                "--namespace",
                namespace,
                "--limit",
                str(limit),
            ]
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=30,
            )
            if result.returncode == 0 and result.stdout:
                try:
                    return json.loads(result.stdout)
                except json.JSONDecodeError:
                    return []
            return []
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return []

    def retrieve(self, key: str, namespace: str = "learning") -> Optional[Any]:
        """Retrieve a specific value from memory."""
        try:
            cmd = [
                self.cli_path,
                self.cli_package,
                "memory",
                "retrieve",
                "--key",
                key,
                "--namespace",
                namespace,
            ]
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=30,
            )
            if result.returncode == 0 and result.stdout:
                try:
                    return json.loads(result.stdout)
                except json.JSONDecodeError:
                    return result.stdout.strip()
            return None
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return None


class LearningCoordinator:
    """
    Coordinates the learning system, tracking edits, analyzing patterns,
    creating rules, and managing the knowledge base.
    """

    def __init__(
        self,
        knowledge_base: Optional[KnowledgeBase] = None,
        memory_client: Optional[MemoryClient] = None,
    ):
        self.knowledge_base = knowledge_base or KnowledgeBase()
        self.memory_client = memory_client or MemoryClient()

        # In-memory storage (would be database in production)
        self.edit_records: Dict[str, EditRecord] = {}
        self.patterns: Dict[str, Pattern] = {}
        self.rules: Dict[str, Rule] = {}

        # Pattern detection thresholds
        self.min_pattern_occurrences = 3
        self.min_confidence_threshold = 0.7

    def record_edit(
        self,
        document_id: str,
        edit_details: Dict[str, Any],
    ) -> EditRecord:
        """
        Track a human edit made to a document.

        Args:
            document_id: ID of the document being edited
            edit_details: Details of the edit including original/edited text

        Returns:
            EditRecord capturing the edit information
        """
        record = EditRecord(
            document_id=document_id,
            user_id=edit_details.get("user_id"),
            original_text=edit_details.get("original_text", ""),
            edited_text=edit_details.get("edited_text", ""),
            edit_type=self._classify_edit_type(
                edit_details.get("original_text", ""),
                edit_details.get("edited_text", ""),
            ),
            section=edit_details.get("section"),
            context=edit_details.get("context", {}),
            metadata=edit_details.get("metadata", {}),
        )

        self.edit_records[record.id] = record

        # Store in memory system for persistence
        self.memory_client.store(
            key=f"edit/{record.id}",
            value=asdict(record),
            namespace="edits",
            tags=[document_id, str(record.edit_type.value)],
        )

        # Trigger pattern analysis after recording
        self._update_pattern_candidates(record)

        return record

    def _classify_edit_type(self, original: str, edited: str) -> FeedbackType:
        """Classify the type of edit based on original and edited text."""
        if not original and edited:
            return FeedbackType.ADDITION
        if original and not edited:
            return FeedbackType.DELETION
        if len(edited) > len(original) * 1.5:
            return FeedbackType.ADDITION
        if len(edited) < len(original) * 0.5:
            return FeedbackType.DELETION

        similarity = SequenceMatcher(None, original, edited).ratio()
        if similarity < 0.5:
            return FeedbackType.RESTRUCTURE
        return FeedbackType.CORRECTION

    def _update_pattern_candidates(self, record: EditRecord) -> None:
        """Update pattern candidates based on a new edit record."""
        # Generate a pattern signature from the edit
        signature = self._generate_pattern_signature(record)

        if signature in self.patterns:
            pattern = self.patterns[signature]
            pattern.occurrence_count += 1
            pattern.last_seen = datetime.now()
            pattern.examples.append({
                "original": record.original_text[:200],
                "edited": record.edited_text[:200],
                "document_id": record.document_id,
            })
            # Keep only last 10 examples
            pattern.examples = pattern.examples[-10:]
        else:
            # Create new pattern candidate
            pattern = Pattern(
                id=signature,
                name=f"Pattern-{signature[:8]}",
                description=f"Auto-detected from {record.edit_type.value}",
                category=self._infer_category(record),
                trigger_conditions=[self._extract_trigger(record)],
                occurrence_count=1,
                status=PatternStatus.DETECTED,
                examples=[{
                    "original": record.original_text[:200],
                    "edited": record.edited_text[:200],
                    "document_id": record.document_id,
                }],
            )
            self.patterns[signature] = pattern

    def _generate_pattern_signature(self, record: EditRecord) -> str:
        """Generate a unique signature for a pattern based on edit characteristics."""
        components = [
            record.edit_type.value,
            record.section or "unknown",
            str(len(record.original_text) // 50),  # Length bucket
            self._extract_key_terms(record.original_text),
        ]
        signature_str = "|".join(components)
        return hashlib.md5(signature_str.encode()).hexdigest()[:16]

    def _extract_key_terms(self, text: str) -> str:
        """Extract key terms from text for pattern matching."""
        words = re.findall(r'\b\w{4,}\b', text.lower())
        word_freq = defaultdict(int)
        for word in words:
            word_freq[word] += 1
        top_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:3]
        return ",".join(w[0] for w in top_words)

    def _infer_category(self, record: EditRecord) -> RuleCategory:
        """Infer the rule category from an edit record."""
        text = record.edited_text.lower()

        if any(term in text for term in ["§", "section", "pursuant", "hereby"]):
            return RuleCategory.LEGAL_CLAUSE
        if record.edit_type == FeedbackType.RESTRUCTURE:
            return RuleCategory.STRUCTURE
        if any(term in text for term in ["format", "indent", "spacing"]):
            return RuleCategory.FORMATTING
        if record.section and "style" in record.section.lower():
            return RuleCategory.STYLE
        return RuleCategory.CONTENT

    def _extract_trigger(self, record: EditRecord) -> str:
        """Extract the trigger condition from an edit."""
        if record.section:
            return f"section:{record.section}"
        return f"content_type:{record.edit_type.value}"

    def analyze_patterns(self) -> List[Pattern]:
        """
        Find recurring correction patterns from recorded edits.

        Returns:
            List of validated patterns ready for rule generation
        """
        validated_patterns = []

        for signature, pattern in self.patterns.items():
            if pattern.occurrence_count >= self.min_pattern_occurrences:
                # Calculate confidence based on consistency
                confidence = self._calculate_pattern_confidence(pattern)
                pattern.confidence_score = confidence

                if confidence >= self.min_confidence_threshold:
                    pattern.status = PatternStatus.VALIDATED
                    validated_patterns.append(pattern)

                    # Store validated pattern in memory
                    self.memory_client.store(
                        key=f"pattern/{pattern.id}",
                        value=asdict(pattern),
                        namespace="patterns",
                        tags=[pattern.category.value, "validated"],
                    )

        return validated_patterns

    def _calculate_pattern_confidence(self, pattern: Pattern) -> float:
        """Calculate confidence score for a pattern."""
        base_confidence = min(pattern.occurrence_count / 10, 1.0)

        # Boost confidence if examples are similar
        if len(pattern.examples) >= 2:
            similarities = []
            for i, ex1 in enumerate(pattern.examples[:-1]):
                for ex2 in pattern.examples[i + 1:]:
                    sim = SequenceMatcher(
                        None, ex1["edited"], ex2["edited"]
                    ).ratio()
                    similarities.append(sim)
            avg_similarity = sum(similarities) / len(similarities) if similarities else 0
            base_confidence *= (0.5 + 0.5 * avg_similarity)

        return round(base_confidence, 3)

    def create_rule(
        self,
        pattern: Pattern,
        action: str,
    ) -> Rule:
        """
        Generate a new rule from an identified pattern.

        Args:
            pattern: The validated pattern to convert to a rule
            action: The action to take when the rule matches

        Returns:
            A new Rule ready for activation
        """
        rule = Rule(
            name=f"Rule-{pattern.name}",
            description=f"Auto-generated from pattern: {pattern.description}",
            category=pattern.category,
            condition=" AND ".join(pattern.trigger_conditions),
            action=action,
            priority=min(5 + pattern.occurrence_count // 2, 10),
            status=RuleStatus.TESTING,
            source_pattern_id=pattern.id,
            metadata={
                "confidence": pattern.confidence_score,
                "examples_count": len(pattern.examples),
            },
        )

        self.rules[rule.id] = rule

        # Store rule in knowledge base
        self.knowledge_base.add_rule(rule)

        # Store in memory system
        self.memory_client.store(
            key=f"rule/{rule.id}",
            value=asdict(rule),
            namespace="rules",
            tags=[rule.category.value, rule.status.value],
        )

        return rule

    def query_knowledge(self, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Search past learnings based on current context.

        Args:
            context: Current document/situation context

        Returns:
            List of relevant knowledge entries
        """
        query = self._build_query_from_context(context)

        # Search local knowledge base
        local_results = self.knowledge_base.search(query)

        # Search memory system for additional patterns
        memory_results = self.memory_client.search(
            query=query,
            namespace="patterns",
            limit=5,
        )

        # Merge and deduplicate results
        all_results = local_results + memory_results
        seen_ids = set()
        unique_results = []
        for result in all_results:
            result_id = result.get("id", str(hash(str(result))))
            if result_id not in seen_ids:
                seen_ids.add(result_id)
                unique_results.append(result)

        return unique_results

    def _build_query_from_context(self, context: Dict[str, Any]) -> str:
        """Build a search query from context."""
        components = []
        if "document_type" in context:
            components.append(context["document_type"])
        if "section" in context:
            components.append(context["section"])
        if "keywords" in context:
            components.extend(context["keywords"])
        if "text_sample" in context:
            key_terms = self._extract_key_terms(context["text_sample"])
            components.append(key_terms)

        return " ".join(components)

    def suggest_improvements(
        self,
        document: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """
        Generate proactive improvement suggestions for a document.

        Args:
            document: Document data to analyze

        Returns:
            List of suggested improvements with confidence scores
        """
        suggestions = []

        # Get applicable rules
        context = {
            "document_type": document.get("type"),
            "sections": list(document.get("sections", {}).keys()),
        }
        applicable_rules = self.knowledge_base.get_applicable_rules(context)

        for rule in applicable_rules:
            # Check if rule condition matches document
            if self._rule_matches_document(rule, document):
                suggestion = {
                    "rule_id": rule.id,
                    "type": rule.category.value,
                    "description": rule.description,
                    "suggested_action": rule.action,
                    "confidence": rule.success_rate,
                    "priority": rule.priority,
                }
                suggestions.append(suggestion)

        # Also query knowledge base for scenario-specific suggestions
        doc_text = document.get("content", "")
        if doc_text:
            kb_suggestions = self.knowledge_base.search(doc_text[:500], top_k=3)
            for kb_entry in kb_suggestions:
                if kb_entry.get("entry_type") == "scenario":
                    suggestions.append({
                        "rule_id": kb_entry.get("id"),
                        "type": "scenario",
                        "description": kb_entry.get("trigger", ""),
                        "suggested_action": kb_entry.get("response", ""),
                        "confidence": kb_entry.get("effectiveness_score", 0.5),
                        "priority": 5,
                    })

        # Sort by priority and confidence
        suggestions.sort(key=lambda x: (x["priority"], x["confidence"]), reverse=True)
        return suggestions

    def _rule_matches_document(self, rule: Rule, document: Dict[str, Any]) -> bool:
        """Check if a rule's condition matches a document."""
        condition = rule.condition.lower()
        doc_text = str(document).lower()

        # Simple condition matching - can be made more sophisticated
        for part in condition.split(" AND "):
            part = part.strip()
            if part.startswith("section:"):
                section = part.split(":", 1)[1]
                if section not in document.get("sections", {}):
                    return False
            elif part.startswith("content_type:"):
                content_type = part.split(":", 1)[1]
                if content_type not in doc_text:
                    return False
            elif part not in doc_text:
                return False
        return True

    def update_success_metrics(self, rule_id: str, success: bool) -> None:
        """
        Track rule effectiveness after application.

        Args:
            rule_id: ID of the rule that was applied
            success: Whether the rule application was successful
        """
        if rule_id in self.rules:
            rule = self.rules[rule_id]
            rule.update_success_metrics(success)

            # Activate rule if it performs well during testing
            if (
                rule.status == RuleStatus.TESTING
                and rule.success_count >= 5
                and rule.success_rate >= 0.8
            ):
                rule.status = RuleStatus.ACTIVE

            # Disable poorly performing rules
            if rule.failure_count >= 5 and rule.success_rate < 0.3:
                rule.status = RuleStatus.DISABLED

            # Update in memory
            self.memory_client.store(
                key=f"rule/{rule_id}",
                value=asdict(rule),
                namespace="rules",
                tags=[rule.category.value, rule.status.value],
            )

        # Also update in knowledge base
        self.knowledge_base.update_rule_metrics(rule_id, success)


class KnowledgeBase:
    """
    Stores and retrieves learned knowledge including scenarios,
    formatting rules, and legal clause mappings.
    """

    def __init__(self, memory_client: Optional[MemoryClient] = None):
        self.memory_client = memory_client or MemoryClient()

        # In-memory storage (would be database in production)
        self.entries: Dict[str, KnowledgeEntry] = {}
        self.rules: Dict[str, Rule] = {}
        self.scenarios: Dict[str, ScenarioRule] = {}

    def add_scenario(self, trigger: str, response: str) -> KnowledgeEntry:
        """
        Store a scenario-based rule.

        Args:
            trigger: The condition that triggers this scenario
            response: The response/action for this scenario

        Returns:
            The created KnowledgeEntry
        """
        entry = KnowledgeEntry(
            entry_type="scenario",
            trigger=trigger,
            response=response,
            context_tags=self._extract_tags(trigger),
        )
        self.entries[entry.id] = entry

        self.memory_client.store(
            key=f"knowledge/scenario/{entry.id}",
            value=asdict(entry),
            namespace="knowledge",
            tags=["scenario"] + entry.context_tags,
        )

        return entry

    def add_formatting_rule(
        self,
        condition: str,
        fix: str,
    ) -> KnowledgeEntry:
        """
        Store a formatting fix rule.

        Args:
            condition: When this formatting rule applies
            fix: The formatting correction to apply

        Returns:
            The created KnowledgeEntry
        """
        entry = KnowledgeEntry(
            entry_type="formatting",
            trigger=condition,
            response=fix,
            context_tags=["formatting"] + self._extract_tags(condition),
        )
        self.entries[entry.id] = entry

        self.memory_client.store(
            key=f"knowledge/formatting/{entry.id}",
            value=asdict(entry),
            namespace="knowledge",
            tags=["formatting"],
        )

        return entry

    def add_legal_clause(
        self,
        situation: str,
        clause: str,
    ) -> KnowledgeEntry:
        """
        Store a legal clause mapping.

        Args:
            situation: The legal situation this clause addresses
            clause: The legal clause text to apply

        Returns:
            The created KnowledgeEntry
        """
        entry = KnowledgeEntry(
            entry_type="legal_clause",
            trigger=situation,
            response=clause,
            context_tags=["legal"] + self._extract_tags(situation),
        )
        self.entries[entry.id] = entry

        self.memory_client.store(
            key=f"knowledge/legal/{entry.id}",
            value=asdict(entry),
            namespace="knowledge",
            tags=["legal_clause"],
        )

        return entry

    def _extract_tags(self, text: str) -> List[str]:
        """Extract relevant tags from text."""
        tags = []
        text_lower = text.lower()

        # Legal terms
        legal_terms = ["custody", "asset", "support", "divorce", "property", "visitation"]
        for term in legal_terms:
            if term in text_lower:
                tags.append(term)

        # Document types
        doc_types = ["agreement", "motion", "petition", "order", "decree"]
        for doc_type in doc_types:
            if doc_type in text_lower:
                tags.append(doc_type)

        return tags

    def search(
        self,
        query: str,
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Semantic search of knowledge base.

        Args:
            query: Search query
            top_k: Number of results to return

        Returns:
            List of matching knowledge entries
        """
        results = []

        # Local search using similarity
        query_lower = query.lower()
        query_terms = set(re.findall(r'\b\w+\b', query_lower))

        for entry in self.entries.values():
            score = self._calculate_relevance_score(entry, query_terms)
            if score > 0.1:
                entry_dict = asdict(entry)
                entry_dict["relevance_score"] = score
                results.append(entry_dict)

        # Also search memory system
        memory_results = self.memory_client.search(
            query=query,
            namespace="knowledge",
            limit=top_k,
        )

        for mem_result in memory_results:
            if isinstance(mem_result, dict):
                mem_result["relevance_score"] = mem_result.get("score", 0.5)
                results.append(mem_result)

        # Sort by relevance and return top_k
        results.sort(key=lambda x: x.get("relevance_score", 0), reverse=True)
        return results[:top_k]

    def _calculate_relevance_score(
        self,
        entry: KnowledgeEntry,
        query_terms: set,
    ) -> float:
        """Calculate relevance score between entry and query terms."""
        entry_text = f"{entry.trigger} {entry.response}".lower()
        entry_terms = set(re.findall(r'\b\w+\b', entry_text))

        if not query_terms:
            return 0.0

        intersection = query_terms & entry_terms
        union = query_terms | entry_terms

        # Jaccard similarity with tag boost
        jaccard = len(intersection) / len(union) if union else 0

        # Boost if tags match
        tag_boost = sum(1 for tag in entry.context_tags if tag in query_terms) * 0.1

        return min(jaccard + tag_boost, 1.0)

    def get_applicable_rules(
        self,
        context: Dict[str, Any],
    ) -> List[Rule]:
        """
        Get rules applicable to the current document context.

        Args:
            context: Current document context

        Returns:
            List of applicable rules
        """
        applicable = []

        for rule in self.rules.values():
            if rule.status != RuleStatus.ACTIVE:
                continue

            # Check if rule category matches context
            doc_type = context.get("document_type", "").lower()
            sections = [s.lower() for s in context.get("sections", [])]

            if rule.category == RuleCategory.LEGAL_CLAUSE and "legal" in doc_type:
                applicable.append(rule)
            elif rule.category == RuleCategory.FORMATTING:
                applicable.append(rule)
            elif rule.category == RuleCategory.SCENARIO:
                # Check trigger conditions against sections
                for trigger in rule.condition.split(" AND "):
                    if any(trigger.lower() in section for section in sections):
                        applicable.append(rule)
                        break

        # Sort by priority
        applicable.sort(key=lambda r: r.priority, reverse=True)
        return applicable

    def add_rule(self, rule: Rule) -> None:
        """Add a rule to the knowledge base."""
        self.rules[rule.id] = rule

    def update_rule_metrics(self, rule_id: str, success: bool) -> None:
        """Update rule metrics in knowledge base."""
        if rule_id in self.rules:
            self.rules[rule_id].update_success_metrics(success)


class FeedbackProcessor:
    """
    Processes human feedback on documents, extracting actionable rules
    and patterns for system improvement.
    """

    def __init__(
        self,
        knowledge_base: Optional[KnowledgeBase] = None,
        learning_coordinator: Optional[LearningCoordinator] = None,
    ):
        self.knowledge_base = knowledge_base or KnowledgeBase()
        self.learning_coordinator = learning_coordinator

        # Feedback storage
        self.feedback_records: Dict[str, Feedback] = {}

        # Category keywords for classification
        self.category_keywords = {
            FeedbackType.CORRECTION: ["wrong", "incorrect", "error", "fix", "should be"],
            FeedbackType.ADDITION: ["add", "include", "missing", "need", "should have"],
            FeedbackType.DELETION: ["remove", "delete", "unnecessary", "too much"],
            FeedbackType.RESTRUCTURE: ["reorganize", "move", "reorder", "restructure"],
            FeedbackType.STYLE_CHANGE: ["style", "tone", "formal", "informal"],
            FeedbackType.CLARIFICATION: ["unclear", "confusing", "clarify", "explain"],
        }

    def process_human_feedback(
        self,
        feedback: Dict[str, Any],
    ) -> Feedback:
        """
        Parse and process natural language feedback.

        Args:
            feedback: Feedback data including content and context

        Returns:
            Processed Feedback object
        """
        feedback_obj = Feedback(
            document_id=feedback.get("document_id"),
            rule_id=feedback.get("rule_id"),
            user_id=feedback.get("user_id"),
            original_content=feedback.get("original_content", ""),
            feedback_content=feedback.get("feedback_content", ""),
            natural_language_notes=feedback.get("notes"),
        )

        # Classify the feedback
        feedback_obj.feedback_type = self.categorize_feedback(feedback_obj)

        # Extract potential rules
        extracted_rules = self.extract_rule(feedback_obj)
        feedback_obj.extracted_rules = [r.id for r in extracted_rules]
        feedback_obj.is_processed = True

        self.feedback_records[feedback_obj.id] = feedback_obj
        return feedback_obj

    def categorize_feedback(self, feedback: Feedback) -> FeedbackType:
        """
        Classify the type of correction based on feedback content.

        Args:
            feedback: Feedback object to categorize

        Returns:
            The determined FeedbackType
        """
        text = f"{feedback.feedback_content} {feedback.natural_language_notes or ''}".lower()

        scores = defaultdict(float)
        for category, keywords in self.category_keywords.items():
            for keyword in keywords:
                if keyword in text:
                    scores[category] += 1

        if scores:
            return max(scores.keys(), key=lambda k: scores[k])

        # Default categorization based on content comparison
        if not feedback.original_content:
            return FeedbackType.ADDITION
        if not feedback.feedback_content:
            return FeedbackType.DELETION

        similarity = SequenceMatcher(
            None,
            feedback.original_content,
            feedback.feedback_content,
        ).ratio()

        if similarity < 0.3:
            return FeedbackType.RESTRUCTURE
        return FeedbackType.CORRECTION

    def extract_rule(self, feedback: Feedback) -> List[Rule]:
        """
        Convert feedback into actionable rules.

        Args:
            feedback: Processed feedback object

        Returns:
            List of extracted rules
        """
        rules = []

        notes = feedback.natural_language_notes or ""
        content_change = feedback.feedback_content

        # Extract explicit rules from notes
        rule_patterns = [
            r"always (.+)",
            r"never (.+)",
            r"should (.+)",
            r"must (.+)",
            r"when (.+), (.+)",
            r"if (.+), then (.+)",
        ]

        for pattern in rule_patterns:
            matches = re.findall(pattern, notes.lower())
            for match in matches:
                if isinstance(match, tuple):
                    condition, action = match
                else:
                    condition = ""
                    action = match

                rule = Rule(
                    name=f"Feedback-Rule-{feedback.id[:8]}",
                    description=f"Extracted from feedback: {notes[:50]}",
                    category=self._infer_rule_category(feedback, action),
                    condition=condition or feedback.feedback_type.value,
                    action=action,
                    status=RuleStatus.TESTING,
                    metadata={
                        "source_feedback_id": feedback.id,
                        "extraction_method": "pattern_matching",
                    },
                )
                rules.append(rule)

        # Create implicit rule from content change
        if feedback.original_content and content_change:
            implicit_rule = Rule(
                name=f"Implicit-Rule-{feedback.id[:8]}",
                description=f"Implicit from correction: {feedback.feedback_type.value}",
                category=self._infer_rule_category(feedback, content_change),
                condition=f"similar_to:{self._get_content_signature(feedback.original_content)}",
                action=f"replace_with:{content_change[:200]}",
                status=RuleStatus.TESTING,
                metadata={
                    "source_feedback_id": feedback.id,
                    "extraction_method": "implicit",
                },
            )
            rules.append(implicit_rule)

        # Add rules to knowledge base
        for rule in rules:
            self.knowledge_base.add_rule(rule)

        return rules

    def _infer_rule_category(
        self,
        feedback: Feedback,
        action: str,
    ) -> RuleCategory:
        """Infer the rule category from feedback and action."""
        action_lower = action.lower()
        feedback_text = f"{feedback.feedback_content} {feedback.natural_language_notes or ''}".lower()

        if any(term in action_lower for term in ["format", "indent", "space", "margin"]):
            return RuleCategory.FORMATTING
        if any(term in feedback_text for term in ["clause", "legal", "section"]):
            return RuleCategory.LEGAL_CLAUSE
        if feedback.feedback_type == FeedbackType.RESTRUCTURE:
            return RuleCategory.STRUCTURE
        if feedback.feedback_type == FeedbackType.STYLE_CHANGE:
            return RuleCategory.STYLE

        return RuleCategory.CONTENT

    def _get_content_signature(self, content: str) -> str:
        """Generate a signature for content matching."""
        words = re.findall(r'\b\w+\b', content.lower())[:10]
        return "_".join(words)

    def apply_to_future(self, rule: Rule) -> bool:
        """
        Mark a rule for future application.

        Args:
            rule: Rule to mark for future use

        Returns:
            Whether the operation succeeded
        """
        rule.status = RuleStatus.ACTIVE
        rule.metadata["apply_to_future"] = True
        rule.updated_at = datetime.now()

        # Store in knowledge base
        self.knowledge_base.add_rule(rule)

        return True


class SelfReflection:
    """
    Analyzes system performance, identifies improvement areas,
    and generates training data for continuous improvement.
    """

    def __init__(
        self,
        learning_coordinator: Optional[LearningCoordinator] = None,
        memory_client: Optional[MemoryClient] = None,
    ):
        self.learning_coordinator = learning_coordinator or LearningCoordinator()
        self.memory_client = memory_client or MemoryClient()

        # Review storage
        self.reviews: Dict[str, DocumentReview] = {}
        self.accuracy_metrics: List[AccuracyMetric] = []
        self.training_data: List[TrainingData] = []

    def post_document_review(self, doc_id: str) -> DocumentReview:
        """
        Analyze what corrections were made to a document.

        Args:
            doc_id: ID of the document to review

        Returns:
            DocumentReview analysis
        """
        review = DocumentReview(document_id=doc_id)

        # Collect all edits for this document
        doc_edits = [
            edit for edit in self.learning_coordinator.edit_records.values()
            if edit.document_id == doc_id
        ]

        review.total_corrections = len(doc_edits)

        # Categorize corrections
        for edit in doc_edits:
            category = edit.edit_type.value
            review.correction_categories[category] = (
                review.correction_categories.get(category, 0) + 1
            )

        # Analyze rules that were applied
        for rule_id, rule in self.learning_coordinator.rules.items():
            if rule.status == RuleStatus.ACTIVE:
                if rule.success_count > 0:
                    review.rules_applied.append(rule_id)
                if rule.failure_count > 0:
                    review.rules_failed.append(rule_id)

        # Calculate quality score
        if doc_edits:
            avg_correction_rate = review.total_corrections / max(1, len(doc_edits))
            review.overall_quality_score = max(0, 1 - avg_correction_rate * 0.1)
        else:
            review.overall_quality_score = 1.0

        # Identify gaps
        review.gaps_identified = self._identify_document_gaps(doc_edits)

        self.reviews[review.id] = review

        # Store review in memory
        self.memory_client.store(
            key=f"review/{review.id}",
            value=asdict(review),
            namespace="reviews",
            tags=[doc_id],
        )

        return review

    def _identify_document_gaps(
        self,
        edits: List[EditRecord],
    ) -> List[str]:
        """Identify gaps based on edit patterns."""
        gaps = []

        # Group edits by type
        type_counts = defaultdict(int)
        for edit in edits:
            type_counts[edit.edit_type.value] += 1

        # Identify areas with many corrections
        for edit_type, count in type_counts.items():
            if count >= 3:
                gaps.append(f"High {edit_type} rate: {count} corrections")

        # Check for repeated patterns
        section_corrections = defaultdict(int)
        for edit in edits:
            if edit.section:
                section_corrections[edit.section] += 1

        for section, count in section_corrections.items():
            if count >= 2:
                gaps.append(f"Section '{section}' needs attention: {count} corrections")

        return gaps

    def identify_gaps(self) -> List[Dict[str, Any]]:
        """
        Find areas needing improvement across all reviews.

        Returns:
            List of identified improvement areas
        """
        gaps = []

        # Aggregate metrics from all reviews
        all_categories = defaultdict(int)
        all_failed_rules = defaultdict(int)

        for review in self.reviews.values():
            for cat, count in review.correction_categories.items():
                all_categories[cat] += count
            for rule_id in review.rules_failed:
                all_failed_rules[rule_id] += 1

        # Identify top problem areas
        for category, count in sorted(
            all_categories.items(), key=lambda x: x[1], reverse=True
        ):
            if count >= 5:
                gaps.append({
                    "type": "category",
                    "name": category,
                    "severity": count,
                    "recommendation": f"Improve {category} handling - {count} corrections needed",
                })

        # Identify failing rules
        for rule_id, fail_count in all_failed_rules.items():
            if fail_count >= 3:
                rule = self.learning_coordinator.rules.get(rule_id)
                gaps.append({
                    "type": "rule",
                    "name": rule.name if rule else rule_id,
                    "severity": fail_count,
                    "recommendation": f"Rule failing frequently - consider revision",
                })

        # Check pattern coverage
        unaddressed_patterns = [
            p for p in self.learning_coordinator.patterns.values()
            if p.status == PatternStatus.DETECTED and p.occurrence_count >= 3
        ]
        for pattern in unaddressed_patterns:
            gaps.append({
                "type": "pattern",
                "name": pattern.name,
                "severity": pattern.occurrence_count,
                "recommendation": f"Unaddressed pattern with {pattern.occurrence_count} occurrences",
            })

        return gaps

    def generate_training_data(self) -> List[TrainingData]:
        """
        Create training data for model improvement.

        Returns:
            List of generated training data entries
        """
        training_entries = []

        # Generate from edit records
        for edit in self.learning_coordinator.edit_records.values():
            if edit.original_text and edit.edited_text:
                entry = TrainingData(
                    input_text=edit.original_text,
                    expected_output=edit.edited_text,
                    context={
                        "document_id": edit.document_id,
                        "section": edit.section,
                        "edit_type": edit.edit_type.value,
                    },
                    category=edit.edit_type.value,
                    quality_score=1.0,  # Human-validated
                )
                training_entries.append(entry)

        # Generate from validated patterns
        for pattern in self.learning_coordinator.patterns.values():
            if pattern.status == PatternStatus.VALIDATED:
                for example in pattern.examples:
                    entry = TrainingData(
                        input_text=example.get("original", ""),
                        expected_output=example.get("edited", ""),
                        context={
                            "pattern_id": pattern.id,
                            "category": pattern.category.value,
                        },
                        category=pattern.category.value,
                        quality_score=pattern.confidence_score,
                    )
                    training_entries.append(entry)

        self.training_data.extend(training_entries)

        # Store in memory for external access
        self.memory_client.store(
            key=f"training_data/{datetime.now().isoformat()}",
            value=[asdict(t) for t in training_entries],
            namespace="training",
            tags=["generated"],
        )

        return training_entries

    def track_accuracy_over_time(self) -> Dict[str, Any]:
        """
        Monitor system improvement over time.

        Returns:
            Accuracy metrics and trends
        """
        # Calculate current period metrics
        current_time = datetime.now()
        period_start = current_time - timedelta(days=7)

        recent_reviews = [
            r for r in self.reviews.values()
            if r.reviewed_at >= period_start
        ]

        if not recent_reviews:
            return {"message": "No recent reviews to analyze"}

        # Calculate average quality
        avg_quality = sum(r.overall_quality_score for r in recent_reviews) / len(recent_reviews)

        # Calculate correction rate
        total_corrections = sum(r.total_corrections for r in recent_reviews)
        avg_corrections = total_corrections / len(recent_reviews)

        # Calculate rule success rate
        successful_rules = sum(len(r.rules_applied) for r in recent_reviews)
        failed_rules = sum(len(r.rules_failed) for r in recent_reviews)
        total_rules = successful_rules + failed_rules
        rule_success_rate = successful_rules / total_rules if total_rules > 0 else 1.0

        current_metric = AccuracyMetric(
            metric_name="weekly_accuracy",
            value=avg_quality,
            period_start=period_start,
            period_end=current_time,
            sample_size=len(recent_reviews),
            metadata={
                "avg_corrections": avg_corrections,
                "rule_success_rate": rule_success_rate,
                "total_corrections": total_corrections,
            },
        )
        self.accuracy_metrics.append(current_metric)

        # Calculate trend
        trend = "stable"
        if len(self.accuracy_metrics) >= 2:
            prev_metric = self.accuracy_metrics[-2]
            if current_metric.value > prev_metric.value + 0.05:
                trend = "improving"
            elif current_metric.value < prev_metric.value - 0.05:
                trend = "declining"

        # Store metrics
        self.memory_client.store(
            key=f"accuracy/{current_metric.id}",
            value=asdict(current_metric),
            namespace="metrics",
            tags=["accuracy", trend],
        )

        return {
            "current_period": {
                "quality_score": round(avg_quality, 3),
                "avg_corrections_per_document": round(avg_corrections, 2),
                "rule_success_rate": round(rule_success_rate, 3),
                "documents_reviewed": len(recent_reviews),
            },
            "trend": trend,
            "historical_metrics": [
                {
                    "period": m.period_start.isoformat(),
                    "quality": m.value,
                }
                for m in self.accuracy_metrics[-5:]
            ],
        }


class ScenarioEngine:
    """
    Handles scenario-based document generation rules,
    automatically applying relevant content based on detected situations.
    """

    def __init__(
        self,
        knowledge_base: Optional[KnowledgeBase] = None,
        memory_client: Optional[MemoryClient] = None,
    ):
        self.knowledge_base = knowledge_base or KnowledgeBase()
        self.memory_client = memory_client or MemoryClient()

        # Scenario rules storage
        self.scenarios: Dict[str, ScenarioRule] = {}

        # Register default scenarios
        self._register_default_scenarios()

    def _register_default_scenarios(self) -> None:
        """Register built-in scenario rules."""
        # Contested custody scenario
        self.register_scenario(
            scenario_name="contested_custody",
            description="Custody dispute between parents",
            detection_criteria={
                "keywords": ["custody", "contested", "dispute", "visitation rights"],
                "document_types": ["divorce_decree", "custody_agreement", "parenting_plan"],
            },
            actions=[
                {
                    "action": "add_section",
                    "section": "visitation_schedule",
                    "content": self._get_visitation_schedule_template(),
                },
                {
                    "action": "add_bullets",
                    "location": "custody_terms",
                    "items": [
                        "Regular visitation schedule for non-custodial parent",
                        "Holiday and vacation rotation schedule",
                        "Transportation arrangements and responsibilities",
                        "Communication guidelines between parents",
                        "Right of first refusal for childcare",
                    ],
                },
            ],
        )

        # Complex assets scenario
        self.register_scenario(
            scenario_name="complex_assets",
            description="Division of complex or high-value assets",
            detection_criteria={
                "keywords": ["business", "investment", "retirement", "real estate portfolio", "stock options"],
                "conditions": ["asset_value > 500000", "business_ownership"],
            },
            actions=[
                {
                    "action": "add_section",
                    "section": "property_division",
                    "content": self._get_property_division_template(),
                },
                {
                    "action": "add_table",
                    "table_name": "asset_valuation",
                    "columns": ["Asset", "Description", "Valuation Date", "Fair Market Value", "Allocation"],
                },
            ],
        )

        # Spousal support scenario
        self.register_scenario(
            scenario_name="spousal_support",
            description="Determination of spousal support/alimony",
            detection_criteria={
                "keywords": ["spousal support", "alimony", "maintenance", "income disparity"],
                "conditions": ["income_difference > 30%", "marriage_duration > 5_years"],
            },
            actions=[
                {
                    "action": "add_section",
                    "section": "income_calculation",
                    "content": self._get_income_calculation_template(),
                },
                {
                    "action": "add_table",
                    "table_name": "income_analysis",
                    "columns": [
                        "Party",
                        "Gross Monthly Income",
                        "Deductions",
                        "Net Monthly Income",
                        "Income Sources",
                    ],
                },
            ],
        )

    def _get_visitation_schedule_template(self) -> str:
        """Get template for visitation schedule."""
        return """
## Visitation Schedule

### Regular Schedule
- [Specify weekday/weekend arrangement]
- [Specify pickup/dropoff times and locations]

### Holiday Schedule
| Holiday | Even Years | Odd Years |
|---------|-----------|-----------|
| New Year's | [Parent] | [Parent] |
| Memorial Day | [Parent] | [Parent] |
| Independence Day | [Parent] | [Parent] |
| Labor Day | [Parent] | [Parent] |
| Thanksgiving | [Parent] | [Parent] |
| Christmas Eve | [Parent] | [Parent] |
| Christmas Day | [Parent] | [Parent] |

### Summer Vacation
- [Specify summer schedule]
- [Notice requirements for vacation time]

### School Breaks
- [Spring break arrangements]
- [Winter break arrangements]
"""

    def _get_property_division_template(self) -> str:
        """Get template for property division."""
        return """
## Property Division

### Real Property
[List and disposition of real estate holdings]

### Business Interests
[Description of business ownership and valuation methodology]

### Retirement Accounts
| Account Type | Account Holder | Institution | Balance | Division |
|--------------|---------------|-------------|---------|----------|
| [Type] | [Name] | [Institution] | $[Amount] | [Split %] |

### Investment Accounts
[Description of investment holdings and division]

### Personal Property
[Division of significant personal property items]

### Debts and Liabilities
| Debt Type | Creditor | Balance | Responsible Party |
|-----------|----------|---------|-------------------|
| [Type] | [Name] | $[Amount] | [Party] |
"""

    def _get_income_calculation_template(self) -> str:
        """Get template for income calculation."""
        return """
## Income Calculation for Support Determination

### Petitioner Income Analysis
| Income Source | Monthly Amount | Annual Amount |
|--------------|----------------|---------------|
| Base Salary | $[Amount] | $[Amount] |
| Bonuses | $[Amount] | $[Amount] |
| Investment Income | $[Amount] | $[Amount] |
| Other Income | $[Amount] | $[Amount] |
| **Gross Total** | $[Amount] | $[Amount] |

### Respondent Income Analysis
| Income Source | Monthly Amount | Annual Amount |
|--------------|----------------|---------------|
| Base Salary | $[Amount] | $[Amount] |
| Bonuses | $[Amount] | $[Amount] |
| Investment Income | $[Amount] | $[Amount] |
| Other Income | $[Amount] | $[Amount] |
| **Gross Total** | $[Amount] | $[Amount] |

### Support Calculation
- Income differential: $[Amount] monthly
- Duration of marriage: [X] years
- Standard of living factor: [Description]
- Recommended support amount: $[Amount] monthly
- Recommended duration: [X] months/years
"""

    def register_scenario(
        self,
        scenario_name: str,
        description: str,
        detection_criteria: Dict[str, Any],
        actions: List[Dict[str, Any]],
        document_types: Optional[List[str]] = None,
        priority: int = 5,
    ) -> ScenarioRule:
        """
        Register a custom scenario rule.

        Args:
            scenario_name: Unique name for the scenario
            description: Description of what the scenario addresses
            detection_criteria: Conditions for detecting this scenario
            actions: List of actions to take when scenario is detected
            document_types: Applicable document types (None for all)
            priority: Priority level (1-10)

        Returns:
            The registered ScenarioRule
        """
        scenario = ScenarioRule(
            scenario_name=scenario_name,
            scenario_description=description,
            detection_criteria=detection_criteria,
            actions=actions,
            document_types=document_types or [],
            priority=priority,
        )

        self.scenarios[scenario_name] = scenario

        # Store in knowledge base
        self.knowledge_base.add_scenario(
            trigger=f"Scenario: {scenario_name}",
            response=json.dumps(actions),
        )

        # Store in memory for cross-session persistence
        self.memory_client.store(
            key=f"scenario/{scenario_name}",
            value=asdict(scenario),
            namespace="scenarios",
            tags=["scenario", scenario_name],
        )

        return scenario

    def detect_scenarios(
        self,
        document: Dict[str, Any],
    ) -> List[ScenarioRule]:
        """
        Detect which scenarios apply to a document.

        Args:
            document: Document data to analyze

        Returns:
            List of applicable scenarios
        """
        applicable = []
        doc_text = str(document).lower()
        doc_type = document.get("type", "").lower()

        for scenario in self.scenarios.values():
            if not scenario.is_active:
                continue

            # Check document type restriction
            if scenario.document_types:
                if not any(dt.lower() in doc_type for dt in scenario.document_types):
                    continue

            # Check keyword criteria
            criteria = scenario.detection_criteria
            keywords = criteria.get("keywords", [])
            keyword_match = any(kw.lower() in doc_text for kw in keywords)

            # Check condition criteria
            conditions = criteria.get("conditions", [])
            condition_match = self._evaluate_conditions(conditions, document)

            if keyword_match or condition_match:
                applicable.append(scenario)

        # Sort by priority
        applicable.sort(key=lambda s: s.priority, reverse=True)
        return applicable

    def _evaluate_conditions(
        self,
        conditions: List[str],
        document: Dict[str, Any],
    ) -> bool:
        """Evaluate condition expressions against document data."""
        if not conditions:
            return False

        for condition in conditions:
            # Parse simple conditions like "asset_value > 500000"
            parts = re.match(r"(\w+)\s*(>|<|>=|<=|==)\s*(\w+)", condition)
            if parts:
                field, op, value = parts.groups()
                doc_value = document.get(field)
                if doc_value is not None:
                    try:
                        doc_value = float(doc_value)
                        compare_value = float(value)
                        if op == ">" and doc_value > compare_value:
                            return True
                        elif op == "<" and doc_value < compare_value:
                            return True
                        elif op == ">=" and doc_value >= compare_value:
                            return True
                        elif op == "<=" and doc_value <= compare_value:
                            return True
                        elif op == "==" and doc_value == compare_value:
                            return True
                    except ValueError:
                        pass

            # Check boolean conditions
            if condition in document and document[condition]:
                return True

        return False

    def apply_scenarios(
        self,
        document: Dict[str, Any],
    ) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
        """
        Apply detected scenarios to a document.

        Args:
            document: Document to enhance

        Returns:
            Tuple of (enhanced document, list of applied actions)
        """
        applicable_scenarios = self.detect_scenarios(document)
        applied_actions = []

        for scenario in applicable_scenarios:
            for action in scenario.actions:
                action_type = action.get("action")

                if action_type == "add_section":
                    section_name = action.get("section")
                    content = action.get("content", "")
                    if "sections" not in document:
                        document["sections"] = {}
                    document["sections"][section_name] = content
                    applied_actions.append({
                        "scenario": scenario.scenario_name,
                        "action": action_type,
                        "section": section_name,
                    })

                elif action_type == "add_bullets":
                    location = action.get("location")
                    items = action.get("items", [])
                    if "bullets" not in document:
                        document["bullets"] = {}
                    document["bullets"][location] = items
                    applied_actions.append({
                        "scenario": scenario.scenario_name,
                        "action": action_type,
                        "location": location,
                        "items_added": len(items),
                    })

                elif action_type == "add_table":
                    table_name = action.get("table_name")
                    columns = action.get("columns", [])
                    if "tables" not in document:
                        document["tables"] = {}
                    document["tables"][table_name] = {
                        "columns": columns,
                        "rows": [],
                    }
                    applied_actions.append({
                        "scenario": scenario.scenario_name,
                        "action": action_type,
                        "table": table_name,
                    })

        return document, applied_actions

    def get_scenario_suggestions(
        self,
        document: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """
        Get suggestions based on detected scenarios without modifying document.

        Args:
            document: Document to analyze

        Returns:
            List of suggestions
        """
        suggestions = []
        applicable_scenarios = self.detect_scenarios(document)

        for scenario in applicable_scenarios:
            suggestion = {
                "scenario": scenario.scenario_name,
                "description": scenario.scenario_description,
                "priority": scenario.priority,
                "suggested_actions": [
                    {
                        "action": a.get("action"),
                        "detail": a.get("section") or a.get("location") or a.get("table_name"),
                    }
                    for a in scenario.actions
                ],
            }
            suggestions.append(suggestion)

        return suggestions

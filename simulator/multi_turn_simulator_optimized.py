#!/usr/bin/env python3
"""
Optimized Multi-turn Conversational Simulator for Telco Support Agent

This version reduces LLM calls through caching, templates, and rule-based logic
while maintaining realistic conversation flow.
"""

import argparse
import json
import random
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4
import hashlib

import openai
from dotenv import load_dotenv

# Add current directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

import requests

# Load environment variables
project_root = Path(__file__).parent.parent
env_local_path = project_root / ".env.local"
if env_local_path.exists():
    load_dotenv(env_local_path)


class PersonalityTrait(Enum):
    """User personality traits that affect conversation style."""
    PATIENT = "patient"
    FRUSTRATED = "frustrated"
    CONFUSED = "confused"
    TECHNICAL = "technical"
    NON_TECHNICAL = "non_technical"
    DETAILED = "detailed"
    BRIEF = "brief"


class IssueStatus(Enum):
    """Status of each issue in the conversation."""
    NOT_RAISED = "not_raised"
    RAISED = "raised"
    BEING_ADDRESSED = "being_addressed"
    RESOLVED = "resolved"
    ESCALATED = "escalated"


@dataclass
class Issue:
    """Represents a single issue the user wants to address."""
    category: str
    description: str
    priority: int  # 1=high, 2=medium, 3=low
    status: IssueStatus = IssueStatus.NOT_RAISED
    requires_followup: bool = False
    resolution_notes: str = ""


@dataclass
class Persona:
    """Customer persona with multiple issues and characteristics."""
    customer_id: str
    name: str
    personality_traits: List[PersonalityTrait]
    issues: List[Issue]
    background: str
    communication_style: str
    satisfaction_threshold: float = 0.7  # When to end conversation happily
    frustration_level: float = 0.0  # Current frustration (0-1)
    patience_level: float = 0.8  # How patient they are (0-1)


@dataclass
class ConversationState:
    """Tracks the state of an ongoing conversation."""
    conversation_id: str
    persona: Persona
    session_id: Optional[str] = None  # Agent session ID
    turn_number: int = 0
    messages: List[Dict[str, Any]] = field(default_factory=list)  # OpenAI format messages
    issues_raised: List[str] = field(default_factory=list)
    issues_resolved: List[str] = field(default_factory=list)
    current_issue_index: int = 0
    satisfaction_score: float = 0.5
    should_end: bool = False
    end_reason: str = ""
    agent_responses: List[str] = field(default_factory=list)  # Just store response strings
    start_time: datetime = field(default_factory=datetime.now)
    end_time: Optional[datetime] = None


class ResponseCache:
    """Cache for common responses to reduce LLM calls."""
    
    def __init__(self, ttl_seconds: int = 3600):
        self.cache: Dict[str, Tuple[str, float]] = {}
        self.ttl = ttl_seconds
        self.hit_count = 0
        self.miss_count = 0
    
    def _make_key(self, *args) -> str:
        """Create a cache key from arguments."""
        key_str = json.dumps(args, sort_keys=True, default=str)
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def get(self, *args) -> Optional[str]:
        """Get cached response if available and not expired."""
        key = self._make_key(*args)
        if key in self.cache:
            response, timestamp = self.cache[key]
            if time.time() - timestamp < self.ttl:
                self.hit_count += 1
                return response
            else:
                del self.cache[key]
        self.miss_count += 1
        return None
    
    def set(self, response: str, *args):
        """Cache a response."""
        key = self._make_key(*args)
        self.cache[key] = (response, time.time())
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total = self.hit_count + self.miss_count
        hit_rate = self.hit_count / total if total > 0 else 0
        return {
            "hits": self.hit_count,
            "misses": self.miss_count,
            "hit_rate": hit_rate,
            "size": len(self.cache)
        }


class ResponseTemplates:
    """Template-based responses for common scenarios."""
    
    # Closing messages by end reason and personality
    CLOSING_TEMPLATES = {
        ("all_issues_resolved", "patient"): ["thanks that worked", "perfect thanks", "all good now", "great, thanks!"],
        ("all_issues_resolved", "frustrated"): ["finally", "ok thanks", "about time", "alright"],
        ("all_issues_resolved", "brief"): ["thanks", "ok", "done", "good"],
        ("satisfied", "patient"): ["ok thanks", "that helps", "sounds good", "appreciate it"],
        ("satisfied", "frustrated"): ["fine", "ok", "whatever", "sure"],
        ("satisfied", "brief"): ["ok", "thx", "k", "got it"],
        ("no_progress", "patient"): ["i'll try later", "thanks anyway", "i'll call instead", "maybe another time"],
        ("no_progress", "frustrated"): ["this isn't working", "forget it", "waste of time", "useless"],
        ("no_progress", "brief"): ["bye", "no", "done", "leaving"],
        ("too_frustrated", "any"): ["forget it", "bye", "done", "leaving"],
        ("max_turns", "any"): ["gotta go", "bye", "thanks", "later"],
    }
    
    # Common follow-up responses by scenario
    FOLLOWUP_TEMPLATES = {
        "clarification_needed": ["what's that?", "how?", "can you explain?", "what do you mean?", "i don't understand"],
        "still_broken": ["still not working", "didn't help", "same problem", "no change", "still broken"],
        "acknowledgment": ["ok", "i see", "got it", "alright", "understood"],
        "need_help": ["can you help?", "what should i do?", "how do i fix this?", "help me", "what now?"],
        "frustrated_no_progress": ["this is taking forever", "come on", "seriously?", "still nothing?", "are you kidding?"],
        "checking": ["let me check", "checking now", "one sec", "hold on", "looking"],
        "confirmation": ["yes", "yeah", "yep", "correct", "that's right"],
        "denial": ["no", "nope", "not really", "negative", "that's not it"],
    }
    
    # Initial query templates by issue category
    INITIAL_QUERY_TEMPLATES = {
        "billing": ["why is my bill so high?", "check my bill please", "billing issue", "wrong charges on my bill", "need help with my bill"],
        "account": ["can't login", "account locked", "check my account", "account issues", "need account help"],
        "tech_support": ["phone not working", "no service", "can't make calls", "network issues", "connection problem"],
        "product": ["want to upgrade", "new plan info", "what plans do you have", "need a better plan", "plan options?"],
        "multi_domain": ["multiple issues", "several problems", "need help with a few things", "got some issues", "various problems"],
    }
    
    @classmethod
    def get_closing(cls, end_reason: str, traits: List[PersonalityTrait]) -> str:
        """Get a closing message based on reason and personality."""
        # Determine primary trait for selection
        if PersonalityTrait.FRUSTRATED in traits:
            trait_key = "frustrated"
        elif PersonalityTrait.BRIEF in traits:
            trait_key = "brief"
        elif PersonalityTrait.PATIENT in traits:
            trait_key = "patient"
        else:
            trait_key = "patient"
        
        # Try specific combination first
        templates = cls.CLOSING_TEMPLATES.get((end_reason, trait_key))
        if not templates:
            # Fall back to generic
            templates = cls.CLOSING_TEMPLATES.get((end_reason, "any"), ["bye", "thanks", "ok"])
        
        return random.choice(templates)
    
    @classmethod
    def get_followup(cls, scenario: str) -> str:
        """Get a follow-up response for a scenario."""
        templates = cls.FOLLOWUP_TEMPLATES.get(scenario, ["ok", "i see"])
        return random.choice(templates)
    
    @classmethod
    def get_initial_query(cls, category: str, description: str = None) -> Optional[str]:
        """Get an initial query template if available."""
        # For simple cases, use templates
        if random.random() < 0.3:  # 30% chance to use template
            templates = cls.INITIAL_QUERY_TEMPLATES.get(category)
            if templates:
                return random.choice(templates)
        return None  # Fall back to LLM generation


class PersonaGenerator:
    """Generates realistic customer personas with multiple issues."""
    
    # Rich issue templates from original
    ISSUE_TEMPLATES = {
        "account": [
            "customer wants to know their current plan details",
            "customer needs to verify their subscription status",
            "customer is asking about autopay settings",
            "customer wants to know when their account was created",
            "customer needs loyalty tier information",
            "customer is checking contract length and renewal dates",
            "customer wants to see all active subscriptions",
            "customer is asking about account segment and benefits",
            "customer needs to verify billing preferences",
            "customer wants to check if they qualify for upgrades"
        ],
        "billing": [
            "customer sees unexpected charges on their bill",
            "customer wants to know when payment is due",
            "customer needs breakdown of current month charges",
            "customer is questioning data usage amounts",
            "customer wants payment history for tax purposes",
            "customer needs to understand prorated charges",
            "customer is asking about auto-pay status",
            "customer wants to dispute a specific charge",
            "customer needs usage details for expense reporting",
            "customer is planning data usage for upcoming month",
        ],
        "tech_support": [
            "customer's phone won't connect to network",
            "customer has slow data speeds",
            "customer can't receive calls but can make them",
            "customer needs help setting up international roaming",
            "customer's voicemail isn't working",
            "customer has poor signal at home",
            "customer's new device won't activate",
            "customer needs help with WiFi calling setup",
            "customer is getting error messages on their phone",
            "customer needs troubleshooting for specific device issues"
        ],
        "product": [
            "customer wants to compare available plans",
            "customer is looking for device upgrade options",
            "customer needs information about current promotions",
            "customer wants to know about 5G compatibility",
            "customer is asking about plan features and benefits",
            "customer needs device specifications and pricing",
            "customer wants to understand family plan options",
            "customer is checking device trade-in values",
            "customer needs information about business plans",
            "customer wants to know about unlimited data options"
        ],
        "multi_domain": [
            "customer's bill is high and wants to explore plan downgrade options",
            "customer wants to understand usage charges and see if different plan would save money",
            "customer is questioning overage fees and needs plan recommendations",
            "customer wants complete account review including usage, billing, and plan optimization",
        ]
    }
    
    PERSONALITY_COMBINATIONS = [
        ([PersonalityTrait.PATIENT, PersonalityTrait.TECHNICAL], 
         "Tech-savvy customer who understands technical details and is patient"),
        ([PersonalityTrait.FRUSTRATED, PersonalityTrait.BRIEF], 
         "Frustrated customer who wants quick answers"),
        ([PersonalityTrait.CONFUSED, PersonalityTrait.NON_TECHNICAL], 
         "Non-technical customer who needs simple explanations"),
        ([PersonalityTrait.DETAILED, PersonalityTrait.PATIENT], 
         "Customer who wants thorough explanations and has time"),
        ([PersonalityTrait.FRUSTRATED, PersonalityTrait.TECHNICAL], 
         "Technical customer frustrated with service issues"),
    ]
    
    FIRST_NAMES = ["Sarah", "Michael", "Jessica", "David", "Emily", "Robert", "Amanda", "James"]
    LAST_NAMES = ["Chen", "Rodriguez", "Williams", "Park", "Johnson", "Taylor", "Davis", "Wilson"]
    
    def __init__(self, customer_id_start: int = 10001, customer_count: int = 1500):
        self.customer_id_start = customer_id_start
        self.customer_count = customer_count
    
    def generate_customer_id(self) -> str:
        """Generate a random customer ID."""
        customer_num = random.randint(
            self.customer_id_start, 
            self.customer_id_start + self.customer_count - 1
        )
        return f"CUS-{customer_num:05d}"
    
    def generate_issues(self, num_issues: int = None, is_multi_domain: bool = None) -> List[Issue]:
        """Generate a set of issues for a persona."""
        if is_multi_domain is None:
            is_multi_domain = random.random() < 0.2
        
        categories = list(self.ISSUE_TEMPLATES.keys())
        issues = []
        
        if is_multi_domain:
            if num_issues is None:
                num_issues = random.choice([2, 3])
            
            multi_scenario = random.choice(self.ISSUE_TEMPLATES["multi_domain"])
            issues.append(Issue(
                category="multi_domain",
                description=multi_scenario,
                priority=1,
                requires_followup=random.random() < 0.4
            ))
            
            related_categories = self._extract_related_categories(multi_scenario)
            if not related_categories:
                related_categories = [c for c in categories if c != "multi_domain"]
            
            for i in range(num_issues - 1):
                category = random.choice(related_categories)
                issue_desc = random.choice(self.ISSUE_TEMPLATES[category])
                issues.append(Issue(
                    category=category,
                    description=issue_desc,
                    priority=i + 2,
                    requires_followup=random.random() < 0.2
                ))
        else:
            if num_issues is None:
                num_issues = 1
            
            single_categories = [c for c in categories if c != "multi_domain"]
            category = random.choice(single_categories)
            
            issue_desc = random.choice(self.ISSUE_TEMPLATES[category])
            issues.append(Issue(
                category=category,
                description=issue_desc,
                priority=1,
                requires_followup=random.random() < 0.3
            ))
        
        return issues
    
    def _extract_related_categories(self, scenario: str) -> List[str]:
        """Extract which categories are mentioned in a multi-domain scenario."""
        scenario_lower = scenario.lower()
        related = []
        
        if any(word in scenario_lower for word in ["bill", "charge", "payment", "usage", "data"]):
            related.append("billing")
        if any(word in scenario_lower for word in ["account", "login", "subscription", "autopay", "contract"]):
            related.append("account")
        if any(word in scenario_lower for word in ["device", "plan", "upgrade", "5g", "family"]):
            related.append("product")
        if any(word in scenario_lower for word in ["connect", "signal", "roaming", "activate", "working"]):
            related.append("tech_support")
        
        return related if related else ["billing", "account", "product", "tech_support"]
    
    def generate_persona(self, force_multi_domain: bool = False) -> Persona:
        """Generate a complete customer persona."""
        first_name = random.choice(self.FIRST_NAMES)
        last_name = random.choice(self.LAST_NAMES)
        customer_id = self.generate_customer_id()
        
        traits, background = random.choice(self.PERSONALITY_COMBINATIONS)
        
        issues = self.generate_issues(is_multi_domain=force_multi_domain)
        
        if PersonalityTrait.BRIEF in traits:
            communication_style = "Short, direct questions. Gets impatient with long explanations."
        elif PersonalityTrait.DETAILED in traits:
            communication_style = "Asks detailed follow-up questions. Wants complete understanding."
        elif PersonalityTrait.TECHNICAL in traits:
            communication_style = "Uses technical terms. Wants specific technical details."
        else:
            communication_style = "Conversational, asks clarifying questions when confused."
        
        patience = 0.8 if PersonalityTrait.PATIENT in traits else 0.4
        satisfaction_threshold = 0.6 if PersonalityTrait.PATIENT in traits else 0.8
        
        return Persona(
            customer_id=customer_id,
            name=f"{first_name} {last_name}",
            personality_traits=traits,
            issues=issues,
            background=background,
            communication_style=communication_style,
            patience_level=patience,
            satisfaction_threshold=satisfaction_threshold
        )


class OptimizedUserSimulator:
    """Optimized LLM-powered user simulator with caching and templates."""
    
    def __init__(self, llm_base_url: str = "http://0.0.0.0:4000", 
                 api_key: str = "sk-1234567890", model: str = "gpt-5-nano",
                 use_cache: bool = True, use_templates: bool = True):
        self.client = openai.OpenAI(base_url=llm_base_url, api_key=api_key)
        self.model = model
        self.use_cache = use_cache
        self.use_templates = use_templates
        self.cache = ResponseCache() if use_cache else None
        self.llm_calls = 0
        self.template_uses = 0
        self.cache_uses = 0
    
    def generate_initial_query(self, persona: Persona, state: ConversationState) -> str:
        """Generate the first message from the user."""
        first_issue = persona.issues[0]
        
        # Try template first
        if self.use_templates:
            template_response = ResponseTemplates.get_initial_query(
                first_issue.category, first_issue.description
            )
            if template_response:
                self.template_uses += 1
                return template_response
        
        # Check cache
        if self.cache:
            cached = self.cache.get("initial", first_issue.category, first_issue.description,
                                   persona.personality_traits)
            if cached:
                self.cache_uses += 1
                return cached
        
        # Fall back to LLM
        system_prompt = f"""You are {persona.name}, a customer typing in a chat support window.

You need help with: {first_issue.description}

IMPORTANT RULES FOR CHAT:
- Write like you're typing on your phone (15-30 words MAX)
- Be informal and direct
- NO long explanations or multiple questions
- Use contractions (don't, can't, won't)

Personality: {'frustrated and brief' if PersonalityTrait.FRUSTRATED in persona.personality_traits else 'casual'}"""

        user_prompt = "Type your first message (remember: 15-30 words, casual chat style):"
        
        self.llm_calls += 1
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.8,
            max_tokens=50
        )
        
        result = response.choices[0].message.content.strip()
        
        # Cache the result
        if self.cache:
            self.cache.set(result, "initial", first_issue.category, first_issue.description,
                          persona.personality_traits)
        
        return result
    
    def generate_response(self, persona: Persona, state: ConversationState, 
                         agent_response: str) -> Tuple[str, bool, str]:
        """Generate user's response based on agent's message."""
        # Analyze and calculate in one go
        analysis_result = self._analyze_and_evaluate(persona, state, agent_response)
        
        # Update state based on analysis
        state.satisfaction_score = analysis_result["satisfaction"]
        for issue in persona.issues:
            if issue.description in analysis_result["addressed_issues"]:
                issue.status = IssueStatus.RESOLVED
                state.issues_resolved.append(issue.description)
        
        # Rule-based decision on ending conversation
        should_end, end_reason = self._should_end_conversation(persona, state)
        
        if should_end:
            # Use template for closing
            if self.use_templates:
                closing = ResponseTemplates.get_closing(end_reason, persona.personality_traits)
                self.template_uses += 1
            else:
                closing = self._generate_closing_message(persona, state, end_reason)
            return closing, True, end_reason
        
        # Generate continuation
        return self._generate_continuation(persona, state, agent_response, 
                                          analysis_result["addressed_issues"]), False, ""
    
    def _analyze_and_evaluate(self, persona: Persona, state: ConversationState,
                              agent_response: str) -> Dict[str, Any]:
        """Combined analysis of addressed issues and satisfaction calculation."""
        # Check cache first
        cache_key = ("analysis", agent_response[:200], [i.description for i in persona.issues if i.status in [IssueStatus.RAISED, IssueStatus.BEING_ADDRESSED]])
        
        if self.cache:
            cached = self.cache.get(*cache_key)
            if cached:
                self.cache_uses += 1
                return json.loads(cached)
        
        # Rule-based analysis for simple cases
        addressed_issues = []
        agent_lower = agent_response.lower()
        
        for issue in persona.issues:
            if issue.status in [IssueStatus.RAISED, IssueStatus.BEING_ADDRESSED]:
                # Simple keyword matching for common resolutions
                issue_keywords = self._extract_issue_keywords(issue.description)
                if any(keyword in agent_lower for keyword in issue_keywords):
                    if any(resolution in agent_lower for resolution in 
                          ["fixed", "resolved", "activated", "updated", "changed", "processed", "completed"]):
                        addressed_issues.append(issue.description)
        
        # If rule-based found something or it's a simple response, use it
        if addressed_issues or len(agent_response) < 100:
            satisfaction = self._calculate_satisfaction_rules(
                persona, state, bool(addressed_issues)
            )
            result = {
                "addressed_issues": addressed_issues,
                "satisfaction": satisfaction
            }
            
            if self.cache:
                self.cache.set(json.dumps(result), *cache_key)
            
            return result
        
        # Fall back to LLM for complex analysis
        system_prompt = """Analyze the agent's response and return JSON with:
1. "addressed_issues": list of issue descriptions that were meaningfully resolved
2. "satisfaction_delta": float between -0.2 and 0.2 for satisfaction change

Issues to check:
"""
        for issue in persona.issues:
            if issue.status in [IssueStatus.RAISED, IssueStatus.BEING_ADDRESSED]:
                system_prompt += f"- {issue.description}\n"
        
        system_prompt += "\nReturn only valid JSON."
        
        self.llm_calls += 1
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Agent response: {agent_response}"}
            ],
            temperature=0.3,
            max_tokens=200
        )
        
        try:
            llm_result = json.loads(response.choices[0].message.content)
            addressed = llm_result.get("addressed_issues", [])
            delta = llm_result.get("satisfaction_delta", 0)
            
            satisfaction = max(0.0, min(1.0, state.satisfaction_score + delta))
            
            result = {
                "addressed_issues": addressed,
                "satisfaction": satisfaction
            }
            
            if self.cache:
                self.cache.set(json.dumps(result), *cache_key)
            
            return result
        except:
            # Fallback if LLM response is malformed
            return {
                "addressed_issues": [],
                "satisfaction": state.satisfaction_score
            }
    
    def _extract_issue_keywords(self, description: str) -> List[str]:
        """Extract keywords from issue description for matching."""
        keywords = []
        desc_lower = description.lower()
        
        # Extract key terms
        if "bill" in desc_lower or "charge" in desc_lower:
            keywords.extend(["bill", "charge", "payment", "invoice"])
        if "data" in desc_lower:
            keywords.extend(["data", "usage", "gb", "megabyte"])
        if "plan" in desc_lower:
            keywords.extend(["plan", "subscription", "package"])
        if "account" in desc_lower:
            keywords.extend(["account", "profile", "settings"])
        if "device" in desc_lower or "phone" in desc_lower:
            keywords.extend(["device", "phone", "handset"])
        if "network" in desc_lower or "signal" in desc_lower:
            keywords.extend(["network", "signal", "connection", "service"])
        
        return keywords
    
    def _calculate_satisfaction_rules(self, persona: Persona, state: ConversationState,
                                     made_progress: bool) -> float:
        """Calculate satisfaction using rules."""
        score = state.satisfaction_score
        
        if made_progress:
            score += 0.15
        elif state.turn_number > 3:
            score -= 0.1
        
        # Personality adjustments
        if PersonalityTrait.PATIENT in persona.personality_traits:
            score += 0.02
        if PersonalityTrait.FRUSTRATED in persona.personality_traits:
            score -= 0.08
        
        return max(0.0, min(1.0, score))
    
    def _should_end_conversation(self, persona: Persona, 
                                state: ConversationState) -> Tuple[bool, str]:
        """Determine if the conversation should end (rule-based)."""
        resolved_count = sum(1 for i in persona.issues if i.status == IssueStatus.RESOLVED)
        
        if resolved_count == len(persona.issues):
            return True, "all_issues_resolved"
        
        if state.satisfaction_score >= persona.satisfaction_threshold and resolved_count >= len(persona.issues) - 1:
            return True, "satisfied"
        
        if state.turn_number > 8 and resolved_count == 0:
            return True, "no_progress"
        
        if persona.frustration_level > 0.9:
            return True, "too_frustrated"
        
        if state.turn_number > 12:
            return True, "max_turns"
        
        return False, ""
    
    def _generate_closing_message(self, persona: Persona, state: ConversationState, 
                                 reason: str) -> str:
        """Generate a closing message using LLM as fallback."""
        # This is only called if templates are disabled
        self.llm_calls += 1
        
        tone_map = {
            "all_issues_resolved": "satisfied",
            "satisfied": "content",
            "no_progress": "frustrated",
            "too_frustrated": "very frustrated"
        }
        tone = tone_map.get(reason, "neutral")
        
        system_prompt = f"""Type a SHORT chat closing message.
Tone: {tone}
Maximum 10 words, casual chat style."""
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": "Type your closing message:"}
            ],
            temperature=0.7,
            max_tokens=30
        )
        
        return response.choices[0].message.content.strip()
    
    def _generate_continuation(self, persona: Persona, state: ConversationState,
                             agent_response: str, addressed_issues: List[str]) -> str:
        """Generate a continuation message."""
        # Determine scenario
        scenario = self._determine_scenario(persona, state, addressed_issues)
        
        # Try template first
        if self.use_templates and random.random() < 0.5:  # 50% chance for templates
            template_response = self._try_template_response(scenario, persona)
            if template_response:
                self.template_uses += 1
                return template_response
        
        # Check cache
        if self.cache:
            cache_key = ("continuation", scenario, agent_response[:100], 
                        persona.personality_traits, state.turn_number)
            cached = self.cache.get(*cache_key)
            if cached:
                self.cache_uses += 1
                return cached
        
        # Generate with strict constraints from the start
        max_words = 15 if PersonalityTrait.BRIEF in persona.personality_traits else 20
        
        system_prompt = f"""You are typing in a chat window.

STRICT RULES:
- Maximum {max_words} words
- Type like you're on your phone
- One thought only
- lowercase ok

Action: {scenario}

Type your message ({max_words} words MAX):"""
        
        self.llm_calls += 1
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Agent said: {agent_response[:200]}"}
            ],
            temperature=0.8,
            max_tokens=40
        )
        
        result = response.choices[0].message.content.strip()
        
        # Validate length
        if len(result.split()) > max_words * 1.2:
            # Too long, use a template fallback
            if self.use_templates:
                result = ResponseTemplates.get_followup("acknowledgment")
                self.template_uses += 1
        
        # Cache the result
        if self.cache:
            self.cache.set(result, *cache_key)
        
        return result
    
    def _determine_scenario(self, persona: Persona, state: ConversationState,
                           addressed_issues: List[str]) -> str:
        """Determine the conversation scenario."""
        unraised_issues = [i for i in persona.issues if i.status == IssueStatus.NOT_RAISED]
        unresolved_issues = [i for i in persona.issues 
                           if i.status in [IssueStatus.RAISED, IssueStatus.BEING_ADDRESSED]]
        
        if addressed_issues and unraised_issues and random.random() < 0.85:
            unraised_issues[0].status = IssueStatus.RAISED
            state.issues_raised.append(unraised_issues[0].description)
            return f"raise new issue: {unraised_issues[0].description}"
        elif unresolved_issues and random.random() < 0.7:
            unresolved_issues[0].status = IssueStatus.BEING_ADDRESSED
            return f"follow up on: {unresolved_issues[0].description}"
        elif not addressed_issues and state.turn_number > 2:
            persona.frustration_level += 0.15
            return "express frustration"
        else:
            return "ask clarification"
    
    def _try_template_response(self, scenario: str, persona: Persona) -> Optional[str]:
        """Try to use a template response for the scenario."""
        if "express frustration" in scenario:
            return ResponseTemplates.get_followup("frustrated_no_progress")
        elif "ask clarification" in scenario:
            return ResponseTemplates.get_followup("clarification_needed")
        elif "follow up" in scenario:
            return ResponseTemplates.get_followup("still_broken")
        elif "raise new issue" in scenario and PersonalityTrait.BRIEF in persona.personality_traits:
            # For brief personalities, use short transition
            return random.choice(["also", "and another thing", "btw", "oh and"])
        
        return None
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about simulator performance."""
        stats = {
            "llm_calls": self.llm_calls,
            "template_uses": self.template_uses,
            "cache_uses": self.cache_uses,
            "total_responses": self.llm_calls + self.template_uses + self.cache_uses
        }
        
        if stats["total_responses"] > 0:
            stats["llm_percentage"] = self.llm_calls / stats["total_responses"] * 100
            stats["optimization_rate"] = (self.template_uses + self.cache_uses) / stats["total_responses"] * 100
        
        if self.cache:
            stats["cache_stats"] = self.cache.get_stats()
        
        return stats


class AgentClient:
    """Simple client for the agent API."""
    
    def __init__(self, base_url: str = "http://127.0.0.1:8010"):
        self.base_url = base_url
        self.api_url = f"{base_url}/chat"
    
    def health_check(self) -> bool:
        """Check if the agent is running."""
        try:
            # First try a simple GET to the base URL
            response = requests.get(self.base_url, timeout=2)
            return True  # If we can connect, server is running
        except:
            # Fallback to trying the chat endpoint with minimal data
            try:
                response = requests.post(
                    self.api_url,
                    json={"messages": [{"role": "user", "content": "hi"}], "customer_id": "test"},
                    timeout=10  # Increased timeout
                )
                return response.status_code == 200
            except Exception as e:
                print(f"Health check failed: {e}")
                return False
    
    def send_message(self, message: str, messages: List[Dict[str, Any]], 
                     customer_id: str, session_id: Optional[str] = None) -> Tuple[str, List[Dict[str, Any]], str]:
        """Send a message to the agent and get response."""
        messages_with_user = messages + [{"role": "user", "content": message}]
        
        payload = {
            "messages": messages_with_user,
            "customer_id": customer_id
        }
        
        if session_id:
            payload["session_id"] = session_id
        
        try:
            response = requests.post(self.api_url, json=payload)
            response.raise_for_status()
            
            data = response.json()
            updated_messages = data["messages"]
            returned_session_id = data.get("session_id", session_id)
            
            agent_response = None
            for msg in reversed(updated_messages):
                if msg.get("role") == "assistant":
                    agent_response = msg.get("content")
                    break
            
            if not agent_response:
                raise Exception("No assistant response found in messages")
            
            return agent_response, updated_messages, returned_session_id
            
        except requests.exceptions.RequestException as e:
            raise Exception(f"Error calling API: {e}")


class ConversationOrchestrator:
    """Orchestrates multi-turn conversations between user simulator and agent."""
    
    def __init__(self, user_simulator: OptimizedUserSimulator, agent_client: AgentClient,
                 persona_generator: PersonaGenerator, verbose: bool = False):
        self.user_simulator = user_simulator
        self.agent_client = agent_client
        self.persona_generator = persona_generator
        self.verbose = verbose
    
    def run_conversation(self, persona: Optional[Persona] = None, 
                        force_multi_domain: bool = False) -> ConversationState:
        """Run a complete multi-turn conversation."""
        if persona is None:
            persona = self.persona_generator.generate_persona(force_multi_domain=force_multi_domain)
        
        state = ConversationState(
            conversation_id=str(uuid4()),
            persona=persona
        )
        
        print(f"\n{'='*60}")
        print(f"Starting conversation for {persona.name} ({persona.customer_id})")
        print(f"Issues to address: {len(persona.issues)}")
        for i, issue in enumerate(persona.issues, 1):
            print(f"  {i}. [{issue.category}] {issue.description}")
        print(f"Personality: {', '.join([t.value for t in persona.personality_traits])}")
        print(f"{'='*60}\n")
        
        try:
            # Generate initial query
            initial_query = self.user_simulator.generate_initial_query(persona, state)
            persona.issues[0].status = IssueStatus.RAISED
            state.issues_raised.append(persona.issues[0].description)
            
            print(f"👤 User: {initial_query}\n")
            
            current_message = initial_query
            
            while not state.should_end and state.turn_number < 15:
                state.turn_number += 1
                
                try:
                    agent_response, updated_messages, session_id = self.agent_client.send_message(
                        current_message,
                        state.messages,
                        persona.customer_id,
                        state.session_id
                    )
                    
                    if not state.session_id and session_id:
                        state.session_id = session_id
                        if self.verbose:
                            print(f"📝 Session ID: {session_id}")
                    
                    print(f"🤖 Agent: {agent_response}\n")
                    
                    state.messages = updated_messages
                    state.agent_responses.append(agent_response)
                    
                    user_response, should_end, end_reason = self.user_simulator.generate_response(
                        persona, state, agent_response
                    )
                    
                    if should_end:
                        print(f"👤 User: {user_response}\n")
                        state.messages.append({"role": "user", "content": user_response})
                        state.should_end = True
                        state.end_reason = end_reason
                        break
                    
                    print(f"👤 User: {user_response}\n")
                    current_message = user_response
                    
                    time.sleep(0.5)
                    
                except Exception as e:
                    print(f"❌ Error in conversation: {e}")
                    state.should_end = True
                    state.end_reason = "error"
                    break
            
            state.end_time = datetime.now()
            
            self._print_conversation_summary(state)
            
        except Exception as e:
            print(f"❌ Conversation failed: {e}")
            state.should_end = True
            state.end_reason = "error"
        
        return state
    
    def _print_conversation_summary(self, state: ConversationState):
        """Print a summary of the conversation."""
        duration = (state.end_time - state.start_time).total_seconds() if state.end_time else 0
        
        print(f"\n{'='*60}")
        print(f"CONVERSATION SUMMARY")
        print(f"{'='*60}")
        print(f"Conversation ID: {state.conversation_id}")
        print(f"Session ID: {state.session_id}")
        print(f"Customer: {state.persona.name} ({state.persona.customer_id})")
        print(f"Duration: {duration:.1f} seconds")
        print(f"Total turns: {state.turn_number}")
        print(f"End reason: {state.end_reason}")
        print(f"Final satisfaction: {state.satisfaction_score:.2f}")
        
        print(f"\nIssue Resolution:")
        for issue in state.persona.issues:
            status_icon = "✅" if issue.status == IssueStatus.RESOLVED else "❌"
            print(f"  {status_icon} [{issue.category}] {issue.description} - {issue.status.value}")
        
        resolved_count = sum(1 for i in state.persona.issues if i.status == IssueStatus.RESOLVED)
        print(f"\nResolution rate: {resolved_count}/{len(state.persona.issues)} issues resolved")
        print(f"{'='*60}\n")


def main():
    """Main function to run the optimized multi-turn simulator."""
    parser = argparse.ArgumentParser(description='Optimized Multi-turn Conversational Simulator')
    parser.add_argument('--num-conversations', type=int, default=1,
                       help='Number of conversations to simulate')
    parser.add_argument('--agent-url', default='http://127.0.0.1:8010',
                       help='Agent API URL')
    parser.add_argument('--llm-url', default='http://0.0.0.0:4000',
                       help='LLM API URL')
    parser.add_argument('--llm-model', default='gpt-5',
                       help='LLM model name')
    parser.add_argument('--export-json', type=str,
                       help='Export conversations to JSON file')
    parser.add_argument('--verbose', action='store_true',
                       help='Show detailed conversation logs')
    parser.add_argument('--force-multi-domain', action='store_true',
                       help='Force multi-domain issues for testing')
    parser.add_argument('--no-cache', action='store_true',
                       help='Disable response caching')
    parser.add_argument('--no-templates', action='store_true',
                       help='Disable template responses')
    parser.add_argument('--show-stats', action='store_true',
                       help='Show optimization statistics')
    
    args = parser.parse_args()
    
    # Initialize components
    print("Initializing optimized multi-turn conversation simulator...")
    
    persona_generator = PersonaGenerator()
    user_simulator = OptimizedUserSimulator(
        llm_base_url=args.llm_url,
        model=args.llm_model,
        use_cache=not args.no_cache,
        use_templates=not args.no_templates
    )
    agent_client = AgentClient(base_url=args.agent_url)
    
    # Check agent health
    print(f"Checking agent at {args.agent_url}...")
    if not agent_client.health_check():
        print("❌ Agent is not running! Please start it with:")
        print("   ./run.sh")
        print("   or: uvicorn load_test_agent.main:app --host 0.0.0.0 --port 8010")
        return 1
    
    print("✅ Agent is healthy\n")
    
    if not args.no_cache:
        print("✅ Response caching enabled")
    if not args.no_templates:
        print("✅ Template responses enabled")
    print()
    
    # Create orchestrator
    orchestrator = ConversationOrchestrator(
        user_simulator=user_simulator,
        agent_client=agent_client,
        persona_generator=persona_generator,
        verbose=args.verbose
    )
    
    # Run conversations
    all_states = []
    for i in range(args.num_conversations):
        print(f"\n{'#'*60}")
        print(f"# CONVERSATION {i+1}/{args.num_conversations}")
        print(f"{'#'*60}")
        
        state = orchestrator.run_conversation(force_multi_domain=args.force_multi_domain)
        all_states.append(state)
        
        if i < args.num_conversations - 1:
            time.sleep(2)
    
    # Print overall summary
    if len(all_states) > 1:
        print(f"\n{'='*60}")
        print(f"OVERALL SUMMARY")
        print(f"{'='*60}")
        print(f"Total conversations: {len(all_states)}")
        
        total_turns = sum(s.turn_number for s in all_states)
        avg_turns = total_turns / len(all_states)
        print(f"Average turns per conversation: {avg_turns:.1f}")
        
        total_issues = sum(len(s.persona.issues) for s in all_states)
        resolved_issues = sum(
            sum(1 for i in s.persona.issues if i.status == IssueStatus.RESOLVED)
            for s in all_states
        )
        print(f"Overall resolution rate: {resolved_issues}/{total_issues} ({resolved_issues/total_issues*100:.1f}%)")
        
        avg_satisfaction = sum(s.satisfaction_score for s in all_states) / len(all_states)
        print(f"Average satisfaction: {avg_satisfaction:.2f}")
        
        end_reasons = {}
        for s in all_states:
            end_reasons[s.end_reason] = end_reasons.get(s.end_reason, 0) + 1
        print(f"\nEnd reasons:")
        for reason, count in end_reasons.items():
            print(f"  {reason}: {count}")
    
    # Show optimization statistics
    if args.show_stats:
        print(f"\n{'='*60}")
        print(f"OPTIMIZATION STATISTICS")
        print(f"{'='*60}")
        stats = user_simulator.get_stats()
        print(f"Total responses generated: {stats['total_responses']}")
        print(f"LLM calls: {stats['llm_calls']} ({stats.get('llm_percentage', 0):.1f}%)")
        print(f"Template uses: {stats['template_uses']}")
        print(f"Cache uses: {stats['cache_uses']}")
        print(f"Optimization rate: {stats.get('optimization_rate', 0):.1f}%")
        
        if 'cache_stats' in stats:
            cache_stats = stats['cache_stats']
            print(f"\nCache performance:")
            print(f"  Hits: {cache_stats['hits']}")
            print(f"  Misses: {cache_stats['misses']}")
            print(f"  Hit rate: {cache_stats['hit_rate']*100:.1f}%")
            print(f"  Cached items: {cache_stats['size']}")
    
    # Export to JSON if requested
    if args.export_json:
        export_data = []
        for state in all_states:
            export_data.append({
                "conversation_id": state.conversation_id,
                "session_id": state.session_id,
                "customer_id": state.persona.customer_id,
                "customer_name": state.persona.name,
                "personality_traits": [t.value for t in state.persona.personality_traits],
                "issues": [
                    {
                        "category": i.category,
                        "description": i.description,
                        "priority": i.priority,
                        "status": i.status.value,
                        "resolved": i.status == IssueStatus.RESOLVED
                    }
                    for i in state.persona.issues
                ],
                "turn_count": state.turn_number,
                "messages": state.messages,
                "satisfaction_score": state.satisfaction_score,
                "end_reason": state.end_reason,
                "duration_seconds": (state.end_time - state.start_time).total_seconds() if state.end_time else 0,
                "resolution_rate": sum(1 for i in state.persona.issues if i.status == IssueStatus.RESOLVED) / len(state.persona.issues)
            })
        
        # Add optimization stats to export
        if args.show_stats:
            export_data.append({
                "optimization_stats": user_simulator.get_stats()
            })
        
        with open(args.export_json, 'w') as f:
            json.dump(export_data, f, indent=2, default=str)
        print(f"\n📄 Conversations exported to {args.export_json}")
    
    print("\n✅ Simulation complete!")
    return 0


if __name__ == "__main__":
    exit(main())
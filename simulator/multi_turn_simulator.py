#!/usr/bin/env python3
"""
Multi-turn Conversational Simulator for Telco Support Agent

This simulator creates realistic multi-turn conversations between
simulated users (powered by LLM) and the telco support agent.
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


class PersonaGenerator:
    """Generates realistic customer personas with multiple issues."""
    
    # Rich issue templates from local_query_generation.py
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
            "customer wants to know their data usage for a specific month",
            "customer needs to check voice minutes used in last billing cycle",
            "customer is asking about SMS usage over a date range",
            "customer wants to compare usage between different months",
            "customer needs total usage breakdown for expense reporting",
            "customer is checking if they're approaching data limits",
            "customer wants to analyze usage patterns over time",
            "customer needs usage details for a specific billing period",
            "customer is tracking usage to optimize their plan",
            "customer wants to know peak usage periods",
            "customer needs usage data for tax deduction purposes",
            "customer is monitoring family member usage on shared plan",
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
            # Billing + Product scenarios
            "customer's bill is high and wants to explore plan downgrade options",
            "customer wants to understand usage charges and see if different plan would save money",
            "customer is questioning overage fees and needs plan recommendations",
            "customer wants to compare their current plan costs with available alternatives",
            # Account + Billing scenarios
            "customer can't access online account and needs to check payment status",
            "customer's autopay failed and wants to verify account settings",
            "customer moved and needs to update account info and understand billing changes",
            # Tech Support + Product scenarios
            "customer has connectivity issues and wants to know if device upgrade would help",
            "customer's device isn't working properly and wondering about replacement options",
            "customer needs help setting up new device and wants to know about compatible plans",
            # Tech Support + Account scenarios
            "customer can't receive calls and wants to verify their account is active",
            "customer has service issues and wants to check if account suspension is the cause",
            # Billing + Tech Support scenarios
            "customer is being charged for services that aren't working properly",
            "customer has high data charges and needs help understanding usage patterns",
            # Account + Product scenarios
            "customer wants to add a line and needs to understand how it affects their current plan",
            "customer's contract is ending and wants to review upgrade options",
            # Triple domain scenarios
            "customer wants complete account review including usage, billing, and plan optimization",
            "customer is considering canceling and wants full service and cost analysis",
            "customer got married and needs to merge accounts, understand billing, and explore family plans",
            # All four domains
            "customer has service issues, billing disputes, wants account changes, and device upgrades",
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
        """Generate a set of issues for a persona.
        
        Args:
            num_issues: Override for number of issues
            is_multi_domain: Whether this is a multi-domain persona
        """
        # Decide if multi-domain if not specified
        if is_multi_domain is None:
            # 20% chance of multi-domain
            is_multi_domain = random.random() < 0.2
        
        categories = list(self.ISSUE_TEMPLATES.keys())
        issues = []
        
        if is_multi_domain:
            # Multi-domain: 2-3 issues
            if num_issues is None:
                num_issues = random.choice([2, 3])
            
            # Start with a multi-domain scenario
            multi_scenario = random.choice(self.ISSUE_TEMPLATES["multi_domain"])
            issues.append(Issue(
                category="multi_domain",
                description=multi_scenario,
                priority=1,
                requires_followup=random.random() < 0.4
            ))
            
            # Add 1-2 related single-domain issues
            # Pick categories mentioned in the multi-domain scenario
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
            # Single-domain: exactly 1 issue
            if num_issues is None:
                num_issues = 1
            
            # Pick a single-domain category (not multi_domain)
            single_categories = [c for c in categories if c != "multi_domain"]
            category = random.choice(single_categories)
            
            # Pick one issue from that category
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
        
        # Check for category keywords
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
        """Generate a complete customer persona.
        
        Args:
            force_multi_domain: Force generation of multi-domain issues for testing
        """
        # Basic info
        first_name = random.choice(self.FIRST_NAMES)
        last_name = random.choice(self.LAST_NAMES)
        customer_id = self.generate_customer_id()
        
        # Personality
        traits, background = random.choice(self.PERSONALITY_COMBINATIONS)
        
        # Issues
        issues = self.generate_issues(is_multi_domain=force_multi_domain)
        
        # Communication style based on traits
        if PersonalityTrait.BRIEF in traits:
            communication_style = "Short, direct questions. Gets impatient with long explanations."
        elif PersonalityTrait.DETAILED in traits:
            communication_style = "Asks detailed follow-up questions. Wants complete understanding."
        elif PersonalityTrait.TECHNICAL in traits:
            communication_style = "Uses technical terms. Wants specific technical details."
        else:
            communication_style = "Conversational, asks clarifying questions when confused."
        
        # Set patience and satisfaction thresholds based on traits
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


class UserSimulator:
    """LLM-powered user simulator that maintains conversation context."""
    
    def __init__(self, llm_base_url: str = "http://0.0.0.0:4000", 
                 api_key: str = "sk-1234567890", model: str = "gpt-5"):
        self.client = openai.OpenAI(base_url=llm_base_url, api_key=api_key)
        self.model = model
    
    def generate_initial_query(self, persona: Persona, state: ConversationState) -> str:
        """Generate the first message from the user."""
        first_issue = persona.issues[0]
        
        system_prompt = f"""You are {persona.name}, a customer typing in a chat support window.

You need help with: {first_issue.description}

IMPORTANT RULES FOR CHAT:
- Write like you're typing on your phone (15-30 words MAX)
- Be informal and direct
- NO long explanations or multiple questions
- NO formal language or detailed requirements
- Use contractions (don't, can't, won't)

GOOD examples (realistic chat):
- "hey my autopay isn't working, can you check?"
- "I need to change my payment method"
- "why is my bill so high this month?"
- "my data is really slow today, what's going on?"
- "can't make calls but texts work fine"

BAD examples (unrealistic):
- "I'd like to review my autopay configuration including payment method, schedule, and failure notifications"
- "Could you please provide comprehensive details about my billing history and usage patterns?"

Personality: {'frustrated and brief' if PersonalityTrait.FRUSTRATED in persona.personality_traits else 'casual'}"""

        user_prompt = "Type your first message (remember: 15-30 words, casual chat style):"
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.8,
            max_tokens=50  # Reduced from 200
        )
        
        return response.choices[0].message.content.strip()
    
    def generate_response(self, persona: Persona, state: ConversationState, 
                         agent_response: str) -> Tuple[str, bool, str]:
        """Generate user's response based on agent's message.
        
        Returns:
            Tuple of (response_text, should_end_conversation, end_reason)
        """
        # Analyze what issues have been addressed
        addressed_issues = self._analyze_addressed_issues(persona, state, agent_response)
        
        # Update satisfaction based on response quality
        state.satisfaction_score = self._calculate_satisfaction(
            persona, state, agent_response, addressed_issues
        )
        
        # Decide if conversation should end
        should_end, end_reason = self._should_end_conversation(persona, state)
        
        if should_end:
            return self._generate_closing_message(persona, state, end_reason), True, end_reason
        
        # Generate continuation
        return self._generate_continuation(persona, state, agent_response, addressed_issues), False, ""
    
    def _get_response_constraints(self, persona: Persona) -> Tuple[int, str]:
        """Get word limit and style guide based on personality.
        
        Returns:
            Tuple of (max_words, style_description)
        """
        # Much stricter limits for realistic chat
        max_words = 20  # Default for chat
        style = "casual chat"
        
        # Adjust based on personality traits (but keep it SHORT)
        if PersonalityTrait.BRIEF in persona.personality_traits:
            max_words = 10
            style = "very brief"
        elif PersonalityTrait.FRUSTRATED in persona.personality_traits:
            max_words = 15
            style = "frustrated and short"
        elif PersonalityTrait.NON_TECHNICAL in persona.personality_traits:
            max_words = 20
            style = "simple questions"
        elif PersonalityTrait.DETAILED in persona.personality_traits:
            max_words = 25  # Even detailed people type short in chat
            style = "specific question"
        elif PersonalityTrait.PATIENT in persona.personality_traits:
            max_words = 20
            style = "polite but brief"
        
        return max_words, style
    
    def _analyze_addressed_issues(self, persona: Persona, state: ConversationState, 
                                  agent_response: str) -> List[str]:
        """Determine which issues the agent addressed."""
        addressed = []
        
        system_prompt = """Analyze which customer issues were addressed in the agent's response.
List the issue descriptions that were meaningfully addressed (not just acknowledged).

Issues to check:
"""
        for issue in persona.issues:
            if issue.status in [IssueStatus.RAISED, IssueStatus.BEING_ADDRESSED]:
                system_prompt += f"- {issue.description}\n"
        
        system_prompt += "\nReturn a JSON list of addressed issue descriptions."
        
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
            addressed = json.loads(response.choices[0].message.content)
        except:
            addressed = []
        
        # Update issue statuses
        for issue in persona.issues:
            if issue.description in addressed:
                issue.status = IssueStatus.RESOLVED
                state.issues_resolved.append(issue.description)
        
        return addressed
    
    def _calculate_satisfaction(self, persona: Persona, state: ConversationState,
                               agent_response: str, addressed_issues: List[str]) -> float:
        """Calculate current satisfaction score."""
        score = state.satisfaction_score
        
        # Increase for addressed issues (less aggressive increase)
        if addressed_issues:
            score += 0.15 * len(addressed_issues)  # Reduced from 0.2
        
        # Decrease if no progress after multiple turns
        if state.turn_number > 3 and not addressed_issues:
            score -= 0.1
        
        # Personality adjustments
        if PersonalityTrait.PATIENT in persona.personality_traits:
            score += 0.02  # Reduced from 0.05
        if PersonalityTrait.FRUSTRATED in persona.personality_traits:
            score -= 0.08  # Increased from 0.05
        
        return max(0.0, min(1.0, score))
    
    def _should_end_conversation(self, persona: Persona, 
                                state: ConversationState) -> Tuple[bool, str]:
        """Determine if the conversation should end."""
        # All issues resolved
        resolved_count = sum(1 for i in persona.issues if i.status == IssueStatus.RESOLVED)
        if resolved_count == len(persona.issues):
            return True, "all_issues_resolved"
        
        # Satisfaction threshold met with MOST issues resolved
        if state.satisfaction_score >= persona.satisfaction_threshold and resolved_count >= len(persona.issues) - 1:
            return True, "satisfied"
        
        # Too many turns without progress
        if state.turn_number > 8 and resolved_count == 0:
            return True, "no_progress"
        
        # Very frustrated
        if persona.frustration_level > 0.9:
            return True, "too_frustrated"
        
        # Natural conversation limit
        if state.turn_number > 12:
            return True, "max_turns"
        
        return False, ""
    
    def _generate_closing_message(self, persona: Persona, state: ConversationState, 
                                 reason: str) -> str:
        """Generate a closing message based on the reason."""
        if reason == "all_issues_resolved":
            tone = "satisfied"
            examples = ["thanks that worked", "ok great thanks", "perfect thanks", "all good now"]
        elif reason == "satisfied":
            tone = "content"
            examples = ["ok thanks", "that helps", "got it thanks", "sounds good"]
        elif reason == "no_progress":
            tone = "frustrated"
            examples = ["this isn't working", "forget it", "nevermind", "i'll call instead"]
        elif reason == "too_frustrated":
            tone = "very frustrated"
            examples = ["this is useless", "forget it", "waste of time", "bye"]
        else:
            tone = "neutral"
            examples = ["gotta go", "ok bye", "i'll check later", "thanks anyway"]
        
        system_prompt = f"""Type a SHORT chat closing message.

Tone: {tone}
Good examples: {examples}

RULES:
- Maximum 10 words
- Type like you're in chat (lowercase ok)
- Be natural and brief
- NO formal language"""
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": "Type your closing message (max 10 words):"}
            ],
            temperature=0.7,
            max_tokens=30
        )
        
        return response.choices[0].message.content.strip()
    
    def _generate_continuation(self, persona: Persona, state: ConversationState,
                             agent_response: str, addressed_issues: List[str]) -> str:
        """Generate a continuation message."""
        # Get response constraints based on personality
        max_words, style_description = self._get_response_constraints(persona)
        
        # Determine next action
        unraised_issues = [i for i in persona.issues if i.status == IssueStatus.NOT_RAISED]
        unresolved_issues = [i for i in persona.issues 
                           if i.status in [IssueStatus.RAISED, IssueStatus.BEING_ADDRESSED]]
        
        # More nuanced decision logic
        if addressed_issues and unraised_issues:
            # Higher chance to continue with more issues
            if random.random() < 0.85:  # Increased from 0.7
                next_issue = unraised_issues[0]
                next_issue.status = IssueStatus.RAISED
                state.issues_raised.append(next_issue.description)
                action = f"acknowledge the help and raise a new issue: {next_issue.description}"
            else:
                action = "ask a clarifying question about the resolved issue"
        elif unresolved_issues and random.random() < 0.7:  # Increased from 0.6
            # Follow up on unresolved issue
            unresolved_issues[0].status = IssueStatus.BEING_ADDRESSED
            action = f"follow up on: {unresolved_issues[0].description}"
        elif not addressed_issues and state.turn_number > 2:
            # Express frustration at lack of progress
            persona.frustration_level += 0.15  # Reduced from 0.2
            action = "express frustration that your issue isn't being resolved"
        else:
            # Clarify or provide more information
            action = "provide clarification or ask a follow-up question"
        
        # Build few-shot examples
        few_shot_examples = """
GOOD chat responses (how real people type):

Example 1:
Agent: "I've activated your eSIM. You should see signal bars now."
Good: "still no service"
Good: "should i restart?"

Example 2:
Agent: "Let me check your account for compatible plans."
Good: "how long will this take"
Good: "just need to know if unlimited works with iphone 15"

Example 3:
Agent: "Your new device is compatible with our 5G network."
Good: "does it work everywhere?"
Good: "what about rural areas"

Example 4:
Agent: "You'll need to update your APN settings."
Good: "what's that?"
Good: "can you help me do it"

BAD responses (TOO LONG/FORMAL):
- "I'd like comprehensive details about all available plans including pricing tiers and feature comparisons"
- "Please enumerate the technical specifications and compatibility requirements"
- "Could you provide a detailed breakdown of charges with historical usage patterns"
- Any message over 30 words
- Multiple questions in one message
- Formal business language
"""
        
        system_prompt = f"""You are {persona.name} typing in a chat window.

{few_shot_examples}

ABSOLUTE RULES:
- You're typing on your phone/computer in a CHAT WINDOW
- Maximum {max_words} words (usually aim for 5-15 words)
- Type like a normal person in chat:
  * lowercase is fine
  * short and direct
  * one thought at a time
  * no formal language
  * use "i", "my", "me"
  * contractions always (don't, can't, won't)

Current mood: {'frustrated' if persona.frustration_level > 0.5 else 'normal'}
Action: {action}

Type your message (5-15 words ideal, NEVER over {max_words}):"""
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Agent said: {agent_response[:500]}"}
            ],
            temperature=0.8,
            max_tokens=50  # Much stricter for chat
        )
        
        generated_response = response.choices[0].message.content.strip()
        
        # Validate response length
        word_count = len(generated_response.split())
        if word_count > max_words * 1.2:  # Allow 20% buffer
            # Response is too long, regenerate with stricter constraints
            return self._regenerate_with_stricter_constraints(
                persona, state, agent_response, addressed_issues, action, max_words
            )
        
        return generated_response
    
    def _regenerate_with_stricter_constraints(self, persona: Persona, state: ConversationState,
                                             agent_response: str, addressed_issues: List[str],
                                             action: str, max_words: int) -> str:
        """Regenerate response with even stricter length constraints."""
        system_prompt = f"""CHAT MESSAGE - MAX {max_words} WORDS!

Examples of realistic chat:
- "that didn't work"
- "still having issues"
- "how much?"
- "can you fix it"
- "when will it work"
- "ok thanks"

Action: {action}

Type a SHORT chat message ({max_words} words MAX):"""
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Keep it under {max_words} words. Agent said: {agent_response[:200]}"}
            ],
            temperature=0.7,
            max_tokens=30  # Very strict for chat
        )
        
        return response.choices[0].message.content.strip()


class AgentClient:
    """Simple client for the agent API, matching test_chat.py pattern."""
    
    def __init__(self, base_url: str = "http://127.0.0.1:8010"):
        self.base_url = base_url
        self.api_url = f"{base_url}/chat"
    
    def health_check(self) -> bool:
        """Check if the agent is running."""
        try:
            # Try the main endpoint with minimal data
            response = requests.post(
                self.api_url,
                json={"messages": [{"role": "user", "content": "test"}], "customer_id": "test", "session_id": None},
                timeout=5
            )
            return response.status_code == 200
        except Exception as e:
            print(f"Health check failed: {e}")
            return False
    
    def send_message(self, message: str, messages: List[Dict[str, Any]], 
                     customer_id: str, session_id: Optional[str] = None) -> Tuple[str, List[Dict[str, Any]], str]:
        """Send a message to the agent and get response.
        
        Returns:
            Tuple of (agent_response, updated_messages, session_id)
        """
        # Add the user message to the messages list
        messages_with_user = messages + [{"role": "user", "content": message}]
        
        payload = {
            "messages": messages_with_user,
            "customer_id": customer_id
        }
        
        # Include session_id if provided
        if session_id:
            payload["session_id"] = session_id
        
        try:
            response = requests.post(self.api_url, json=payload)
            response.raise_for_status()
            
            data = response.json()
            updated_messages = data["messages"]
            returned_session_id = data.get("session_id", session_id)
            
            # Extract the last assistant message
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
    
    def __init__(self, user_simulator: UserSimulator, agent_client: AgentClient,
                 persona_generator: PersonaGenerator, verbose: bool = False):
        self.user_simulator = user_simulator
        self.agent_client = agent_client
        self.persona_generator = persona_generator
        self.verbose = verbose
    
    def run_conversation(self, persona: Optional[Persona] = None, force_multi_domain: bool = False) -> ConversationState:
        """Run a complete multi-turn conversation."""
        # Generate persona if not provided
        if persona is None:
            persona = self.persona_generator.generate_persona(force_multi_domain=force_multi_domain)
        
        # Initialize conversation state
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
            
            # Main conversation loop
            current_message = initial_query
            
            while not state.should_end and state.turn_number < 15:
                state.turn_number += 1
                
                # Get agent response
                try:
                    agent_response, updated_messages, session_id = self.agent_client.send_message(
                        current_message,
                        state.messages,
                        persona.customer_id,
                        state.session_id
                    )
                    
                    # Store session_id (will be set on first call)
                    if not state.session_id and session_id:
                        state.session_id = session_id
                        if self.verbose:
                            print(f"📝 Session ID: {session_id}")
                    
                    print(f"🤖 Agent: {agent_response}\n")
                    
                    # Update messages with the new messages from the agent
                    state.messages = updated_messages
                    state.agent_responses.append(agent_response)
                    
                    # Generate user response
                    user_response, should_end, end_reason = self.user_simulator.generate_response(
                        persona, state, agent_response
                    )
                    
                    if should_end:
                        print(f"👤 User: {user_response}\n")
                        # Add final user message to messages
                        state.messages.append({"role": "user", "content": user_response})
                        state.should_end = True
                        state.end_reason = end_reason
                        break
                    
                    print(f"👤 User: {user_response}\n")
                    current_message = user_response
                    
                    # Small delay to avoid overwhelming the agent
                    time.sleep(0.5)
                    
                except Exception as e:
                    print(f"❌ Error in conversation: {e}")
                    state.should_end = True
                    state.end_reason = "error"
                    break
            
            # Set end time
            state.end_time = datetime.now()
            
            # Print summary
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
    """Main function to run the multi-turn simulator."""
    parser = argparse.ArgumentParser(description='Multi-turn Conversational Simulator')
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
    
    args = parser.parse_args()
    
    # Initialize components
    print("Initializing multi-turn conversation simulator...")
    
    persona_generator = PersonaGenerator()
    user_simulator = UserSimulator(
        llm_base_url=args.llm_url,
        model=args.llm_model
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
            time.sleep(2)  # Brief pause between conversations
    
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
        
        with open(args.export_json, 'w') as f:
            json.dump(export_data, f, indent=2, default=str)
        print(f"\n📄 Conversations exported to {args.export_json}")
        
    print("\n✅ Simulation complete!")
    return 0


if __name__ == "__main__":
    exit(main())
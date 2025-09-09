#!/usr/bin/env python3
"""
Parallel Multi-turn Conversational Simulator for Telco Support Agent

This version runs multiple conversations concurrently for massive speed improvements.
Implements Phase 1 (Async Infrastructure) and Phase 2 (Parallel Conversation Management).
"""

import argparse
import asyncio
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
from concurrent.futures import ThreadPoolExecutor

import openai
import aiohttp
from dotenv import load_dotenv
from tqdm.asyncio import tqdm

# Add current directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

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
    satisfaction_threshold: float = 0.7
    frustration_level: float = 0.0
    patience_level: float = 0.8


@dataclass
class ConversationState:
    """Tracks the state of an ongoing conversation."""
    conversation_id: str
    persona: Persona
    session_id: Optional[str] = None
    turn_number: int = 0
    messages: List[Dict[str, Any]] = field(default_factory=list)
    issues_raised: List[str] = field(default_factory=list)
    issues_resolved: List[str] = field(default_factory=list)
    current_issue_index: int = 0
    satisfaction_score: float = 0.5
    should_end: bool = False
    end_reason: str = ""
    agent_responses: List[str] = field(default_factory=list)
    start_time: datetime = field(default_factory=datetime.now)
    end_time: Optional[datetime] = None


class ResponseCache:
    """Thread-safe cache for responses."""
    
    def __init__(self, ttl_seconds: int = 3600):
        self.cache: Dict[str, Tuple[str, float]] = {}
        self.ttl = ttl_seconds
        self.hit_count = 0
        self.miss_count = 0
        self._lock = asyncio.Lock()
    
    def _make_key(self, *args) -> str:
        """Create a cache key from arguments."""
        key_str = json.dumps(args, sort_keys=True, default=str)
        return hashlib.md5(key_str.encode()).hexdigest()
    
    async def get(self, *args) -> Optional[str]:
        """Get cached response if available and not expired."""
        async with self._lock:
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
    
    async def set(self, response: str, *args):
        """Cache a response."""
        async with self._lock:
            key = self._make_key(*args)
            self.cache[key] = (response, time.time())


class ResponseTemplates:
    """Template-based responses for common scenarios."""
    
    CLOSING_TEMPLATES = {
        ("all_issues_resolved", "patient"): ["thanks that worked", "perfect thanks", "all good now"],
        ("all_issues_resolved", "frustrated"): ["finally", "ok thanks", "about time"],
        ("all_issues_resolved", "brief"): ["thanks", "ok", "done"],
        ("satisfied", "patient"): ["ok thanks", "that helps", "sounds good"],
        ("satisfied", "frustrated"): ["fine", "ok", "whatever"],
        ("satisfied", "brief"): ["ok", "thx", "k"],
        ("no_progress", "patient"): ["i'll try later", "thanks anyway", "i'll call instead"],
        ("no_progress", "frustrated"): ["this isn't working", "forget it", "waste of time"],
        ("too_frustrated", "any"): ["forget it", "bye", "done"],
        ("max_turns", "any"): ["gotta go", "bye", "thanks"],
    }
    
    FOLLOWUP_TEMPLATES = {
        "clarification_needed": ["what's that?", "how?", "can you explain?"],
        "still_broken": ["still not working", "didn't help", "same problem"],
        "acknowledgment": ["ok", "i see", "got it"],
        "frustrated_no_progress": ["this is taking forever", "come on", "seriously?"],
    }
    
    @classmethod
    def get_closing(cls, end_reason: str, traits: List[PersonalityTrait]) -> str:
        """Get a closing message based on reason and personality."""
        if PersonalityTrait.FRUSTRATED in traits:
            trait_key = "frustrated"
        elif PersonalityTrait.BRIEF in traits:
            trait_key = "brief"
        elif PersonalityTrait.PATIENT in traits:
            trait_key = "patient"
        else:
            trait_key = "patient"
        
        templates = cls.CLOSING_TEMPLATES.get((end_reason, trait_key))
        if not templates:
            templates = cls.CLOSING_TEMPLATES.get((end_reason, "any"), ["bye"])
        
        return random.choice(templates)


class PersonaGenerator:
    """Generates realistic customer personas with multiple issues."""
    
    ISSUE_TEMPLATES = {
        "account": [
            "customer wants to know their current plan details",
            "customer needs to verify their subscription status",
            "customer is asking about autopay settings",
        ],
        "billing": [
            "customer sees unexpected charges on their bill",
            "customer wants to know when payment is due",
            "customer needs breakdown of current month charges",
        ],
        "tech_support": [
            "customer's phone won't connect to network",
            "customer has slow data speeds",
            "customer can't receive calls but can make them",
        ],
        "product": [
            "customer wants to compare available plans",
            "customer is looking for device upgrade options",
            "customer needs information about current promotions",
        ],
    }
    
    PERSONALITY_COMBINATIONS = [
        ([PersonalityTrait.PATIENT, PersonalityTrait.TECHNICAL], 
         "Tech-savvy customer who understands technical details and is patient"),
        ([PersonalityTrait.FRUSTRATED, PersonalityTrait.BRIEF], 
         "Frustrated customer who wants quick answers"),
        ([PersonalityTrait.CONFUSED, PersonalityTrait.NON_TECHNICAL], 
         "Non-technical customer who needs simple explanations"),
    ]
    
    FIRST_NAMES = ["Sarah", "Michael", "Jessica", "David", "Emily", "Robert"]
    LAST_NAMES = ["Chen", "Rodriguez", "Williams", "Park", "Johnson", "Taylor"]
    
    def __init__(self, customer_id_start: int = 10001):
        self.customer_id_start = customer_id_start
        self.counter = 0
    
    def generate_persona(self) -> Persona:
        """Generate a complete customer persona."""
        self.counter += 1
        customer_id = f"CUS-{self.customer_id_start + self.counter:05d}"
        
        first_name = random.choice(self.FIRST_NAMES)
        last_name = random.choice(self.LAST_NAMES)
        
        traits, background = random.choice(self.PERSONALITY_COMBINATIONS)
        
        # Generate 1-2 issues
        num_issues = random.choice([1, 2])
        categories = random.sample(list(self.ISSUE_TEMPLATES.keys()), num_issues)
        issues = []
        
        for i, category in enumerate(categories):
            issue_desc = random.choice(self.ISSUE_TEMPLATES[category])
            issues.append(Issue(
                category=category,
                description=issue_desc,
                priority=i + 1,
                requires_followup=random.random() < 0.3
            ))
        
        if PersonalityTrait.BRIEF in traits:
            communication_style = "Short, direct questions."
        elif PersonalityTrait.TECHNICAL in traits:
            communication_style = "Uses technical terms."
        else:
            communication_style = "Conversational."
        
        patience = 0.8 if PersonalityTrait.PATIENT in traits else 0.4
        
        return Persona(
            customer_id=customer_id,
            name=f"{first_name} {last_name}",
            personality_traits=traits,
            issues=issues,
            background=background,
            communication_style=communication_style,
            patience_level=patience
        )


class AsyncUserSimulator:
    """Async LLM-powered user simulator."""
    
    def __init__(self, llm_base_url: str = "http://0.0.0.0:4000", 
                 api_key: str = "sk-1234567890", model: str = "gpt-5-nano"):
        self.client = openai.AsyncOpenAI(base_url=llm_base_url, api_key=api_key)
        self.model = model
        self.cache = ResponseCache()
        self.llm_calls = 0
        self.template_uses = 0
    
    async def generate_initial_query(self, persona: Persona, state: ConversationState) -> str:
        """Generate the first message from the user."""
        first_issue = persona.issues[0]
        
        # Try cache first
        cached = await self.cache.get("initial", first_issue.category, 
                                      first_issue.description, persona.personality_traits)
        if cached:
            return cached
        
        # Simple template for speed
        if random.random() < 0.5:
            self.template_uses += 1
            templates = {
                "billing": "why is my bill so high?",
                "account": "can't login to my account",
                "tech_support": "my phone isn't working",
                "product": "what plans do you have?",
            }
            return templates.get(first_issue.category, "need help")
        
        # LLM generation
        system_prompt = f"""You are {persona.name}, a customer typing in chat.
Issue: {first_issue.description}
Write ONE short message (max 20 words). Be casual and direct."""
        
        self.llm_calls += 1
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": "Type your message:"}
            ],
            temperature=0.8,
            max_tokens=50
        )
        
        result = response.choices[0].message.content.strip()
        await self.cache.set(result, "initial", first_issue.category, 
                           first_issue.description, persona.personality_traits)
        return result
    
    async def generate_response(self, persona: Persona, state: ConversationState, 
                               agent_response: str) -> Tuple[str, bool, str]:
        """Generate user's response based on agent's message."""
        # Simple rule-based analysis for speed
        agent_lower = agent_response.lower()
        issue_resolved = any(word in agent_lower for word in 
                            ["fixed", "resolved", "completed", "activated", "updated"])
        
        if issue_resolved:
            for issue in persona.issues:
                if issue.status in [IssueStatus.RAISED, IssueStatus.BEING_ADDRESSED]:
                    issue.status = IssueStatus.RESOLVED
                    state.issues_resolved.append(issue.description)
        
        # Update satisfaction
        if issue_resolved:
            state.satisfaction_score = min(1.0, state.satisfaction_score + 0.3)
        elif state.turn_number > 3:
            state.satisfaction_score = max(0.0, state.satisfaction_score - 0.1)
        
        # Check if should end
        resolved_count = sum(1 for i in persona.issues if i.status == IssueStatus.RESOLVED)
        
        if resolved_count == len(persona.issues):
            self.template_uses += 1
            closing = ResponseTemplates.get_closing("all_issues_resolved", persona.personality_traits)
            return closing, True, "all_issues_resolved"
        
        if state.turn_number > 8:
            self.template_uses += 1
            closing = ResponseTemplates.get_closing("max_turns", persona.personality_traits)
            return closing, True, "max_turns"
        
        # Generate continuation
        if random.random() < 0.4:  # 40% template usage
            self.template_uses += 1
            return random.choice(ResponseTemplates.FOLLOWUP_TEMPLATES["acknowledgment"]), False, ""
        
        # Quick LLM response
        self.llm_calls += 1
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": f"You are {persona.name}. Respond in max 15 words."},
                {"role": "user", "content": f"Agent said: {agent_response[:200]}"}
            ],
            temperature=0.8,
            max_tokens=30
        )
        
        return response.choices[0].message.content.strip(), False, ""


class AsyncAgentClient:
    """Async client for the agent API with connection pooling."""
    
    def __init__(self, base_url: str = "http://127.0.0.1:8010", 
                 max_connections: int = 20):
        self.base_url = base_url
        self.api_url = f"{base_url}/chat"
        self.connector = None
        self.session = None
        self.max_connections = max_connections
    
    async def __aenter__(self):
        """Initialize the session with connection pooling."""
        self.connector = aiohttp.TCPConnector(
            limit=self.max_connections,
            limit_per_host=self.max_connections,
            keepalive_timeout=30
        )
        self.session = aiohttp.ClientSession(connector=self.connector)
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Clean up the session."""
        if self.session:
            await self.session.close()
        if self.connector:
            await self.connector.close()
    
    async def health_check(self) -> bool:
        """Check if the agent is running."""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(self.base_url, timeout=2) as response:
                    return True
        except:
            try:
                async with aiohttp.ClientSession() as session:
                    payload = {"messages": [{"role": "user", "content": "hi"}], "customer_id": "test"}
                    async with session.post(self.api_url, json=payload, timeout=5) as response:
                        return response.status == 200
            except:
                return False
    
    async def send_message(self, message: str, messages: List[Dict[str, Any]], 
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
            async with self.session.post(self.api_url, json=payload, timeout=30) as response:
                response.raise_for_status()
                data = await response.json()
                
                updated_messages = data["messages"]
                returned_session_id = data.get("session_id", session_id)
                
                agent_response = None
                for msg in reversed(updated_messages):
                    if msg.get("role") == "assistant":
                        agent_response = msg.get("content")
                        break
                
                if not agent_response:
                    raise Exception("No assistant response found")
                
                return agent_response, updated_messages, returned_session_id
                
        except asyncio.TimeoutError:
            raise Exception("Request timed out")
        except Exception as e:
            raise Exception(f"Error calling API: {e}")


class ParallelConversationOrchestrator:
    """Orchestrates multiple conversations in parallel."""
    
    def __init__(self, user_simulator: AsyncUserSimulator, 
                 agent_client: AsyncAgentClient,
                 persona_generator: PersonaGenerator,
                 max_concurrent: int = 5,
                 verbose: bool = False):
        self.user_simulator = user_simulator
        self.agent_client = agent_client
        self.persona_generator = persona_generator
        self.max_concurrent = max_concurrent
        self.verbose = verbose
        self.api_semaphore = asyncio.Semaphore(max_concurrent)
        self.completed_conversations = []
        self.failed_conversations = []
    
    async def run_single_conversation(self, persona: Optional[Persona] = None, 
                                     conv_index: int = 0) -> ConversationState:
        """Run a single conversation asynchronously."""
        if persona is None:
            persona = self.persona_generator.generate_persona()
        
        state = ConversationState(
            conversation_id=str(uuid4()),
            persona=persona
        )
        
        if self.verbose:
            print(f"\n[Conv {conv_index}] Starting {persona.name} ({persona.customer_id})")
        
        try:
            # Generate initial query
            initial_query = await self.user_simulator.generate_initial_query(persona, state)
            persona.issues[0].status = IssueStatus.RAISED
            state.issues_raised.append(persona.issues[0].description)
            
            current_message = initial_query
            
            while not state.should_end and state.turn_number < 10:
                state.turn_number += 1
                
                # Use semaphore to limit concurrent API calls
                async with self.api_semaphore:
                    agent_response, updated_messages, session_id = await self.agent_client.send_message(
                        current_message,
                        state.messages,
                        persona.customer_id,
                        state.session_id
                    )
                
                if not state.session_id and session_id:
                    state.session_id = session_id
                
                state.messages = updated_messages
                state.agent_responses.append(agent_response)
                
                user_response, should_end, end_reason = await self.user_simulator.generate_response(
                    persona, state, agent_response
                )
                
                if should_end:
                    state.messages.append({"role": "user", "content": user_response})
                    state.should_end = True
                    state.end_reason = end_reason
                    break
                
                current_message = user_response
                
                # Small async delay to avoid overwhelming
                await asyncio.sleep(0.1)
            
            state.end_time = datetime.now()
            
            if self.verbose:
                duration = (state.end_time - state.start_time).total_seconds()
                resolved = sum(1 for i in persona.issues if i.status == IssueStatus.RESOLVED)
                print(f"[Conv {conv_index}] Completed: {state.turn_number} turns, "
                      f"{resolved}/{len(persona.issues)} resolved, {duration:.1f}s")
            
            return state
            
        except Exception as e:
            if self.verbose:
                print(f"[Conv {conv_index}] Failed: {e}")
            state.should_end = True
            state.end_reason = "error"
            state.end_time = datetime.now()
            return state
    
    async def run_parallel_conversations(self, num_conversations: int, 
                                        show_progress: bool = True) -> List[ConversationState]:
        """Run multiple conversations in parallel."""
        print(f"\n{'='*60}")
        print(f"PARALLEL CONVERSATION EXECUTION")
        print(f"{'='*60}")
        print(f"Running {num_conversations} conversations")
        print(f"Max concurrent: {self.max_concurrent}")
        print(f"{'='*60}\n")
        
        start_time = time.time()
        
        # Create conversation tasks
        tasks = []
        for i in range(num_conversations):
            persona = self.persona_generator.generate_persona()
            task = self.run_single_conversation(persona, i)
            tasks.append(task)
        
        # Run with progress bar if requested
        if show_progress:
            results = []
            for task in tqdm.as_completed(tasks, desc="Conversations", unit="conv"):
                result = await task
                results.append(result)
                if result.end_reason != "error":
                    self.completed_conversations.append(result)
                else:
                    self.failed_conversations.append(result)
        else:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            for result in results:
                if isinstance(result, ConversationState):
                    if result.end_reason != "error":
                        self.completed_conversations.append(result)
                    else:
                        self.failed_conversations.append(result)
        
        end_time = time.time()
        total_duration = end_time - start_time
        
        self._print_summary(total_duration)
        
        return self.completed_conversations
    
    def _print_summary(self, total_duration: float):
        """Print execution summary."""
        print(f"\n{'='*60}")
        print(f"EXECUTION SUMMARY")
        print(f"{'='*60}")
        
        total_convs = len(self.completed_conversations) + len(self.failed_conversations)
        print(f"Total conversations: {total_convs}")
        print(f"Successful: {len(self.completed_conversations)}")
        print(f"Failed: {len(self.failed_conversations)}")
        print(f"Total time: {total_duration:.1f} seconds")
        print(f"Average time per conversation: {total_duration/total_convs:.1f} seconds")
        
        if self.completed_conversations:
            avg_turns = sum(s.turn_number for s in self.completed_conversations) / len(self.completed_conversations)
            total_issues = sum(len(s.persona.issues) for s in self.completed_conversations)
            resolved_issues = sum(
                sum(1 for i in s.persona.issues if i.status == IssueStatus.RESOLVED)
                for s in self.completed_conversations
            )
            avg_satisfaction = sum(s.satisfaction_score for s in self.completed_conversations) / len(self.completed_conversations)
            
            print(f"\nConversation metrics:")
            print(f"  Average turns: {avg_turns:.1f}")
            print(f"  Resolution rate: {resolved_issues}/{total_issues} ({resolved_issues/total_issues*100:.1f}%)")
            print(f"  Average satisfaction: {avg_satisfaction:.2f}")
        
        print(f"\nSimulator performance:")
        print(f"  LLM calls: {self.user_simulator.llm_calls}")
        print(f"  Template uses: {self.user_simulator.template_uses}")
        total_responses = self.user_simulator.llm_calls + self.user_simulator.template_uses
        if total_responses > 0:
            opt_rate = self.user_simulator.template_uses / total_responses * 100
            print(f"  Optimization rate: {opt_rate:.1f}%")
        
        print(f"\n⚡ Parallel speedup: {total_convs/total_duration:.1f} conv/sec")
        print(f"{'='*60}\n")


async def main():
    """Main async function."""
    parser = argparse.ArgumentParser(description='Parallel Multi-turn Simulator')
    parser.add_argument('--num-conversations', type=int, default=10,
                       help='Number of conversations to simulate')
    parser.add_argument('--max-concurrent', type=int, default=5,
                       help='Maximum concurrent conversations')
    parser.add_argument('--agent-url', default='http://127.0.0.1:8010',
                       help='Agent API URL')
    parser.add_argument('--llm-url', default='http://0.0.0.0:4000',
                       help='LLM API URL')
    parser.add_argument('--llm-model', default='gpt-5',
                       help='LLM model name')
    parser.add_argument('--verbose', action='store_true',
                       help='Show detailed logs')
    parser.add_argument('--no-progress', action='store_true',
                       help='Disable progress bar')
    
    args = parser.parse_args()
    
    print("Initializing parallel conversation simulator...")
    
    # Initialize components
    persona_generator = PersonaGenerator()
    user_simulator = AsyncUserSimulator(
        llm_base_url=args.llm_url,
        model=args.llm_model
    )
    
    async with AsyncAgentClient(
        base_url=args.agent_url,
        max_connections=args.max_concurrent * 2
    ) as agent_client:
        
        # Check agent health
        print(f"Checking agent at {args.agent_url}...")
        if not await agent_client.health_check():
            print("❌ Agent is not running! Please start it with:")
            print("   uvicorn load_test_agent.main:app --host 0.0.0.0 --port 8010")
            return 1
        
        print("✅ Agent is healthy\n")
        
        # Create orchestrator
        orchestrator = ParallelConversationOrchestrator(
            user_simulator=user_simulator,
            agent_client=agent_client,
            persona_generator=persona_generator,
            max_concurrent=args.max_concurrent,
            verbose=args.verbose
        )
        
        # Run parallel conversations
        await orchestrator.run_parallel_conversations(
            num_conversations=args.num_conversations,
            show_progress=not args.no_progress
        )
    
    return 0


if __name__ == "__main__":
    exit(asyncio.run(main()))
import os
import json
import uuid
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
from openai import OpenAI
from dotenv import load_dotenv
from . import tools
from .utils import function_to_schema

# Load environment variables
load_dotenv()

import logging
import mlflow

mlflow.openai.autolog()


class Agent:
    """
    An agent that can have a conversation and use tools.
    """
    def __init__(self):
        # Get environment variables
        api_key = os.getenv("OPENAI_API_KEY")
        base_url = os.getenv("OPENAI_API_BASE")
        self.model_name = os.getenv("MODEL_NAME")

        if not api_key or not self.model_name:
            raise ValueError("MODEL_NAME, OPENAI_API_KEY environment variable not set.")

        # Initialize the OpenAI client
        self.client = OpenAI(api_key=api_key, base_url=base_url)


        # The list of available tools
        self.available_tools = {
            "get_billing_history": tools.get_billing_history,
            "get_charge_details": tools.get_charge_details,
            "get_active_promotions": tools.get_active_promotions,
            "get_account_balance": tools.get_account_balance,
            "get_saved_payment_methods": tools.get_saved_payment_methods,
            "process_payment": tools.process_payment,
            "get_network_status": tools.get_network_status,
            "run_remote_device_diagnostics": tools.run_remote_device_diagnostics,
            "create_support_ticket": tools.create_support_ticket,
            "search_knowledge_base": tools.search_knowledge_base,
            "get_insurance_coverage": tools.get_insurance_coverage,
            "process_insurance_claim": tools.process_insurance_claim,
            "get_data_usage": tools.get_data_usage,
            "escalate_to_human_agent": tools.escalate_to_human_agent,
        }

        # The tool definitions for the OpenAI API - generated from function signatures
        self.tool_definitions = [
            function_to_schema(tools.get_billing_history),
            function_to_schema(tools.get_charge_details),
            function_to_schema(tools.get_active_promotions),
            function_to_schema(tools.get_account_balance),
            function_to_schema(tools.get_saved_payment_methods),
            function_to_schema(tools.process_payment),
            function_to_schema(tools.get_network_status),
            function_to_schema(tools.run_remote_device_diagnostics),
            function_to_schema(tools.create_support_ticket),
            function_to_schema(tools.search_knowledge_base),
            function_to_schema(tools.get_insurance_coverage),
            function_to_schema(tools.process_insurance_claim),
            function_to_schema(tools.get_data_usage),
            function_to_schema(tools.escalate_to_human_agent),
        ]

    @mlflow.trace
    def get_system_message(self, customer_id: str) -> dict:
        return {"role": "system", "content": f"You are a helpful telco customer support agent. Your customer ID is {customer_id}. Do not ask for it."}

    @mlflow.trace
    def chat(self, messages: list, customer_id: str, session_id: Optional[str] = None) -> dict:
        """
        Runs a single turn of the conversation.
        Returns: dict with messages and session_id
        """
        # Generate session_id if not provided
        if not session_id:
            session_id = str(uuid.uuid4())

        mlflow.update_current_trace(
        metadata={
            "mlflow.trace.user": customer_id,      # Links this trace to a specific user
            "mlflow.trace.session": session_id, # Groups this trace with others in the same conversation
        }
        )
        
        # Always ensure system message is at the beginning
        system_msg = self.get_system_message(customer_id)
        if not messages or messages[0].get("role") != "system":
            messages = [system_msg] + messages
        else:
            # Update the existing system message
            messages[0] = system_msg

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            tools=self.tool_definitions,
            tool_choice="auto",
        )

        max_iterations = 10
        iteration = 0
        
        while iteration < max_iterations:
            if iteration == 0:
                response_message = response.choices[0].message
            else:
                response_message = response.choices[0].message
            
            tool_calls = response_message.tool_calls
            
            if tool_calls:
                messages.append(response_message.model_dump())
                
                for tool_call in tool_calls:
                    function_name = tool_call.function.name
                    function_to_call = self.available_tools[function_name]
                    function_args = json.loads(tool_call.function.arguments)
                    
                    if "customer_id" in function_to_call.__code__.co_varnames and "customer_id" not in function_args:
                        function_args["customer_id"] = customer_id
                    

                    function_response = function_to_call(**function_args)
                    
                    messages.append(
                        {
                            "tool_call_id": tool_call.id,
                            "role": "tool",
                            "name": function_name,
                            "content": function_response,
                        }
                    )
                
                # Make another API call to continue the conversation
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    tools=self.tool_definitions,
                    tool_choice="auto",
                )
                iteration += 1
            else:
                # No more tool calls, we have the final response
                messages.append({"role": "assistant", "content": response_message.content})
                break
        
        # If we hit max iterations, use the last response as final
        if iteration == max_iterations:
            messages.append({"role": "assistant", "content": response.choices[0].message.content})
        
        return {"messages": messages, "session_id": session_id}

app = FastAPI()
agent = Agent()

class ChatRequest(BaseModel):
    messages: List[Dict[str, Any]]
    customer_id: str
    session_id: Optional[str] = None

class ChatResponse(BaseModel):
    messages: List[Dict[str, Any]]
    session_id: str

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Handles a single turn of a conversation.
    """
    result = agent.chat(request.messages, request.customer_id, request.session_id)
    
    return ChatResponse(messages=result["messages"], session_id=result["session_id"])

@app.get("/")
async def root():
    return {"message": "Telco Agent API is running."}
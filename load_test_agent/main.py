import os
import json
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Dict, Any
from openai import OpenAI
from dotenv import load_dotenv
from load_test_agent import tools

# Load environment variables
load_dotenv()

class Agent:
    """
    An agent that can have a conversation and use tools.
    """
    def __init__(self):
        # Get environment variables
        api_key = os.getenv("OPENAI_API_KEY")
        base_url = os.getenv("OPENAI_API_BASE")
        self.model_name = os.getenv("MODEL_NAME", "gpt-4-turbo")

        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set.")

        # Initialize the OpenAI client
        self.client = OpenAI(api_key=api_key, base_url=base_url)

        # A fake customer ID for demonstration purposes
        self.customer_id = "cust_12345"

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

        # The tool definitions for the OpenAI API
        self.tool_definitions = [
            {
                "type": "function",
                "function": {
                    "name": "get_billing_history",
                    "description": "Retrieves the billing history for a given customer.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "customer_id": {"type": "string", "description": "The ID of the customer."}
                        },
                        "required": ["customer_id"],
                    },
                },
            },
            # ... (omitting the rest for brevity in thought process)
        ]

    def get_initial_messages(self) -> list:
        return [
            {"role": "system", "content": f"You are a helpful telco customer support agent. Your customer ID is {self.customer_id}. Do not ask for it."},
        ]

    def chat(self, messages: list, user_input: str) -> tuple[list, str]:
        """
        Runs a single turn of the conversation.
        """
        messages.append({"role": "user", "content": user_input})

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            tools=self.tool_definitions,
            tool_choice="auto",
        )

        response_message = response.choices[0].message
        tool_calls = response_message.tool_calls

        if tool_calls:
            messages.append(response_message.model_dump())

            for tool_call in tool_calls:
                function_name = tool_call.function.name
                function_to_call = self.available_tools[function_name]
                function_args = json.loads(tool_call.function.arguments)
                
                if "customer_id" in function_to_call.__code__.co_varnames and "customer_id" not in function_args:
                    function_args["customer_id"] = self.customer_id

                function_response = function_to_call(**function_args)
                
                messages.append(
                    {
                        "tool_call_id": tool_call.id,
                        "role": "tool",
                        "name": function_name,
                        "content": function_response,
                    }
                )

            second_response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                tools=self.tool_definitions,
                tool_choice="auto",
            )
            final_response = second_response.choices[0].message.content
            messages.append({"role": "assistant", "content": final_response})
        else:
            final_response = response_message.content
            messages.append({"role": "assistant", "content": final_response})
        
        return messages, final_response

app = FastAPI()
agent = Agent()

class ChatRequest(BaseModel):
    history: List[Dict[str, Any]]
    message: str

class ChatResponse(BaseModel):
    history: List[Dict[str, Any]]
    response: str

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Handles a single turn of a conversation.
    """
    messages = request.history
    if not messages:
        messages = agent.get_initial_messages()

    updated_messages, response = agent.chat(messages, request.message)
    
    return ChatResponse(history=updated_messages, response=response)

@app.get("/")
async def root():
    return {"message": "Telco Agent API is running."}
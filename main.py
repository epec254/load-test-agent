import os
import json
import sys
from openai import OpenAI
from dotenv import load_dotenv
import tools

# Load environment variables from .env file
load_dotenv()

# Get environment variables
api_key = os.getenv("OPENAI_API_KEY")
base_url = os.getenv("OPENAI_API_BASE")
model_name = os.getenv("MODEL_NAME", "gpt-4-turbo")

if not api_key:
    raise ValueError("OPENAI_API_KEY environment variable not set.")

# Initialize the OpenAI client
client = OpenAI(api_key=api_key, base_url=base_url)

# A fake customer ID for demonstration purposes
customer_id = "cust_12345"

# The list of available tools
available_tools = {
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
tool_definitions = [
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
    {
        "type": "function",
        "function": {
            "name": "process_payment",
            "description": "Processes a payment for a customer.",
            "parameters": {
                "type": "object",
                "properties": {
                    "customer_id": {"type": "string", "description": "The ID of the customer."},
                    "amount": {"type": "number", "description": "The amount to pay."},
                    "payment_method_id": {"type": "string", "description": "The ID of the payment method to use."}
                },
                "required": ["customer_id", "amount", "payment_method_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_network_status",
            "description": "Retrieves the network status for a given location.",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string", "description": "The location to check (e.g., city, address)."}
                },
                "required": ["location"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "create_support_ticket",
            "description": "Creates a support ticket.",
            "parameters": {
                "type": "object",
                "properties": {
                    "customer_id": {"type": "string", "description": "The ID of the customer."},
                    "issue_description": {"type": "string", "description": "A description of the issue."}
                },
                "required": ["customer_id", "issue_description"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_knowledge_base",
            "description": "Searches the knowledge base for a given query.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "The search query."}
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_data_usage",
            "description": "Retrieves data usage for a customer.",
            "parameters": {
                "type": "object",
                "properties": {
                    "customer_id": {"type": "string", "description": "The ID of the customer."}
                },
                "required": ["customer_id"],
            },
        },
    },
]

def run_turn(messages: list, user_input: str):
    """
    Runs a single turn of the conversation.
    """
    print(f"\n> You: {user_input}")
    messages.append({"role": "user", "content": user_input})

    # First API call to get tool calls
    response = client.chat.completions.create(
        model=model_name,
        messages=messages,
        tools=tool_definitions,
        tool_choice="auto",
    )

    response_message = response.choices[0].message
    tool_calls = response_message.tool_calls

    if tool_calls:
        messages.append(response_message)

        # Execute all tool calls
        for tool_call in tool_calls:
            function_name = tool_call.function.name
            function_to_call = available_tools[function_name]
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

        # Second API call to get the final response
        second_response = client.chat.completions.create(
            model=model_name,
            messages=messages,
            tools=tool_definitions,
            tool_choice="auto",
        )
        final_response = second_response.choices[0].message.content
        print(f"Agent: {final_response}")
        messages.append({"role": "assistant", "content": final_response})
    else:
        final_response = response_message.content
        print(f"Agent: {final_response}")
        messages.append({"role": "assistant", "content": final_response})
    
    return messages

def main():
    """
    Main entry point for the simulated multi-turn conversation.
    """
    if len(sys.argv) < 2:
        print("Usage: python main.py \"<turn 1>\" \"<turn 2>\" ...")
        sys.exit(1)

    conversation_turns = sys.argv[1:]
    
    messages = [
        {"role": "system", "content": f"You are a helpful telco customer support agent. Your customer ID is {customer_id}. Do not ask for it."},
    ]

    print("--- Starting Simulated Conversation ---")
    for turn_input in conversation_turns:
        messages = run_turn(messages, turn_input)
    print("--- Conversation Ended ---")

if __name__ == "__main__":
    main()
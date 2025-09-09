
import requests
import json

API_URL = "http://127.0.0.1:8010/chat"

def run_chat_test():
    """
    Simulates a multi-turn conversation with the agent API.
    """
    messages = []
    customer_id = "cust_12345"
    session_id = None  # Will be set by the agent on first request
    
    turns = [
        "Why is my bill so high?",
        # "What is that $40 charge?",
        # "Ok, please dispute that charge."
    ]

    print("-- Starting Chat Test --")

    for i, turn in enumerate(turns):
        print(f"\n> You: {turn}")
        
        # Add user message to the conversation
        messages.append({"role": "user", "content": turn})
        
        payload = {
            "messages": messages,
            "customer_id": customer_id
        }
        
        # Include session_id if we have one (after first request)
        if session_id:
            payload["session_id"] = session_id
        
        try:
            response = requests.post(API_URL, json=payload)
            print(f"Request URL: {response.request.url}")
            print(f"Request Headers: {response.request.headers}")
            print(f"Request Body: {response.request.body}")
            print(f"Response Status Code: {response.status_code}")
            print(f"Response Headers: {response.headers}")
            print(f"Response Text: {response.text}")
            response.raise_for_status()  # Raise an exception for bad status codes
            
            data = response.json()
            messages = data["messages"]
            
            # Extract and store the session_id
            if "session_id" in data:
                if i == 0:
                    print(f"Session ID created: {data['session_id']}")
                session_id = data["session_id"]
            
            # Extract the last assistant message
            agent_response = None
            for msg in reversed(messages):
                if msg.get("role") == "assistant":
                    agent_response = msg.get("content")
                    break
            
            if agent_response:
                print(f"Agent: {agent_response}")
            
        except requests.exceptions.RequestException as e:
            print(f"Error calling API: {e}")
            break

    print(f"\n-- Chat Test Ended (Session: {session_id}) --")

if __name__ == "__main__":
    run_chat_test()

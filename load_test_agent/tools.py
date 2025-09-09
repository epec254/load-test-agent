
"""
Fake tools for the Telco Customer Support Agent.
"""
import json
import random
import datetime
from typing import Literal

def get_billing_history(customer_id: str) -> str:
    """
    Retrieves the billing history for a given customer.
    
    Args:
        customer_id: The ID of the customer
    """
    print(f"Tool: get_billing_history, customer_id: {customer_id}")
    return json.dumps({
        "status": "success",
        "data": {
            "statements": [
                {"date": "2025-08-15", "amount": 125.50, "status": "paid"},
                {"date": "2025-07-15", "amount": 120.00, "status": "paid"},
                {"date": "2025-06-15", "amount": 180.00, "status": "paid", "unrecognized_charge": 40.00},
            ]
        }
    })

def get_charge_details(charge_id: str) -> str:
    """
    Retrieves details for a specific charge.
    
    Args:
        charge_id: The ID of the charge
    """
    print(f"Tool: get_charge_details, charge_id: {charge_id}")
    return json.dumps({
        "status": "success",
        "data": {
            "charge_id": charge_id,
            "description": "One-time data pack",
            "amount": 40.00,
            "date": "2025-06-10"
        }
    })

def get_active_promotions(customer_id: str) -> str:
    """
    Retrieves active promotions for a given customer.
    
    Args:
        customer_id: The ID of the customer
    """
    print(f"Tool: get_active_promotions, customer_id: {customer_id}")
    return json.dumps({
        "status": "success",
        "data": {
            "promotions": [
                {"name": "Loyalty Discount", "discount": "10%", "expires": "2025-12-31"}
            ]
        }
    })

def get_account_balance(customer_id: str) -> str:
    """
    Retrieves the current account balance for a given customer.
    
    Args:
        customer_id: The ID of the customer
    """
    print(f"Tool: get_account_balance, customer_id: {customer_id}")
    return json.dumps({
        "status": "success",
        "data": {
            "balance": 150.00,
            "due_date": "2025-09-25"
        }
    })

def get_saved_payment_methods(customer_id: str) -> str:
    """
    Retrieves the saved payment methods for a given customer.
    
    Args:
        customer_id: The ID of the customer
    """
    print(f"Tool: get_saved_payment_methods, customer_id: {customer_id}")
    return json.dumps({
        "status": "success",
        "data": {
            "methods": [
                {"type": "Visa", "last4": "1234", "id": "pm_1"},
                {"type": "Mastercard", "last4": "5678", "id": "pm_2"}
            ]
        }
    })

def process_payment(customer_id: str, amount: float, payment_method_id: str):
    """
    Processes a payment for a customer.
    """
    params = ProcessPaymentParams(customer_id=customer_id, amount=amount, payment_method_id=payment_method_id)
    print(f"Tool: process_payment, customer_id: {params.customer_id}, amount: {params.amount}, payment_method_id: {params.payment_method_id}")
    return json.dumps({
        "status": "success",
        "data": {
            "transaction_id": f"txn_{random.randint(1000, 9999)}",
            "amount": params.amount,
            "new_balance": 150.00 - params.amount
        }
    })

def get_network_status(location: str):
    """
    Retrieves the network status for a given location.
    """
    params = GetNetworkStatusParams(location=location)
    print(f"Tool: get_network_status, location: {params.location}")
    return json.dumps({
        "status": "success",
        "data": {
            "location": params.location,
            "network_status": "nominal",
            "known_outages": []
        }
    })

def run_remote_device_diagnostics(device_imei: str):
    """
    Runs remote diagnostics on a device.
    """
    params = RunRemoteDeviceDiagnosticsParams(device_imei=device_imei)
    print(f"Tool: run_remote_device_diagnostics, device_imei: {params.device_imei}")
    return json.dumps({
        "status": "success",
        "data": {
            "imei": device_imei,
            "diagnostics_result": "All systems nominal. Device is connected to the network."
        }
    })

def create_support_ticket(customer_id: str, issue_description: str):
    """
    Creates a support ticket.
    """
    params = CreateSupportTicketParams(customer_id=customer_id, issue_description=issue_description)
    print(f"Tool: create_support_ticket, customer_id: {params.customer_id}, issue_description: {params.issue_description}")
    return json.dumps({
        "status": "success",
        "data": {
            "ticket_id": f"tkt_{random.randint(10000, 99999)}",
            "status": "open",
            "created_at": datetime.datetime.now().isoformat()
        }
    })

def search_knowledge_base(query: str):
    """
    Searches the knowledge base for a given query.
    """
    params = SearchKnowledgeBaseParams(query=query)
    print(f"Tool: search_knowledge_base, query: {params.query}")
    if "no service" in params.query.lower():
        return json.dumps({
            "status": "success",
            "data": {
                "articles": [
                    {"title": "Troubleshooting 'No Service' issues", "article_id": "kb-001", "summary": "1. Restart your device. 2. Check for network outages in your area. 3. Reseat your SIM card."}
                ]
            }
        })
    return json.dumps({"status": "success", "data": {"articles": []}})


def get_insurance_coverage(customer_id: str, device_imei: str) -> str:
    """
    Retrieves insurance coverage for a device.
    
    Args:
        customer_id: The ID of the customer
        device_imei: The IMEI of the device
    """
    print(f"Tool: get_insurance_coverage, customer_id: {customer_id}, device_imei: {device_imei}")
    return json.dumps({
        "status": "success",
        "data": {
            "imei": device_imei,
            "is_covered": True,
            "deductible": {
                "screen_damage": 100.00,
                "water_damage": 250.00,
                "theft_loss": 500.00
            }
        }
    })

def process_insurance_claim(
    customer_id: str,
    device_imei: str,
    claim_type: Literal["screen_damage", "water_damage", "theft_loss"],
    incident_description: str
) -> str:
    """
    Processes an insurance claim.
    
    Args:
        customer_id: The ID of the customer
        device_imei: The IMEI of the device
        claim_type: The type of claim (screen_damage, water_damage, or theft_loss)
        incident_description: Description of the incident
    """
    print(f"Tool: process_insurance_claim, customer_id: {customer_id}, device_imei: {device_imei}, claim_type: {claim_type}")
    return json.dumps({
        "status": "success",
        "data": {
            "claim_id": f"clm_{random.randint(1000, 9999)}",
            "status": "processing",
            "next_steps": "You will receive an email with further instructions within 24 hours."
        }
    })

def get_data_usage(customer_id: str) -> str:
    """
    Retrieves data usage for a customer.
    
    Args:
        customer_id: The ID of the customer
    """
    print(f"Tool: get_data_usage, customer_id: {customer_id}")
    return json.dumps({
        "status": "success",
        "data": {
            "total_data": "20GB",
            "used_data": "15GB",
            "remaining_data": "5GB",
            "cycle_end_date": "2025-09-30"
        }
    })

def escalate_to_human_agent(customer_id: str, ticket_id: str) -> str:
    """
    Escalates the issue to a human agent.
    
    Args:
        customer_id: The ID of the customer
        ticket_id: The ID of the support ticket
    """
    print(f"Tool: escalate_to_human_agent, customer_id: {customer_id}, ticket_id: {ticket_id}")
    return json.dumps({
        "status": "success",
        "data": {
            "message": "I have escalated your issue to a human agent. They will contact you shortly.",
            "reference_id": f"esc_{random.randint(1000, 9999)}"
        }
    })

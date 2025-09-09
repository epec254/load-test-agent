"""
Fake tools for the Telco Customer Support Agent with realistic random data generation.
"""
import json
import random
import datetime
from typing import Literal
import mlflow

@mlflow.trace(span_type="TOOL")
def get_billing_history(customer_id: str) -> str:
    """
    Retrieves the billing history for a given customer.
    
    Args:
        customer_id: The ID of the customer
    """
    print(f"Tool: get_billing_history, customer_id: {customer_id}")
    
    # Generate 3-6 months of billing history
    num_months = random.randint(3, 6)
    statements = []
    base_amount = random.uniform(80, 200)
    
    for i in range(num_months):
        date = datetime.datetime.now() - datetime.timedelta(days=30 * i)
        amount = base_amount + random.uniform(-30, 50)
        
        # Random statuses and issues
        status_options = ["paid", "paid", "paid", "pending", "overdue"]
        status = random.choice(status_options)
        
        statement = {
            "date": date.strftime("%Y-%m-%d"),
            "amount": round(amount, 2),
            "status": status
        }
        
        # Occasionally add issues
        if random.random() < 0.3:
            issue_types = ["unrecognized_charge", "roaming_fees", "overage_charges", "equipment_fee"]
            issue = random.choice(issue_types)
            statement[issue] = round(random.uniform(15, 75), 2)
        
        # Occasionally add credits
        if random.random() < 0.2:
            statement["credit_applied"] = round(random.uniform(10, 50), 2)
            
        statements.append(statement)
    
    return json.dumps({
        "status": "success",
        "data": {
            "statements": statements,
            "account_type": random.choice(["individual", "family", "business"]),
            "auto_pay_enabled": random.choice([True, False])
        }
    })

@mlflow.trace(span_type="TOOL")
def get_charge_details(charge_id: str) -> str:
    """
    Retrieves details for a specific charge.
    
    Args:
        charge_id: The ID of the charge
    """
    print(f"Tool: get_charge_details, charge_id: {charge_id}")
    
    charge_types = [
        {"description": "International roaming - Canada", "category": "roaming", "amount": random.uniform(25, 150)},
        {"description": "Premium SMS services", "category": "premium_services", "amount": random.uniform(5, 30)},
        {"description": "Data overage (5GB)", "category": "overage", "amount": random.uniform(30, 75)},
        {"description": "Device protection plan", "category": "insurance", "amount": 15.99},
        {"description": "One-time data pack (10GB)", "category": "add_on", "amount": 40.00},
        {"description": "International calls - UK", "category": "international", "amount": random.uniform(10, 80)},
        {"description": "Device upgrade fee", "category": "equipment", "amount": 35.00},
        {"description": "Late payment fee", "category": "fee", "amount": 10.00},
        {"description": "Number change fee", "category": "service", "amount": 25.00},
        {"description": "Premium streaming bundle", "category": "entertainment", "amount": 29.99}
    ]
    
    charge = random.choice(charge_types)
    date = datetime.datetime.now() - datetime.timedelta(days=random.randint(1, 60))
    
    return json.dumps({
        "status": "success",
        "data": {
            "charge_id": charge_id,
            "description": charge["description"],
            "category": charge["category"],
            "amount": round(charge["amount"], 2),
            "date": date.strftime("%Y-%m-%d"),
            "tax_included": random.choice([True, False]),
            "refundable": random.choice([True, False, False])  # Less likely to be refundable
        }
    })
@mlflow.trace(span_type="TOOL")
def get_active_promotions(customer_id: str) -> str:
    """
    Retrieves active promotions for a given customer.
    
    Args:
        customer_id: The ID of the customer
    """
    print(f"Tool: get_active_promotions, customer_id: {customer_id}")
    
    all_promotions = [
        {"name": "Loyalty Discount", "discount": "10%", "type": "percentage"},
        {"name": "Family Plan Savings", "discount": "$20", "type": "fixed"},
        {"name": "New Customer Promo", "discount": "50%", "type": "percentage", "months_remaining": 3},
        {"name": "Student Discount", "discount": "15%", "type": "percentage"},
        {"name": "Military Discount", "discount": "25%", "type": "percentage"},
        {"name": "Bundle Savings", "discount": "$30", "type": "fixed"},
        {"name": "Referral Credit", "discount": "$50", "type": "one_time"},
        {"name": "Holiday Special", "discount": "20%", "type": "percentage", "months_remaining": 2}
    ]
    
    # Randomly select 0-3 promotions
    num_promos = random.randint(0, 3)
    selected_promos = random.sample(all_promotions, min(num_promos, len(all_promotions)))
    
    for promo in selected_promos:
        # Add expiration dates
        if "months_remaining" in promo:
            expires = datetime.datetime.now() + datetime.timedelta(days=30 * promo["months_remaining"])
        else:
            expires = datetime.datetime.now() + datetime.timedelta(days=random.randint(30, 365))
        promo["expires"] = expires.strftime("%Y-%m-%d")
    
    return json.dumps({
        "status": "success",
        "data": {
            "promotions": selected_promos,
            "eligible_for_new_promos": random.choice([True, False])
        }
    })
@mlflow.trace(span_type="TOOL")
def get_account_balance(customer_id: str) -> str:
    """
    Retrieves the current account balance for a given customer.
    
    Args:
        customer_id: The ID of the customer
    """
    print(f"Tool: get_account_balance, customer_id: {customer_id}")
    
    # Generate various balance scenarios
    balance_scenarios = [
        {"balance": 0, "status": "current"},
        {"balance": random.uniform(50, 300), "status": "due"},
        {"balance": random.uniform(300, 800), "status": "overdue"},
        {"balance": -random.uniform(10, 100), "status": "credit"},
        {"balance": random.uniform(100, 250), "status": "due_soon"}
    ]
    
    scenario = random.choice(balance_scenarios)
    due_date = datetime.datetime.now() + datetime.timedelta(days=random.randint(1, 30))
    
    response_data = {
        "balance": round(scenario["balance"], 2),
        "status": scenario["status"],
        "due_date": due_date.strftime("%Y-%m-%d")
    }
    
    # Add additional details based on status
    if scenario["status"] == "overdue":
        response_data["days_overdue"] = random.randint(1, 60)
        response_data["late_fee_applied"] = random.choice([True, False])
    elif scenario["status"] == "credit":
        response_data["credit_reason"] = random.choice(["refund", "promotional_credit", "loyalty_reward"])
    
    response_data["minimum_payment"] = round(max(25, scenario["balance"] * 0.1), 2) if scenario["balance"] > 0 else 0
    response_data["auto_pay_scheduled"] = random.choice([True, False])
    
    return json.dumps({
        "status": "success",
        "data": response_data
    })
@mlflow.trace(span_type="TOOL")
def get_saved_payment_methods(customer_id: str) -> str:
    """
    Retrieves the saved payment methods for a given customer.
    
    Args:
        customer_id: The ID of the customer
    """
    print(f"Tool: get_saved_payment_methods, customer_id: {customer_id}")
    
    card_types = ["Visa", "Mastercard", "American Express", "Discover"]
    bank_names = ["Chase", "Bank of America", "Wells Fargo", "Citi"]
    
    num_methods = random.randint(0, 4)
    methods = []
    
    for i in range(num_methods):
        method_type = random.choice(["card", "card", "bank"])  # Cards more common
        
        if method_type == "card":
            method = {
                "type": random.choice(card_types),
                "last4": str(random.randint(1000, 9999)),
                "id": f"pm_{i+1}",
                "expires": f"{random.randint(1, 12):02d}/{random.randint(24, 29)}",
                "is_default": i == 0
            }
        else:
            method = {
                "type": "Bank Account",
                "bank_name": random.choice(bank_names),
                "last4": str(random.randint(1000, 9999)),
                "id": f"pm_{i+1}",
                "account_type": random.choice(["checking", "savings"]),
                "is_default": i == 0
            }
        
        methods.append(method)
    
    return json.dumps({
        "status": "success",
        "data": {
            "methods": methods,
            "wallet_balance": round(random.uniform(0, 50), 2) if random.random() < 0.3 else 0
        }
    })
@mlflow.trace(span_type="TOOL")
def process_payment(customer_id: str, amount: float, payment_method_id: str):
    """
    Processes a payment for a customer.
    """
    print(f"Tool: process_payment, customer_id: {customer_id}, amount: {amount}, payment_method_id: {payment_method_id}")
    
    # Simulate different payment outcomes
    outcome = random.choices(
        ["success", "declined", "insufficient_funds", "processing"],
        weights=[0.8, 0.1, 0.05, 0.05]
    )[0]
    
    if outcome == "success":
        # Get previous balance (simulate)
        prev_balance = random.uniform(100, 500)
        new_balance = max(0, prev_balance - amount)
        
        return json.dumps({
            "status": "success",
            "data": {
                "transaction_id": f"txn_{random.randint(10000, 99999)}",
                "amount": amount,
                "previous_balance": round(prev_balance, 2),
                "new_balance": round(new_balance, 2),
                "processed_at": datetime.datetime.now().isoformat(),
                "receipt_sent_to": f"customer_{customer_id}@email.com"
            }
        })
    elif outcome == "declined":
        return json.dumps({
            "status": "error",
            "error": {
                "code": "payment_declined",
                "message": "Payment was declined by the card issuer",
                "suggestion": "Please try a different payment method or contact your bank"
            }
        })
    elif outcome == "insufficient_funds":
        return json.dumps({
            "status": "error",
            "error": {
                "code": "insufficient_funds",
                "message": "Insufficient funds in the account",
                "suggestion": "Please ensure sufficient balance and try again"
            }
        })
    else:  # processing
        return json.dumps({
            "status": "processing",
            "data": {
                "message": "Payment is being processed",
                "estimated_completion": "Within 24 hours",
                "reference_id": f"ref_{random.randint(10000, 99999)}"
            }
        })
@mlflow.trace(span_type="TOOL")
def get_network_status(location: str):
    """
    Retrieves the network status for a given location.
    """
    print(f"Tool: get_network_status, location: {location}")
    
    status_scenarios = [
        {
            "network_status": "nominal",
            "signal_strength": "excellent",
            "known_outages": []
        },
        {
            "network_status": "degraded",
            "signal_strength": "fair",
            "known_outages": [],
            "issues": ["High network congestion in area", "Expected resolution: 2 hours"]
        },
        {
            "network_status": "outage",
            "signal_strength": "no signal",
            "known_outages": [{
                "type": "tower_maintenance",
                "started": (datetime.datetime.now() - datetime.timedelta(hours=2)).isoformat(),
                "estimated_resolution": (datetime.datetime.now() + datetime.timedelta(hours=4)).isoformat(),
                "affected_services": ["voice", "data", "sms"]
            }]
        },
        {
            "network_status": "partial_outage",
            "signal_strength": "poor",
            "known_outages": [{
                "type": "equipment_failure",
                "started": (datetime.datetime.now() - datetime.timedelta(hours=1)).isoformat(),
                "estimated_resolution": "Under investigation",
                "affected_services": ["data"]
            }]
        },
        {
            "network_status": "maintenance",
            "signal_strength": "good",
            "known_outages": [],
            "scheduled_maintenance": {
                "date": (datetime.datetime.now() + datetime.timedelta(days=2)).strftime("%Y-%m-%d"),
                "time": "02:00-06:00 AM",
                "impact": "Service may be intermittent"
            }
        }
    ]
    
    scenario = random.choice(status_scenarios)
    scenario["location"] = location
    scenario["last_updated"] = datetime.datetime.now().isoformat()
    scenario["coverage_type"] = random.choice(["5G", "4G LTE", "4G", "3G"])
    
    return json.dumps({
        "status": "success",
        "data": scenario
    })
@mlflow.trace(span_type="TOOL")
def run_remote_device_diagnostics(device_imei: str):
    """
    Runs remote diagnostics on a device.
    """
    print(f"Tool: run_remote_device_diagnostics, device_imei: {device_imei}")
    
    diagnostic_scenarios = [
        {
            "diagnostics_result": "All systems nominal",
            "device_status": "connected",
            "signal_strength": random.randint(3, 5),
            "battery_level": random.randint(20, 100),
            "last_seen": datetime.datetime.now().isoformat()
        },
        {
            "diagnostics_result": "SIM card not detected",
            "device_status": "offline",
            "issues_found": ["SIM_NOT_DETECTED"],
            "recommended_actions": [
                "Remove and reinsert SIM card",
                "Clean SIM card contacts",
                "Try SIM in different device to test"
            ]
        },
        {
            "diagnostics_result": "Network authentication failure",
            "device_status": "authentication_error",
            "issues_found": ["NETWORK_AUTH_FAILED"],
            "recommended_actions": [
                "Restart device",
                "Reset network settings",
                "Contact support for account verification"
            ]
        },
        {
            "diagnostics_result": "Device software outdated",
            "device_status": "connected",
            "signal_strength": random.randint(1, 3),
            "issues_found": ["OUTDATED_SOFTWARE"],
            "current_version": "12.1.0",
            "latest_version": "14.2.1",
            "recommended_actions": ["Update device software to latest version"]
        },
        {
            "diagnostics_result": "Roaming detected",
            "device_status": "roaming",
            "signal_strength": random.randint(2, 4),
            "roaming_network": random.choice(["AT&T", "Verizon", "T-Mobile"]),
            "roaming_charges_apply": random.choice([True, False])
        },
        {
            "diagnostics_result": "Hardware issue detected",
            "device_status": "degraded",
            "issues_found": ["ANTENNA_MALFUNCTION"],
            "signal_strength": 1,
            "recommended_actions": [
                "Device may need repair",
                "Visit service center for diagnostic"
            ]
        }
    ]
    
    scenario = random.choice(diagnostic_scenarios)
    scenario["imei"] = device_imei
    scenario["device_model"] = random.choice(["iPhone 14", "Samsung S23", "Google Pixel 7", "iPhone 15 Pro"])
    scenario["diagnostic_id"] = f"diag_{random.randint(10000, 99999)}"
    
    return json.dumps({
        "status": "success",
        "data": scenario
    })
@mlflow.trace(span_type="TOOL")
def create_support_ticket(customer_id: str, issue_description: str):
    """
    Creates a support ticket.
    """
    print(f"Tool: create_support_ticket, customer_id: {customer_id}, issue_description: {issue_description}")
    
    # Determine priority based on keywords
    high_priority_keywords = ["no service", "emergency", "urgent", "cannot call", "fraud", "stolen"]
    medium_priority_keywords = ["slow", "billing", "charge", "payment"]
    
    priority = "P2"  # Default
    for keyword in high_priority_keywords:
        if keyword.lower() in issue_description.lower():
            priority = "P0"
            break
    
    if priority != "P0":
        for keyword in medium_priority_keywords:
            if keyword.lower() in issue_description.lower():
                priority = "P1"
                break
    
    ticket_id = f"tkt_{random.randint(100000, 999999)}"
    
    response = {
        "ticket_id": ticket_id,
        "status": "open",
        "priority": priority,
        "created_at": datetime.datetime.now().isoformat(),
        "customer_id": customer_id,
        "assigned_to": random.choice(["auto_queue", "tier1_support", "specialized_team"])
    }
    
    # Add estimated resolution time based on priority
    if priority == "P0":
        response["estimated_resolution"] = "Within 2 hours"
        response["escalated"] = True
    elif priority == "P1":
        response["estimated_resolution"] = "Within 24 hours"
    else:
        response["estimated_resolution"] = "Within 48-72 hours"
    
    return json.dumps({
        "status": "success",
        "data": response
    })
@mlflow.trace(span_type="TOOL")
def search_knowledge_base(query: str):
    """
    Searches the knowledge base for a given query.
    """
    print(f"Tool: search_knowledge_base, query: {query}")
    
    # Knowledge base with various articles
    kb_articles = {
        "no service": [
            {
                "title": "Troubleshooting 'No Service' issues",
                "article_id": "kb-001",
                "relevance_score": 0.95,
                "summary": "1. Restart your device. 2. Check for network outages in your area. 3. Remove and reinsert SIM card. 4. Reset network settings.",
                "last_updated": "2025-08-01"
            },
            {
                "title": "SIM Card Troubleshooting Guide",
                "article_id": "kb-002",
                "relevance_score": 0.75,
                "summary": "How to properly insert, clean, and troubleshoot SIM card issues.",
                "last_updated": "2025-07-15"
            }
        ],
        "payment": [
            {
                "title": "Payment Methods and Options",
                "article_id": "kb-010",
                "relevance_score": 0.90,
                "summary": "Add credit cards, bank accounts, or use digital wallets. Set up AutoPay for convenience.",
                "last_updated": "2025-08-10"
            },
            {
                "title": "Troubleshooting Failed Payments",
                "article_id": "kb-011",
                "relevance_score": 0.85,
                "summary": "Common reasons for payment failures and how to resolve them.",
                "last_updated": "2025-08-05"
            }
        ],
        "data usage": [
            {
                "title": "Understanding Your Data Usage",
                "article_id": "kb-020",
                "relevance_score": 0.92,
                "summary": "How to check data usage, what uses the most data, and tips to reduce consumption.",
                "last_updated": "2025-08-12"
            },
            {
                "title": "Data Saving Tips",
                "article_id": "kb-021",
                "relevance_score": 0.80,
                "summary": "Enable data saver mode, use Wi-Fi when available, disable auto-play videos.",
                "last_updated": "2025-07-20"
            }
        ],
        "slow": [
            {
                "title": "Fixing Slow Data Speeds",
                "article_id": "kb-030",
                "relevance_score": 0.88,
                "summary": "1. Check network congestion. 2. Clear cache. 3. Update carrier settings. 4. Check data throttling limits.",
                "last_updated": "2025-08-08"
            }
        ],
        "roaming": [
            {
                "title": "International Roaming Guide",
                "article_id": "kb-040",
                "relevance_score": 0.93,
                "summary": "How to enable roaming, check rates, and avoid unexpected charges while traveling.",
                "last_updated": "2025-08-15"
            }
        ],
        "device": [
            {
                "title": "Device Setup and Configuration",
                "article_id": "kb-050",
                "relevance_score": 0.78,
                "summary": "Initial setup, transferring data, and configuring your new device.",
                "last_updated": "2025-07-25"
            }
        ]
    }
    
    # Search for matching articles
    found_articles = []
    query_lower = query.lower()
    
    for keyword, articles in kb_articles.items():
        if keyword in query_lower:
            found_articles.extend(articles)
    
    # If no specific match, return generic troubleshooting
    if not found_articles:
        found_articles = [
            {
                "title": "General Troubleshooting Steps",
                "article_id": "kb-999",
                "relevance_score": 0.50,
                "summary": "Basic troubleshooting steps for common issues: restart device, check settings, contact support.",
                "last_updated": "2025-08-01"
            }
        ]
    
    # Sort by relevance and limit to top 3
    found_articles.sort(key=lambda x: x["relevance_score"], reverse=True)
    found_articles = found_articles[:3]
    
    return json.dumps({
        "status": "success",
        "data": {
            "query": query,
            "articles": found_articles,
            "total_results": len(found_articles)
        }
    })
@mlflow.trace(span_type="TOOL")
def get_insurance_coverage(customer_id: str, device_imei: str) -> str:
    """
    Retrieves insurance coverage for a device.
    
    Args:
        customer_id: The ID of the customer
        device_imei: The IMEI of the device
    """
    print(f"Tool: get_insurance_coverage, customer_id: {customer_id}, device_imei: {device_imei}")
    
    coverage_scenarios = [
        {
            "is_covered": True,
            "plan_type": "Premium Protection",
            "deductible": {
                "screen_damage": 100.00,
                "water_damage": 250.00,
                "theft_loss": 500.00,
                "other_damage": 200.00
            },
            "coverage_start_date": (datetime.datetime.now() - datetime.timedelta(days=random.randint(30, 365))).strftime("%Y-%m-%d"),
            "monthly_premium": 15.99,
            "claims_this_year": random.randint(0, 2)
        },
        {
            "is_covered": True,
            "plan_type": "Basic Protection",
            "deductible": {
                "screen_damage": 150.00,
                "water_damage": 350.00,
                "theft_loss": 750.00
            },
            "coverage_start_date": (datetime.datetime.now() - datetime.timedelta(days=random.randint(30, 365))).strftime("%Y-%m-%d"),
            "monthly_premium": 8.99,
            "claims_this_year": random.randint(0, 1)
        },
        {
            "is_covered": False,
            "reason": "Coverage expired",
            "expired_date": (datetime.datetime.now() - datetime.timedelta(days=random.randint(1, 30))).strftime("%Y-%m-%d"),
            "renewal_available": True,
            "renewal_premium": 12.99
        },
        {
            "is_covered": False,
            "reason": "No coverage purchased",
            "eligible_for_enrollment": True,
            "available_plans": [
                {"name": "Basic Protection", "monthly_premium": 8.99},
                {"name": "Premium Protection", "monthly_premium": 15.99}
            ]
        },
        {
            "is_covered": True,
            "plan_type": "Extended Warranty Only",
            "deductible": {
                "manufacturer_defects": 0.00,
                "accidental_damage": "Not covered",
                "theft_loss": "Not covered"
            },
            "coverage_start_date": (datetime.datetime.now() - datetime.timedelta(days=random.randint(30, 365))).strftime("%Y-%m-%d"),
            "monthly_premium": 4.99,
            "claims_this_year": 0
        }
    ]
    
    scenario = random.choice(coverage_scenarios)
    scenario["imei"] = device_imei
    scenario["device_model"] = random.choice(["iPhone 14 Pro", "Samsung Galaxy S23", "Google Pixel 8", "iPhone 15"])
    scenario["device_value"] = random.randint(600, 1500)
    
    if scenario.get("is_covered") and scenario.get("claims_this_year", 0) >= 2:
        scenario["warning"] = "Maximum claims limit (2) reached for this year"
    
    return json.dumps({
        "status": "success",
        "data": scenario
    })
@mlflow.trace(span_type="TOOL")
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
    
    claim_outcomes = [
        {
            "status": "approved",
            "claim_id": f"clm_{random.randint(100000, 999999)}",
            "deductible_amount": {
                "screen_damage": 100.00,
                "water_damage": 250.00,
                "theft_loss": 500.00
            }.get(claim_type, 200.00),
            "next_steps": [
                "Deductible payment required",
                "Replacement device will ship within 24-48 hours",
                "Return damaged device within 10 days using prepaid label"
            ],
            "replacement_device": {
                "model": "Same model (refurbished)",
                "color_options": ["Space Gray", "Silver", "Blue"],
                "estimated_delivery": (datetime.datetime.now() + datetime.timedelta(days=2)).strftime("%Y-%m-%d")
            }
        },
        {
            "status": "pending_review",
            "claim_id": f"clm_{random.randint(100000, 999999)}",
            "reason": "Additional documentation required",
            "required_documents": {
                "theft_loss": ["Police report", "Proof of purchase"],
                "water_damage": ["Photos of damage", "Incident details"],
                "screen_damage": ["Photos of damage"]
            }.get(claim_type, ["Photos of damage"]),
            "review_timeframe": "24-48 hours",
            "next_steps": ["Upload required documents through customer portal", "Claim specialist will review"]
        },
        {
            "status": "denied",
            "claim_id": f"clm_{random.randint(100000, 999999)}",
            "reason": random.choice([
                "Coverage limit exceeded for the year",
                "Incident not covered under current plan",
                "Device shows signs of intentional damage",
                "Claim filed outside of coverage period"
            ]),
            "appeal_available": True,
            "appeal_deadline": (datetime.datetime.now() + datetime.timedelta(days=30)).strftime("%Y-%m-%d")
        },
        {
            "status": "processing",
            "claim_id": f"clm_{random.randint(100000, 999999)}",
            "estimated_completion": "Within 24 hours",
            "current_step": random.choice([
                "Verifying coverage",
                "Reviewing incident details",
                "Checking device history",
                "Processing approval"
            ]),
            "next_steps": ["You will receive an email update once review is complete"]
        }
    ]
    
    # Higher chance of approval for legitimate-sounding claims
    if any(word in incident_description.lower() for word in ["accident", "dropped", "stolen", "broke"]):
        outcome = random.choices(claim_outcomes, weights=[0.6, 0.2, 0.1, 0.1])[0]
    else:
        outcome = random.choice(claim_outcomes)
    
    outcome["incident_date"] = datetime.datetime.now().strftime("%Y-%m-%d")
    outcome["claim_type"] = claim_type
    
    return json.dumps({
        "status": "success",
        "data": outcome
    })
@mlflow.trace(span_type="TOOL")
def get_data_usage(customer_id: str) -> str:
    """
    Retrieves data usage for a customer.
    
    Args:
        customer_id: The ID of the customer
    """
    print(f"Tool: get_data_usage, customer_id: {customer_id}")
    
    plan_types = [
        {"total_data": "Unlimited", "throttle_after": "50GB", "high_speed_used": random.uniform(5, 45)},
        {"total_data": "20GB", "used_data": random.uniform(0, 20)},
        {"total_data": "10GB", "used_data": random.uniform(0, 10)},
        {"total_data": "5GB", "used_data": random.uniform(0, 5)},
        {"total_data": "Unlimited", "throttle_after": "100GB", "high_speed_used": random.uniform(10, 95)}
    ]
    
    plan = random.choice(plan_types)
    cycle_start = datetime.datetime.now() - datetime.timedelta(days=random.randint(1, 25))
    cycle_end = cycle_start + datetime.timedelta(days=30)
    
    response = {
        "plan_name": random.choice(["Premium Unlimited", "Essential Data", "Basic Plan", "Family Share"]),
        "cycle_start_date": cycle_start.strftime("%Y-%m-%d"),
        "cycle_end_date": cycle_end.strftime("%Y-%m-%d"),
        "days_remaining": (cycle_end - datetime.datetime.now()).days
    }
    
    if "Unlimited" in plan.get("total_data", ""):
        response["total_data"] = "Unlimited"
        response["high_speed_data_cap"] = plan["throttle_after"]
        response["high_speed_used"] = f"{plan['high_speed_used']:.1f}GB"
        response["currently_throttled"] = plan["high_speed_used"] > float(plan["throttle_after"].replace("GB", ""))
        if response["currently_throttled"]:
            response["throttled_speed"] = "2G speeds (128kbps)"
    else:
        total = float(plan["total_data"].replace("GB", ""))
        used = plan["used_data"]
        response["total_data"] = plan["total_data"]
        response["used_data"] = f"{used:.1f}GB"
        response["remaining_data"] = f"{(total - used):.1f}GB"
        response["percentage_used"] = round((used / total) * 100, 1)
        
        if response["percentage_used"] > 90:
            response["warning"] = "Data usage is over 90%. Consider adding a data pack."
        elif response["percentage_used"] > 100:
            response["overage_charges"] = round((used - total) * 15, 2)  # $15 per GB overage
    
    # Add top usage categories
    response["top_usage_apps"] = random.sample([
        {"app": "Netflix", "data_used": f"{random.uniform(2, 10):.1f}GB"},
        {"app": "YouTube", "data_used": f"{random.uniform(1, 8):.1f}GB"},
        {"app": "Instagram", "data_used": f"{random.uniform(0.5, 3):.1f}GB"},
        {"app": "TikTok", "data_used": f"{random.uniform(1, 5):.1f}GB"},
        {"app": "Spotify", "data_used": f"{random.uniform(0.5, 2):.1f}GB"},
        {"app": "Web Browsing", "data_used": f"{random.uniform(0.5, 2):.1f}GB"}
    ], 3)
    
    return json.dumps({
        "status": "success",
        "data": response
    })
@mlflow.trace(span_type="TOOL")
def escalate_to_human_agent(customer_id: str, ticket_id: str) -> str:
    """
    Escalates the issue to a human agent.
    
    Args:
        customer_id: The ID of the customer
        ticket_id: The ID of the support ticket
    """
    print(f"Tool: escalate_to_human_agent, customer_id: {customer_id}, ticket_id: {ticket_id}")
    
    current_hour = datetime.datetime.now().hour
    is_business_hours = 8 <= current_hour < 20  # 8 AM to 8 PM
    
    if is_business_hours:
        wait_time = random.randint(2, 15)
        response = {
            "message": f"I have escalated your issue to a human agent. Current wait time is approximately {wait_time} minutes.",
            "queue_position": random.randint(3, 20),
            "estimated_wait_time": f"{wait_time} minutes",
            "reference_id": f"esc_{random.randint(10000, 99999)}",
            "department": random.choice(["Technical Support", "Billing Support", "Customer Care"]),
            "callback_available": True
        }
    else:
        response = {
            "message": "I have escalated your issue to a human agent. Our agents are currently offline, but your case has been prioritized for the next available agent.",
            "reference_id": f"esc_{random.randint(10000, 99999)}",
            "department": random.choice(["Technical Support", "Billing Support", "Customer Care"]),
            "business_hours": "8:00 AM - 8:00 PM EST",
            "callback_scheduled": True,
            "callback_time": "Tomorrow morning between 8:00 AM - 10:00 AM"
        }
    
    response["ticket_id"] = ticket_id
    response["priority_level"] = random.choice(["High", "Urgent", "Normal"])
    
    return json.dumps({
        "status": "success",
        "data": response
    })
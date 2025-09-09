#!/usr/bin/env python
"""Test script for enhanced fake tools"""

from load_test_agent.tools import *
import json

def test_tools():
    print("="*50)
    print("Testing Enhanced Fake Tools")
    print("="*50)
    
    # Test billing history
    print("\n1. Testing get_billing_history:")
    for _ in range(2):
        result = get_billing_history('cust_123')
        data = json.loads(result)
        print(f"  - {len(data['data']['statements'])} statements, Account: {data['data']['account_type']}, Auto-pay: {data['data']['auto_pay_enabled']}")
    
    # Test charge details 
    print("\n2. Testing get_charge_details:")
    for _ in range(3):
        result = get_charge_details(f'chg_{_}')
        data = json.loads(result)
        print(f"  - {data['data']['description']}: ${data['data']['amount']}")
    
    # Test account balance
    print("\n3. Testing get_account_balance:")
    for _ in range(3):
        result = get_account_balance('cust_123')
        data = json.loads(result)
        print(f"  - Balance: ${data['data']['balance']}, Status: {data['data']['status']}")
    
    # Test payment methods
    print("\n4. Testing get_saved_payment_methods:")
    result = get_saved_payment_methods('cust_123')
    data = json.loads(result)
    print(f"  - {len(data['data']['methods'])} payment methods saved")
    
    # Test network status
    print("\n5. Testing get_network_status:")
    locations = ['New York, NY', 'San Francisco, CA', 'Chicago, IL']
    for loc in locations:
        result = get_network_status(loc)
        data = json.loads(result)
        print(f"  - {loc}: {data['data']['network_status']} ({data['data']['signal_strength']})")
    
    # Test device diagnostics
    print("\n6. Testing run_remote_device_diagnostics:")
    for _ in range(3):
        result = run_remote_device_diagnostics(f'IMEI_{_}')
        data = json.loads(result)
        print(f"  - {data['data']['device_model']}: {data['data']['diagnostics_result']}")
    
    # Test support ticket creation
    print("\n7. Testing create_support_ticket:")
    issues = ["No service for 2 hours", "Billing charge incorrect", "Slow data speeds"]
    for issue in issues:
        result = create_support_ticket('cust_123', issue)
        data = json.loads(result)
        print(f"  - {issue}: Priority {data['data']['priority']}, ETA: {data['data']['estimated_resolution']}")
    
    # Test knowledge base search
    print("\n8. Testing search_knowledge_base:")
    queries = ["no service", "payment", "data usage", "slow internet"]
    for query in queries:
        result = search_knowledge_base(query)
        data = json.loads(result)
        articles = data['data']['articles']
        print(f"  - '{query}': {len(articles)} articles found")
        if articles:
            print(f"    Top: {articles[0]['title']}")
    
    # Test insurance coverage
    print("\n9. Testing get_insurance_coverage:")
    for _ in range(3):
        result = get_insurance_coverage('cust_123', f'IMEI_{_}')
        data = json.loads(result)
        covered = data['data'].get('is_covered', False)
        plan = data['data'].get('plan_type', 'No coverage')
        print(f"  - Device {_}: {'Covered' if covered else 'Not covered'} ({plan if covered else data['data'].get('reason', 'Unknown')})")
    
    # Test data usage
    print("\n10. Testing get_data_usage:")
    for _ in range(3):
        result = get_data_usage(f'cust_{_}')
        data = json.loads(result)
        total = data['data'].get('total_data', 'Unknown')
        if total == "Unlimited":
            used = data['data'].get('high_speed_used', 'Unknown')
            print(f"  - Plan: {data['data']['plan_name']}, Unlimited (used {used} high-speed)")
        else:
            used = data['data'].get('used_data', '0GB')
            remaining = data['data'].get('remaining_data', 'Unknown')
            print(f"  - Plan: {data['data']['plan_name']}, Used: {used}/{total} ({remaining} left)")
    
    # Test payment processing
    print("\n11. Testing process_payment:")
    for _ in range(3):
        result = process_payment('cust_123', 150.00, 'pm_1')
        data = json.loads(result)
        status = data['status']
        if status == 'success':
            print(f"  - Payment successful: Transaction {data['data']['transaction_id']}")
        elif status == 'error':
            print(f"  - Payment failed: {data['error']['message']}")
        elif status == 'processing':
            print(f"  - Payment processing: {data['data']['message']}")
    
    print("\n" + "="*50)
    print("All tests completed successfully!")
    print("="*50)

if __name__ == "__main__":
    test_tools()
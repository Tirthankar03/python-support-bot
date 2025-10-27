#!/usr/bin/env python3
"""
Simple test script for the IT Support Chatbot API
"""
import requests
import json

# Test configuration
BASE_URL = "http://localhost:8000"
TEST_CHAT_ID = "test_user_123"
TEST_NAME = "Test User"

def test_health():
    """Test health endpoint"""
    print("Testing health endpoint...")
    response = requests.get(f"{BASE_URL}/health")
    print(f"Status: {response.status_code}")
    print(f"Response: {response.json()}")
    print()

def test_support_endpoint():
    """Test the main support endpoint"""
    print("Testing support endpoint...")
    
    test_cases = [
        {
            "query": "Hi, my laptop won't turn on",
            "expected_keywords": ["laptop", "troubleshoot", "ticket"]
        },
        {
            "query": "I'm having Wi-Fi issues",
            "expected_keywords": ["wi-fi", "network", "router"]
        },
        {
            "query": "What's the weather like?",
            "expected_keywords": ["IT support", "laptop", "wi-fi"]
        },
        {
            "query": "My password is 123456",
            "expected_keywords": ["sensitive", "can't process"]
        }
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"Test case {i}: {test_case['query']}")
        
        payload = {
            "query": test_case["query"],
            "chatId": TEST_CHAT_ID,
            "name": TEST_NAME,
            "history": []
        }
        
        try:
            response = requests.post(f"{BASE_URL}/support", json=payload)
            print(f"Status: {response.status_code}")
            
            if response.status_code == 200:
                data = response.json()
                print(f"Response: {data['response']}")
                
                # Check if response contains expected keywords
                response_text = data['response'].lower()
                for keyword in test_case['expected_keywords']:
                    if keyword.lower() in response_text:
                        print(f"✓ Contains expected keyword: {keyword}")
                    else:
                        print(f"✗ Missing expected keyword: {keyword}")
            else:
                print(f"Error: {response.text}")
                
        except Exception as e:
            print(f"Exception: {e}")
        
        print("-" * 50)

def test_tickets_endpoint():
    """Test tickets endpoint"""
    print("Testing tickets endpoint...")
    
    try:
        response = requests.get(f"{BASE_URL}/tickets/{TEST_CHAT_ID}")
        print(f"Status: {response.status_code}")
        
        if response.status_code == 200:
            tickets = response.json()
            print(f"Tickets: {json.dumps(tickets, indent=2)}")
        else:
            print(f"Error: {response.text}")
            
    except Exception as e:
        print(f"Exception: {e}")
    
    print()

def test_history_endpoint():
    """Test history endpoint"""
    print("Testing history endpoint...")
    
    try:
        response = requests.get(f"{BASE_URL}/history/{TEST_CHAT_ID}")
        print(f"Status: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"History: {json.dumps(data, indent=2)}")
        else:
            print(f"Error: {response.text}")
            
    except Exception as e:
        print(f"Exception: {e}")
    
    print()

if __name__ == "__main__":
    print("IT Support Chatbot API Test")
    print("=" * 50)
    
    # Test all endpoints
    test_health()
    test_support_endpoint()
    test_tickets_endpoint()
    test_history_endpoint()
    
    print("Test completed!")

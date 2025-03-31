#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Test Client for Fraud Detection API
Sends test requests to the fraud detection API
"""

import os
import json
import random
import uuid
import time
import argparse
import requests
from datetime import datetime
import pandas as pd
from faker import Faker

# Initialize Faker
fake = Faker()

class FraudAPIClient:
    """Client for testing the Fraud Detection API"""
    
    def __init__(self, base_url="http://localhost:5000"):
        """Initialize with API base URL"""
        self.base_url = base_url
    
    def health_check(self):
        """Test the health check endpoint"""
        url = f"{self.base_url}/health"
        
        try:
            response = requests.get(url)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"Health check failed: {e}")
            return None
    
    def check_transaction(self, transaction):
        """Send a transaction to the fraud check endpoint"""
        url = f"{self.base_url}/api/v1/fraud/check"
        
        try:
            response = requests.post(url, json=transaction)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"Transaction check failed: {e}")
            return None
    
    def get_fraud_alerts(self, limit=100):
        """Get recent fraud alerts"""
        url = f"{self.base_url}/api/v1/fraud/alerts?limit={limit}"
        
        try:
            response = requests.get(url)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"Get alerts failed: {e}")
            return None
    
    def get_fraud_stats(self):
        """Get fraud statistics"""
        url = f"{self.base_url}/api/v1/fraud/stats"
        
        try:
            response = requests.get(url)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"Get stats failed: {e}")
            return None
    
    def get_customer_profile(self, customer_id):
        """Get customer profile"""
        url = f"{self.base_url}/api/v1/fraud/customer/{customer_id}"
        
        try:
            response = requests.get(url)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"Get customer profile failed: {e}")
            return None

def load_test_data(data_path="../data/sample_transactions.csv"):
    """Load test transaction data"""
    if not os.path.exists(data_path):
        print(f"Test data file not found: {data_path}")
        print("Please run generate_sample_data.py to create test data")
        return None
    
    try:
        df = pd.read_csv(data_path)
        print(f"Loaded {len(df)} test transactions from {data_path}")
        return df
    except Exception as e:
        print(f"Failed to load test data: {e}")
        return None

def generate_test_transaction():
    """Generate a random test transaction"""
    # Generate a realistic transaction
    amount = random.uniform(10, 2000)
    
    # Sometimes generate a high-value transaction
    if random.random() < 0.1:
        amount = random.uniform(5000, 20000)
    
    transaction = {
        'transaction_id': str(uuid.uuid4()),
        'customer_id': str(uuid.uuid4()),
        'amount': round(amount, 2),
        'merchant': fake.company(),
        'location': fake.city(),
        'transaction_time': datetime.now().isoformat(),
        'card_present': random.random() > 0.2,
        'currency': 'USD'
    }
    
    return transaction

def run_test_suite(client, test_data=None, num_tests=10, delay=1):
    """Run a test suite against the API"""
    print(f"Running test suite with {num_tests} transactions...")
    
    # Test health check
    print("\nTesting health check...")
    health = client.health_check()
    if health:
        print(f"Health check successful: {health['status']}")
    
    # Test transaction check
    print("\nTesting transaction check...")
    results = []
    
    for i in range(num_tests):
        if test_data is not None and not test_data.empty:
            # Use a random transaction from test data
            idx = random.randint(0, len(test_data) - 1)
            row = test_data.iloc[idx]
            
            # Convert row to dict and ensure all fields are the right type
            transaction = row.to_dict()

            for key, value in list(transaction.items()):
                if pd.isna(value):
                    transaction[key] = None
            
            # Clean up the transaction (remove is_fraud flag)
            if 'is_fraud' in transaction:
                is_fraud = transaction.pop('is_fraud')
                print(f"Transaction {i+1}/{num_tests} - Known {'fraudulent' if is_fraud else 'legitimate'}")
            else:
                print(f"Transaction {i+1}/{num_tests}")
                
        else:
            # Generate a random transaction
            transaction = generate_test_transaction()
            print(f"Transaction {i+1}/{num_tests} - Random")
        
        # Send to API
        result = client.check_transaction(transaction)
        if result:
            fraud_status = "FRAUD DETECTED" if result['is_fraudulent'] else "legitimate"
            if result['confidence'] is not None:
                print(f"  Result: {fraud_status} (confidence: {result['confidence']:.2f})")
            else:
                print(f"  Result: {fraud_status} (confidence: 0.00)")
            print(f"  Alert type: {result['alert_type']}")
            results.append(result)
        
        # Add delay between requests
        if i < num_tests - 1:
            time.sleep(delay)
    
    # Test fraud alerts
    print("\nTesting fraud alerts...")
    alerts = client.get_fraud_alerts(limit=10)
    if alerts:
        print(f"Retrieved {alerts['count']} fraud alerts")
        if alerts['count'] > 0:
            for i, alert in enumerate(alerts['alerts'][:3]):
                print(f"  Alert {i+1}: {alert['alert_type']} (score: {alert['confidence_score']})")
    
    # Test fraud stats
    print("\nTesting fraud statistics...")
    stats = client.get_fraud_stats()
    if stats and 'overall_stats' in stats:
        overall = stats['overall_stats']
        print(f"  Total transactions: {overall['total_transactions']}")
        print(f"  Fraud count: {overall['fraud_count']}")

        if overall['fraud_rate'] is not None:
            print(f"  Fraud rate: {overall['fraud_rate']*100:.2f}%")
        else:
            print("  Fraud rate: 0.00%")
            
        if overall['avg_amount'] is not None:
            # Check if avg_amount is a string or a number
            if isinstance(overall['avg_amount'], str):
                print(f"  Average transaction amount: ${overall['avg_amount']}")
            else:
                print(f"  Average transaction amount: ${overall['avg_amount']:.2f}")
        else:
            print("  Average transaction amount: $0.00")
    
    # Test customer profile
    if results and len(results) > 0:
        # Get a customer ID from a previous transaction
        customer_id = results[0]['transaction_id']
        
        print(f"\nTesting customer profile for {customer_id}...")
        profile = client.get_customer_profile(customer_id)
        if profile and 'customer_id' in profile:
            print(f"  Customer: {profile['name']} ({profile['customer_id']})")
            print(f"  Transaction count: {profile['transaction_count']}")
            print(f"  Fraud count: {profile['fraud_count']}")
            
            if 'recent_transactions' in profile and len(profile['recent_transactions']) > 0:
                print(f"  Recent transactions: {len(profile['recent_transactions'])}")
        else:
            print("  Customer profile not found (expected for generated customers)")
    
    print("\nTest suite complete!")

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='Test client for Fraud Detection API')
    parser.add_argument('--url', default='http://localhost:5000', help='API base URL')
    parser.add_argument('--data', default='../data/sample_transactions.csv', help='Test data file')
    parser.add_argument('--count', type=int, default=10, help='Number of test transactions')
    parser.add_argument('--delay', type=float, default=1.0, help='Delay between requests (seconds)')
    parser.add_argument('--stats', action='store_true', help='Only show fraud statistics')
    parser.add_argument('--alerts', action='store_true', help='Only show fraud alerts')
    
    args = parser.parse_args()
    
    # Create API client
    client = FraudAPIClient(base_url=args.url)
    
    # Check if API is available
    health = client.health_check()
    if not health:
        print("API not available. Please make sure the API server is running.")
        return 1
    
    # Handle specific commands
    if args.stats:
        print("Retrieving fraud statistics...")
        stats = client.get_fraud_stats()
        if stats and 'overall_stats' in stats:
            overall = stats['overall_stats']
            print(f"Total transactions: {overall['total_transactions']}")
            print(f"Fraud count: {overall['fraud_count']}")
            print(f"  Fraud rate: {overall['fraud_rate']*100:.2f if overall['fraud_rate'] is not None else 0.00}%")
            print(f"  Average transaction amount: ${overall['avg_amount']:.2f if overall['avg_amount'] is not None else 0.00}")
            
            if 'daily_stats' in stats and len(stats['daily_stats']) > 0:
                print("\nDaily fraud rates:")
                for day in stats['daily_stats']:
                    print(f"  {day['day']}: {day['fraud_rate']*100:.2f if day['fraud_rate'] is not None else 0.00}% ({day['fraud_count']}/{day['transaction_count']})")

        return 0
        
    if args.alerts:
        print("Retrieving fraud alerts...")
        alerts = client.get_fraud_alerts(limit=20)
        if alerts and 'alerts' in alerts:
            print(f"Retrieved {alerts['count']} fraud alerts")
            for i, alert in enumerate(alerts['alerts']):
                print(f"{i+1}. Transaction: {alert['transaction_id']}")
                print(f"   Type: {alert['alert_type']}")
                print(f"   Score: {alert['confidence_score']}")
                print(f"   Time: {alert['alert_time']}")
                print()
        return 0
    
    # Load test data if available
    test_data = load_test_data(args.data)
    
    # Run test suite
    run_test_suite(client, test_data, args.count, args.delay)
    
    
    return 0

if __name__ == "__main__":
    exit(main())
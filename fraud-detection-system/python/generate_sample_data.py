#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Sample Data Generator for Fraud Detection System
Generates sample data for testing and development
"""

import os
import random
import uuid
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from faker import Faker

# Initialize Faker
fake = Faker()

def generate_customers(num_customers=100):
    """Generate sample customer data"""
    print(f"Generating {num_customers} customers...")
    
    customers = []
    for i in range(num_customers):
        customer_id = str(uuid.uuid4())
        customers.append({
            'customer_id': customer_id,
            'name': fake.name(),
            'email': fake.email(),
            'address': fake.address(),
            'phone': fake.phone_number(),
            'created_at': fake.date_time_between(start_date='-5y', end_date='now').isoformat()
        })
    
    # Convert to DataFrame and save to CSV
    df = pd.DataFrame(customers)
    os.makedirs('../data', exist_ok=True)
    df.to_csv('../data/sample_customers.csv', index=False)
    print(f"Saved {len(df)} customers to ../data/sample_customers.csv")
    
    return customers

def generate_transactions(customers, num_transactions=10000, fraud_ratio=0.01):
    """Generate sample transaction data"""
    print(f"Generating {num_transactions} transactions (fraud ratio: {fraud_ratio*100}%)...")
    
    transactions = []
    fraud_count = 0
    
    # Transaction patterns for each customer
    customer_patterns = {}
    for customer in customers:
        customer_id = customer['customer_id']
        customer_patterns[customer_id] = {
            'avg_amount': random.uniform(50, 1000),
            'std_amount': random.uniform(20, 200),
            'usual_merchants': [fake.company() for _ in range(random.randint(3, 8))],
            'usual_locations': [fake.city() for _ in range(random.randint(1, 5))],
            'active_hours': (
                random.randint(7, 12),
                random.randint(13, 23)
            )
        }
    
    # Generate transactions
    end_date = datetime.now()
    start_date = end_date - timedelta(days=90)
    
    for i in range(num_transactions):
        # Pick a customer
        customer = random.choice(customers)
        customer_id = customer['customer_id']
        pattern = customer_patterns[customer_id]
        
        # Determine if this will be a fraudulent transaction
        is_fraud = random.random() < fraud_ratio
        
        # Generate transaction timestamp
        transaction_time = fake.date_time_between(start_date=start_date, end_date=end_date)
        
        if is_fraud:
            # Fraudulent transaction patterns
            fraud_pattern = random.choice(['amount', 'location', 'merchant', 'time'])
            
            if fraud_pattern == 'amount':
                # Unusually large amount
                amount = pattern['avg_amount'] * random.uniform(5, 15)
                merchant = random.choice(pattern['usual_merchants'])
                location = random.choice(pattern['usual_locations'])
                
            elif fraud_pattern == 'location':
                # Unusual location
                amount = random.normalvariate(pattern['avg_amount'], pattern['std_amount'])
                merchant = random.choice(pattern['usual_merchants'])
                location = fake.city()
                while location in pattern['usual_locations']:
                    location = fake.city()
                
            elif fraud_pattern == 'merchant':
                # Unusual merchant
                amount = random.normalvariate(pattern['avg_amount'], pattern['std_amount'])
                merchant = fake.company()
                while merchant in pattern['usual_merchants']:
                    merchant = fake.company()
                location = random.choice(pattern['usual_locations'])
                
            else:  # time
                # Unusual time (late night)
                amount = random.normalvariate(pattern['avg_amount'], pattern['std_amount'])
                merchant = random.choice(pattern['usual_merchants'])
                location = random.choice(pattern['usual_locations'])
                # Override to late night
                transaction_time = transaction_time.replace(hour=random.randint(1, 5))
            
            fraud_count += 1
            
        else:
            # Regular transaction
            amount = max(0.01, random.normalvariate(pattern['avg_amount'], pattern['std_amount']))
            merchant = random.choice(pattern['usual_merchants'])
            location = random.choice(pattern['usual_locations'])
        
        # Create transaction
        transaction = {
            'transaction_id': str(uuid.uuid4()),
            'customer_id': customer_id,
            'amount': round(amount, 2),
            'merchant': merchant,
            'location': location,
            'transaction_time': transaction_time.isoformat(),
            'card_present': random.random() > 0.2,  # 80% are card present
            'currency': 'USD',
            'is_fraud': is_fraud
        }
        
        transactions.append(transaction)
    
    # Convert to DataFrame and save to CSV
    df = pd.DataFrame(transactions)
    df.to_csv('../data/sample_transactions.csv', index=False)
    print(f"Saved {len(df)} transactions to ../data/sample_transactions.csv")
    print(f"Fraud count: {fraud_count} ({fraud_count/len(df)*100:.2f}%)")
    
    return transactions

def generate_sql_inserts(customers, transactions):
    """Generate SQL INSERT statements for the sample data"""
    print("Generating SQL insert statements...")
    
    with open('../data/sample_data.sql', 'w') as f:
        # Write header
        f.write("-- Sample data for Fraud Detection System\n\n")
        
        # Customer inserts
        f.write("-- Customer data\n")
        for customer in customers:
            f.write(
                f"INSERT INTO customers (customer_id, name, email, created_at) "
                f"VALUES ('{customer['customer_id']}', '{customer['name']}', "
                f"'{customer['email']}', '{customer['created_at']}');\n"
            )
        
        f.write("\n-- Transaction data\n")
        for tx in transactions:
            # Extract fraud flag (but don't include in the insert)
            is_fraud = tx.pop('is_fraud', False)
            
            # Create transaction insert
            cols = ", ".join(tx.keys())
            vals = ", ".join([
                f"'{v}'" if isinstance(v, str) else str(v) 
                for v in tx.values()
            ])
            
            f.write(f"INSERT INTO transactions ({cols}) VALUES ({vals});\n")
            
            # If fraudulent, create alert insert
            if is_fraud:
                f.write(
                    f"INSERT INTO fraud_alerts (transaction_id, alert_type, confidence_score, notes) "
                    f"VALUES ('{tx['transaction_id']}', 'sample_data', 0.9, 'Generated sample fraud');\n"
                )
        
        print(f"Saved SQL inserts to ../data/sample_data.sql")

def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate sample data for fraud detection system')
    parser.add_argument('--customers', type=int, default=100, help='Number of customers to generate')
    parser.add_argument('--transactions', type=int, default=10000, help='Number of transactions to generate')
    parser.add_argument('--fraud', type=float, default=0.01, help='Fraud ratio (0-1)')
    
    args = parser.parse_args()
    
    # Generate data
    customers = generate_customers(args.customers)
    transactions = generate_transactions(customers, args.transactions, args.fraud)
    generate_sql_inserts(customers, transactions)
    
    print("Sample data generation complete!")

if __name__ == "__main__":
    main()
#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Transaction Generator for Fraud Detection System
Generates synthetic transaction data and sends it to Kafka
"""

import json
import random
import time
import uuid
import argparse
import yaml
import os
import sys
from datetime import datetime, timedelta
from kafka import KafkaProducer
import pandas as pd
import numpy as np
from faker import Faker
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("generator.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("TransactionGenerator")

# Initialize Faker
fake = Faker()

class TransactionGenerator:
    """Generator for synthetic financial transaction data"""
    
    def __init__(self, config_path="config/config.yml"):
        """Initialize the generator with configuration"""
        self.config = self._load_config(config_path)
        self.producer = self._create_producer()
        self.customers = self._generate_customer_profiles()
        self.merchants_by_category = self._generate_merchant_categories()
        self.transaction_count = 0
        self.fraud_count = 0
        self.start_time = time.time()
        
    def _load_config(self, config_path):
        """Load configuration from YAML file"""
        try:
            with open(config_path, 'r') as config_file:
                return yaml.safe_load(config_file)
        except Exception as e:
            logger.error(f"Failed to load config: {e}")
            # Default configuration if file loading fails
            return {
                "kafka": {
                    "bootstrap_servers": "localhost:9092",
                    "topics": {
                        "transactions": "transactions"
                    }
                },
                "rules": {
                    "amount_threshold": 5000,
                }
            }
    
    def _create_producer(self):
        """Create and return a Kafka producer"""
        try:
            bootstrap_servers = self.config.get("kafka", {}).get(
                "bootstrap_servers", "localhost:9092"
            )
            return KafkaProducer(
                bootstrap_servers=bootstrap_servers,
                value_serializer=lambda v: json.dumps(v).encode('utf-8')
            )
        except Exception as e:
            logger.error(f"Failed to create Kafka producer: {e}")
            logger.info("Running in offline mode - transactions will be printed to console")
            return None
    
    def _generate_customer_profiles(self, num_customers=1000):
        """Generate synthetic customer profiles"""
        logger.info(f"Generating {num_customers} customer profiles")
        customers = []
        
        # Different customer segments
        segments = [
            {"name": "Low Spender", "weight": 0.3, "amount_range": (10, 100), "std_range": (5, 20)},
            {"name": "Average Consumer", "weight": 0.5, "amount_range": (50, 500), "std_range": (20, 80)},
            {"name": "High Roller", "weight": 0.15, "amount_range": (500, 3000), "std_range": (100, 500)},
            {"name": "Business Account", "weight": 0.05, "amount_range": (1000, 10000), "std_range": (500, 2000)}
        ]
        
        for i in range(num_customers):
            # Select customer segment based on weights
            segment = random.choices(
                segments, 
                weights=[s["weight"] for s in segments], 
                k=1
            )[0]
            
            # Generate customer profile
            avg_amount = random.uniform(*segment["amount_range"])
            std_amount = random.uniform(*segment["std_range"])
            
            # Generate travel patterns (some customers travel more)
            traveler_type = random.choices(
                ["non-traveler", "occasional", "frequent"],
                weights=[0.7, 0.2, 0.1],
                k=1
            )[0]
            
            if traveler_type == "non-traveler":
                num_locations = random.randint(1, 3)
            elif traveler_type == "occasional":
                num_locations = random.randint(3, 8)
            else:  # frequent traveler
                num_locations = random.randint(8, 15)
            
            customers.append({
                'customer_id': str(uuid.uuid4()),
                'name': fake.name(),
                'age': random.randint(18, 85),
                'address': fake.address(),
                'segment': segment["name"],
                'avg_transaction_amount': avg_amount,
                'std_transaction_amount': std_amount,
                'card_types': random.sample(
                    ['debit', 'credit', 'virtual', 'platinum', 'business'],
                    k=random.randint(1, 3)
                ),
                'usual_locations': [fake.city() for _ in range(num_locations)],
                'usual_merchants': [],  # Will be filled later
                'usual_hours': self._generate_active_hours(),
                'creation_date': fake.date_time_between(start_date='-5y', end_date='now')
            })
        
        logger.info(f"Generated {len(customers)} customer profiles")
        return customers
    
    def _generate_active_hours(self):
        """Generate the hours when a customer is typically active"""
        is_night_owl = random.random() < 0.1
        
        if is_night_owl:
            # Night owls might be active during late hours
            start_hour = random.randint(19, 23)
            end_hour = random.randint(0, 7)
        else:
            # Regular people are active during the day
            start_hour = random.randint(7, 12)
            end_hour = random.randint(start_hour + 5, 23)
        
        return (start_hour, end_hour)
    
    def _generate_merchant_categories(self):
        """Generate merchant categories with example merchants"""
        categories = {
            "retail": [f"{fake.company()} Retail" for _ in range(30)],
            "grocery": [f"{fake.company()} Grocery" for _ in range(20)],
            "restaurant": [f"{fake.company()} Restaurant" for _ in range(40)],
            "travel": [f"{fake.company()} Travel" for _ in range(15)],
            "subscription": [f"{fake.company()} Subscription" for _ in range(25)],
            "entertainment": [f"{fake.company()} Entertainment" for _ in range(20)],
            "health": [f"{fake.company()} Healthcare" for _ in range(15)],
            "other": [f"{fake.company()}" for _ in range(25)]
        }
        
        # Assign typical merchants to each customer
        for customer in self.customers:
            # Customer's favorite categories (different for each customer)
            fav_categories = random.sample(
                list(categories.keys()),
                k=random.randint(3, len(categories))
            )
            
            # Pick merchants from favorite categories
            for category in fav_categories:
                num_merchants = random.randint(1, 5)
                customer['usual_merchants'].extend(
                    random.sample(categories[category], k=min(num_merchants, len(categories[category])))
                )
        
        return categories
    
    def generate_normal_transaction(self, customer):
        """Generate a normal transaction for a customer"""
        # Pick a merchant the customer usually visits
        if customer['usual_merchants'] and random.random() < 0.8:
            merchant = random.choice(customer['usual_merchants'])
        else:
            # Occasionally, customer visits a new merchant
            category = random.choice(list(self.merchants_by_category.keys()))
            merchant = random.choice(self.merchants_by_category[category])
        
        # Normal distribution for transaction amount based on customer's profile
        amount = np.random.normal(
            customer['avg_transaction_amount'], 
            customer['std_transaction_amount']
        )
        # Ensure amount is positive and round to 2 decimal places
        amount = max(0.01, round(amount, 2))
        
        # Pick a location (mostly from their usual locations)
        if customer['usual_locations'] and random.random() < 0.9:
            location = random.choice(customer['usual_locations'])
        else:
            location = fake.city()
        
        # Pick a transaction time (mostly during their active hours)
        now = datetime.now()
        if random.random() < 0.8:
            start_hour, end_hour = customer['usual_hours']
            if start_hour < end_hour:
                hour = random.randint(start_hour, end_hour)
            else:
                # Handle night owls (e.g. active from 22h to 4h)
                hour = random.randint(start_hour, 23) if random.random() < 0.5 else random.randint(0, end_hour)
        else:
            hour = random.randint(0, 23)
            
        # Set transaction time
        transaction_time = now.replace(hour=hour, minute=random.randint(0, 59), second=random.randint(0, 59))
        
        # Select a card type
        card_type = random.choice(customer['card_types'])
        
        # Determine if card is present (in-person transaction)
        # More likely for retail, grocery, restaurant
        is_card_present = random.random() < 0.7
        
        # Build transaction object
        transaction = {
            'transaction_id': str(uuid.uuid4()),
            'customer_id': customer['customer_id'],
            'merchant': merchant,
            'amount': amount,
            'location': location,
            'transaction_time': transaction_time.isoformat(),
            'card_present': is_card_present,
            'card_type': card_type,
            'currency': 'USD'
        }
        
        return transaction
    
    def generate_fraudulent_transaction(self, customer):
        """Generate a fraudulent transaction based on known fraud patterns"""
        # Choose a fraud pattern
        pattern = random.choice([
            'unusual_amount',
            'unusual_location',
            'unusual_merchant',
            'unusual_hour',
            'velocity'
        ])
        
        # Start with a normal transaction
        transaction = self.generate_normal_transaction(customer)
        
        # Modify based on fraud pattern
        if pattern == 'unusual_amount':
            # Very large amount compared to customer's normal pattern
            transaction['amount'] = round(customer['avg_transaction_amount'] * random.uniform(5, 20), 2)
            transaction['fraud_pattern'] = 'unusual_amount'
            
        elif pattern == 'unusual_location':
            # Transaction from unusual location, often international
            countries = ['Nigeria', 'Russia', 'China', 'Brazil', 'India', 'Ukraine', 'Romania']
            transaction['location'] = f"{fake.city()}, {random.choice(countries)}"
            transaction['fraud_pattern'] = 'unusual_location'
            
        elif pattern == 'unusual_merchant':
            # Unusual merchant category, often associated with fraud
            suspicious_merchants = [
                "XCryptoExchange",
                "QuickCash ATM",
                "DigitalGiftCards",
                "OneTimePayments",
                "Offshore Investments Ltd",
                "QuickForeignTransfer"
            ]
            transaction['merchant'] = random.choice(suspicious_merchants)
            transaction['fraud_pattern'] = 'unusual_merchant'
            
        elif pattern == 'unusual_hour':
            # Transaction at unusual hour for this customer
            start_hour, end_hour = customer['usual_hours']
            unusual_hours = list(range(0, 24))
            for h in range(start_hour, end_hour + 1):
                if h in unusual_hours:
                    unusual_hours.remove(h)
            
            if unusual_hours:
                hour = random.choice(unusual_hours)
                transaction_time = datetime.fromisoformat(transaction['transaction_time'])
                transaction['transaction_time'] = transaction_time.replace(hour=hour).isoformat()
                transaction['fraud_pattern'] = 'unusual_hour'
                
        elif pattern == 'velocity':
            # This would be detected by multiple transactions in short time
            # Just mark it for now, the actual detection would happen in Flink
            transaction['fraud_pattern'] = 'velocity'
            
        # Add a flag for validation/testing purposes
        transaction['is_fraudulent'] = True
        
        return transaction
    
    def save_to_csv(self, transactions, filename='../data/sample_transactions.csv'):
        """Save transactions to CSV file for testing/analysis"""
        df = pd.DataFrame(transactions)
        df.to_csv(filename, index=False)
        logger.info(f"Saved {len(transactions)} transactions to {filename}")
    
    def run(self, num_transactions=None, fraud_ratio=0.01, transactions_per_second=10, save_sample=True):
        """
        Run the transaction generator
        
        Args:
            num_transactions: Total transactions to generate (None for unlimited)
            fraud_ratio: Percentage of transactions that should be fraudulent
            transactions_per_second: Number of transactions per second
            save_sample: Whether to save a sample of transactions to CSV
        """
        logger.info(f"Starting transaction generator")
        logger.info(f"Fraud ratio: {fraud_ratio*100}%")
        logger.info(f"Rate: {transactions_per_second} transactions per second")
        
        transactions = []
        
        try:
            i = 0
            while num_transactions is None or i < num_transactions:
                # Select a random customer
                customer = random.choice(self.customers)
                
                # Determine if this transaction should be fraudulent
                is_fraud = random.random() < fraud_ratio
                
                if is_fraud:
                    transaction = self.generate_fraudulent_transaction(customer)
                    self.fraud_count += 1
                else:
                    transaction = self.generate_normal_transaction(customer)
                
                # Add to collection if we're saving samples
                if save_sample and len(transactions) < 10000:
                    transactions.append(transaction)
                
                # Send transaction to Kafka
                if self.producer:
                    topic = self.config.get("kafka", {}).get("topics", {}).get("transactions", "transactions")
                    self.producer.send(topic, transaction)
                else:
                    # If no Kafka, print to console occasionally
                    if i % 100 == 0:
                        logger.debug(f"Transaction: {json.dumps(transaction, indent=2)}")
                
                self.transaction_count += 1
                i += 1
                
                # Log progress periodically
                if self.transaction_count % 100 == 0:
                    elapsed = time.time() - self.start_time
                    tps = self.transaction_count / elapsed if elapsed > 0 else 0
                    fraud_percentage = (self.fraud_count / self.transaction_count) * 100
                    
                    logger.info(f"Generated {self.transaction_count} transactions "
                          f"({self.fraud_count} fraudulent - {fraud_percentage:.2f}%), "
                          f"Rate: {tps:.2f} TPS")
                
                # Control the sending rate
                time.sleep(1 / transactions_per_second)
                
                # Save sample periodically
                if save_sample and len(transactions) >= 10000:
                    self.save_to_csv(transactions)
                    transactions = []  # Reset after saving
        
        except KeyboardInterrupt:
            logger.info("Generator stopped by user")
        except Exception as e:
            logger.error(f"Error generating transactions: {e}")
        finally:
            # Save any remaining transactions
            if save_sample and transactions:
                self.save_to_csv(transactions)
                
            # Flush and close Kafka producer
            if self.producer:
                self.producer.flush()
                self.producer.close()
                
            logger.info("Generator shutdown complete")
            
            
def main():
    """Main entry point for the script"""
    parser = argparse.ArgumentParser(description='Generate synthetic transaction data')
    parser.add_argument('--config', default='config/config.yml', help='Path to config file')
    parser.add_argument('--topic', help='Kafka topic to use (overrides config)')
    parser.add_argument('--fraud', type=float, default=0.01, help='Fraud ratio (0-1)')
    parser.add_argument('--tps', type=int, default=10, help='Transactions per second')
    parser.add_argument('--count', type=int, default=None, help='Number of transactions to generate (default: unlimited)')
    parser.add_argument('--save', action='store_true', help='Save sample to CSV')
    
    args = parser.parse_args()
    
    # Create and run generator
    generator = TransactionGenerator(config_path=args.config)
    generator.run(
        num_transactions=args.count,
        fraud_ratio=args.fraud,
        transactions_per_second=args.tps,
        save_sample=args.save
    )


if __name__ == "__main__":
    main()
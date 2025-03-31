#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Rule-Based Fraud Detection for Fraud Detection System
Implements rules to detect potentially fraudulent transactions
"""

import yaml
import logging
import json
from datetime import datetime, timedelta
from database import DatabaseManager

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("FraudRules")

class FraudRuleEngine:
    """Rule engine for fraud detection"""
    
    def __init__(self, config_path="config/config.yml"):
        """Initialize with configuration"""
        self.config = self._load_config(config_path)
        self.db = DatabaseManager(config_path)
        
        # Load rules from config
        self.rules_config = self.config.get("rules", {})
        
        # Transaction history cache (for velocity checks)
        # In a real system, this would use Redis or another caching system
        self.transaction_cache = {}
        
    def _load_config(self, config_path):
        """Load configuration from YAML file"""
        try:
            with open(config_path, 'r') as config_file:
                return yaml.safe_load(config_file)
        except Exception as e:
            logger.error(f"Failed to load config: {e}")
            # Default configuration
            return {
                "rules": {
                    "amount_threshold": 5000,
                    "velocity_threshold": 3,
                    "location_jump_check": True,
                    "unusual_time_check": True
                }
            }
    
    def check_amount_threshold(self, transaction):
        """Check if transaction amount exceeds threshold"""
        amount = transaction.get('amount', 0)
        threshold = self.rules_config.get("amount_threshold", 5000)
        
        if amount > threshold:
            confidence = min(0.5 + (amount / threshold - 1) * 0.1, 0.95)
            return True, confidence, "large_amount"
        
        return False, 0, None
    
    def check_velocity(self, transaction):
        """Check for multiple transactions in a short time period"""
        customer_id = transaction.get('customer_id')
        curr_time = datetime.fromisoformat(transaction.get('transaction_time').replace('Z', '+00:00'))
        threshold = self.rules_config.get("velocity_threshold", 3)
        
        # Get customer's recent transactions from cache or database
        if customer_id in self.transaction_cache:
            recent_transactions = self.transaction_cache[customer_id]
        else:
            # Fetch from database
            db_transactions = self.db.get_customer_transactions(customer_id, limit=10)
            recent_transactions = []
            
            for t in db_transactions:
                tx_time = t.transaction_time
                recent_transactions.append({
                    'transaction_id': t.transaction_id,
                    'amount': t.amount,
                    'transaction_time': tx_time
                })
            
            # Update cache
            self.transaction_cache[customer_id] = recent_transactions
        
        # Add current transaction to recent list
        recent_transactions.append({
            'transaction_id': transaction.get('transaction_id'),
            'amount': transaction.get('amount'),
            'transaction_time': curr_time
        })
        
        # Sort by time
        recent_transactions.sort(key=lambda x: x['transaction_time'], reverse=True)
        
        # Limit cache size
        recent_transactions = recent_transactions[:20]
        self.transaction_cache[customer_id] = recent_transactions
        
        # Count transactions in the last minute
        one_minute_ago = curr_time - timedelta(minutes=1)
        count_last_minute = sum(1 for t in recent_transactions 
                              if t['transaction_time'] >= one_minute_ago)
        
        if count_last_minute > threshold:
            confidence = min(0.6 + (count_last_minute / threshold - 1) * 0.1, 0.95)
            return True, confidence, "velocity"
        
        return False, 0, None
    
    def check_location_jump(self, transaction):
        """Check for impossible travel (transactions from distant locations in short time)"""
        if not self.rules_config.get("location_jump_check", True):
            return False, 0, None
            
        customer_id = transaction.get('customer_id')
        curr_location = transaction.get('location', '')
        curr_time = datetime.fromisoformat(transaction.get('transaction_time').replace('Z', '+00:00'))
        
        # Skip check if no location data
        if not curr_location:
            return False, 0, None
        
        # Get customer's recent transactions
        if customer_id in self.transaction_cache:
            recent_transactions = self.transaction_cache[customer_id]
        else:
            # Skip check if no cache data
            return False, 0, None
        
        # Find most recent transaction before current one
        prev_transactions = [t for t in recent_transactions 
                           if t['transaction_id'] != transaction.get('transaction_id')
                           and 'location' in t and t['location']]
        
        if not prev_transactions:
            return False, 0, None
            
        prev_transaction = prev_transactions[0]
        prev_location = prev_transaction.get('location', '')
        prev_time = prev_transaction.get('transaction_time')
        
        # Skip if previous location is same as current
        if prev_location == curr_location:
            return False, 0, None
            
        # Simple location jump detection
        # In a real system, you would use geolocation APIs to calculate distance
        # and determine if travel between locations in the time window is possible
        time_diff = (curr_time - prev_time).total_seconds() / 3600  # in hours
        
        # Simplified check: If locations differ and time window is small
        # This is a very basic check and would need enhancement in a real system
        if time_diff < 2:  # Less than 2 hours between transactions
            confidence = 0.7
            return True, confidence, "location_jump"
        
        return False, 0, None
    
    def check_unusual_time(self, transaction):
        """Check if transaction occurs at unusual time for this customer"""
        if not self.rules_config.get("unusual_time_check", True):
            return False, 0, None
            
        # Extract hour from transaction time
        curr_time = datetime.fromisoformat(transaction.get('transaction_time').replace('Z', '+00:00'))
        hour = curr_time.hour
        
        # Late night transactions (2am - 5am) might be suspicious
        # This is a simplified check
        if 2 <= hour <= 5:
            confidence = 0.5  # Lower confidence since many legitimate transactions happen at night
            return True, confidence, "unusual_time"
        
        return False, 0, None
    
    def evaluate_transaction(self, transaction):
        """
        Apply all fraud detection rules to a transaction
        
        Returns:
            tuple: (is_fraudulent, confidence_score, alert_type)
        """
        # Check each rule
        amount_check = self.check_amount_threshold(transaction)
        velocity_check = self.check_velocity(transaction)
        location_check = self.check_location_jump(transaction)
        time_check = self.check_unusual_time(transaction)
        
        # Combine results - if any rule flags fraud
        checks = [amount_check, velocity_check, location_check, time_check]
        fraud_checks = [c for c in checks if c[0]]
        
        if fraud_checks:
            # Get the highest confidence rule
            is_fraud, confidence, alert_type = max(fraud_checks, key=lambda x: x[1])
            return is_fraud, confidence, alert_type
        
        return False, 0, None

# For direct testing
if __name__ == "__main__":
    # Create a test transaction
    test_transaction = {
        'transaction_id': '12345',
        'customer_id': 'c-789',
        'amount': 6000,
        'location': 'New York, USA',
        'transaction_time': datetime.now().isoformat(),
        'merchant': 'Test Merchant',
        'card_present': False
    }
    
    # Create and test rule engine
    rule_engine = FraudRuleEngine()
    is_fraud, confidence, alert_type = rule_engine.evaluate_transaction(test_transaction)
    
    print(f"Transaction evaluation:")
    print(f"  Fraud detected: {is_fraud}")
    print(f"  Confidence: {confidence:.2f}")
    print(f"  Alert type: {alert_type}")
    
    # Test with a velocity transaction
    velocity_transaction = test_transaction.copy()
    velocity_transaction['transaction_id'] = '12346'
    velocity_transaction['amount'] = 100
    
    is_fraud, confidence, alert_type = rule_engine.evaluate_transaction(velocity_transaction)
    
    print(f"\nVelocity transaction evaluation:")
    print(f"  Fraud detected: {is_fraud}")
    print(f"  Confidence: {confidence:.2f}")
    print(f"  Alert type: {alert_type}")
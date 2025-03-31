#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
REST API for Fraud Detection System
Provides HTTP endpoints to interact with the fraud detection system
"""

import yaml
import json
import logging
from datetime import datetime
from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
from database import DatabaseManager
from fraud_rules import FraudRuleEngine
from model_predictor import FraudModelPredictor

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("FraudAPI")

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj) if not np.isnan(obj) else None
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)
    
# Initialize Flask app
app = Flask(__name__)
app.json_encoder = NpEncoder

# Load configuration
def load_config(config_path="config/config.yml"):
    """Load configuration from YAML file"""
    try:
        with open(config_path, 'r') as config_file:
            return yaml.safe_load(config_file)
    except Exception as e:
        logger.error(f"Failed to load config: {e}")
        return {}

# Global variables
config = load_config()
db = DatabaseManager()
rule_engine = FraudRuleEngine()
model_predictor = FraudModelPredictor()

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'ok',
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/v1/fraud/check', methods=['POST'])
def check_transaction():
    """
    Check a transaction for fraud
    
    Expects JSON with transaction details:
    {
        "transaction_id": "string",
        "customer_id": "string",
        "amount": number,
        "merchant": "string",
        "location": "string",
        "transaction_time": "string (ISO format)",
        "card_present": boolean,
        "currency": "string"
    }
    """
    try:
        # Get transaction data from request
        transaction = request.json
        
        if not transaction:
            return jsonify({
                'error': 'No transaction data provided'
            }), 400
        
        # Validate required fields
        required_fields = ['transaction_id', 'customer_id', 'amount']
        for field in required_fields:
            if field not in transaction:
                return jsonify({
                    'error': f'Missing required field: {field}'
                }), 400
        
        # Get customer stats from database
        customer_stats = get_customer_stats(transaction['customer_id'])
        
        # Apply rule-based detection
        rule_fraud, rule_confidence, rule_type = rule_engine.evaluate_transaction(transaction)
        
        # Apply ML model-based detection
        ml_fraud, ml_confidence, ml_type = model_predictor.predict(transaction, customer_stats)
        
        # Combine results (if either method flags fraud)
        is_fraudulent = rule_fraud or ml_fraud
        
        # Use the higher confidence score
        confidence = max(rule_confidence, ml_confidence)
        
        # Determine alert type (use the method with higher confidence)
        alert_type = rule_type if rule_confidence >= ml_confidence else ml_type
        
        # Store transaction in database (asynchronously in a real system)
        store_transaction(transaction, is_fraudulent, confidence, alert_type)
        
        # Return result
        return jsonify({
            'transaction_id': transaction['transaction_id'],
            'is_fraudulent': is_fraudulent,
            'confidence': confidence,
            'alert_type': alert_type,
            'rule_based_result': {
                'is_fraudulent': rule_fraud,
                'confidence': rule_confidence,
                'alert_type': rule_type
            },
            'ml_based_result': {
                'is_fraudulent': ml_fraud,
                'confidence': ml_confidence,
                'alert_type': ml_type
            },
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error checking transaction: {e}")
        return jsonify({
            'error': 'Internal server error',
            'message': str(e)
        }), 500

@app.route('/api/v1/fraud/alerts', methods=['GET'])
def get_fraud_alerts():
    """Get recent fraud alerts"""
    try:
        # Get limit from query parameter (default to 100)
        limit = request.args.get('limit', 100, type=int)
        
        # Get alerts from database
        alerts = db.get_recent_fraud_alerts(limit)
        
        # Convert to list of dictionaries
        result = []
        for alert in alerts:
            result.append({
                'alert_id': alert.alert_id,
                'transaction_id': alert.transaction_id,
                'alert_time': alert.alert_time.isoformat() if alert.alert_time else None,
                'alert_type': alert.alert_type,
                'confidence_score': alert.confidence_score,
                'is_confirmed': alert.is_confirmed
            })
        
        return jsonify({
            'alerts': result,
            'count': len(result),
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error getting fraud alerts: {e}")
        return jsonify({
            'error': 'Internal server error',
            'message': str(e)
        }), 500

@app.route('/api/v1/fraud/stats', methods=['GET'])
def get_fraud_stats():
    """Get fraud statistics"""
    try:
        # Query database for stats
        stats_query = """
        SELECT 
            COUNT(*) as total_transactions,
            SUM(CASE WHEN fa.alert_id IS NOT NULL THEN 1 ELSE 0 END) as fraud_count,
            AVG(t.amount) as avg_amount,
            MAX(t.amount) as max_amount,
            MIN(t.amount) as min_amount
        FROM 
            transactions t
        LEFT JOIN 
            fraud_alerts fa ON t.transaction_id = fa.transaction_id
        """
        
        stats_result = db.execute_raw_query(stats_query)
        
        if not stats_result:
            return jsonify({
                'error': 'No data available'
            }), 404
        
        # Get fraud by day
        daily_query = """
        SELECT 
            DATE(t.transaction_time) as day,
            COUNT(*) as transaction_count,
            SUM(CASE WHEN fa.alert_id IS NOT NULL THEN 1 ELSE 0 END) as fraud_count
        FROM 
            transactions t
        LEFT JOIN 
            fraud_alerts fa ON t.transaction_id = fa.transaction_id
        GROUP BY 
            DATE(t.transaction_time)
        ORDER BY 
            day DESC
        LIMIT 30
        """
        
        daily_result = db.execute_raw_query(daily_query)
        
        # Format results
        daily_stats = []
        for day in daily_result:
            daily_stats.append({
                'day': day['day'].isoformat() if hasattr(day['day'], 'isoformat') else day['day'],
                'transaction_count': day['transaction_count'],
                'fraud_count': day['fraud_count'],
                'fraud_rate': day['fraud_count'] / day['transaction_count'] if day['transaction_count'] > 0 else 0
            })
        
        # Calculate fraud rate
        stats = stats_result[0]
        fraud_rate = stats['fraud_count'] / stats['total_transactions'] if stats['total_transactions'] > 0 else 0
        
        return jsonify({
            'overall_stats': {
                'total_transactions': stats['total_transactions'],
                'fraud_count': stats['fraud_count'],
                'fraud_rate': fraud_rate,
                'avg_amount': stats['avg_amount'],
                'max_amount': stats['max_amount'],
                'min_amount': stats['min_amount']
            },
            'daily_stats': daily_stats,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error getting fraud stats: {e}")
        return jsonify({
            'error': 'Internal server error',
            'message': str(e)
        }), 500

@app.route('/api/v1/fraud/customer/<customer_id>', methods=['GET'])
def get_customer_profile(customer_id):
    """Get customer profile and transaction history"""
    try:
        # Query database for customer profile
        profile_query = """
        SELECT 
            c.customer_id,
            c.name,
            c.email,
            cp.avg_transaction_amount,
            cp.common_merchants,
            cp.common_locations,
            COUNT(t.transaction_id) as transaction_count,
            SUM(CASE WHEN fa.alert_id IS NOT NULL THEN 1 ELSE 0 END) as fraud_count
        FROM 
            customers c
        LEFT JOIN 
            customer_profiles cp ON c.customer_id = cp.customer_id
        LEFT JOIN 
            transactions t ON c.customer_id = t.customer_id
        LEFT JOIN 
            fraud_alerts fa ON t.transaction_id = fa.transaction_id
        WHERE 
            c.customer_id = %s
        GROUP BY 
            c.customer_id, c.name, c.email, cp.avg_transaction_amount, cp.common_merchants, cp.common_locations
        """
        
        profile_result = db.execute_raw_query(profile_query, {'customer_id': customer_id})
        
        if not profile_result:
            return jsonify({
                'error': 'Customer not found'
            }), 404
        
        # Get recent transactions
        transactions = db.get_customer_transactions(customer_id, limit=20)
        
        # Format transactions
        transaction_list = []
        for tx in transactions:
            transaction_list.append({
                'transaction_id': tx.transaction_id,
                'amount': tx.amount,
                'merchant': tx.merchant,
                'transaction_time': tx.transaction_time.isoformat() if tx.transaction_time else None,
                'location': tx.location,
                'is_fraudulent': bool(tx.fraud_alerts)
            })
        
        # Format profile
        profile = profile_result[0]
        fraud_rate = profile['fraud_count'] / profile['transaction_count'] if profile['transaction_count'] > 0 else 0
        
        return jsonify({
            'customer_id': profile['customer_id'],
            'name': profile['name'],
            'email': profile['email'],
            'avg_transaction_amount': profile['avg_transaction_amount'],
            'common_merchants': profile['common_merchants'],
            'common_locations': profile['common_locations'],
            'transaction_count': profile['transaction_count'],
            'fraud_count': profile['fraud_count'],
            'fraud_rate': fraud_rate,
            'recent_transactions': transaction_list,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error getting customer profile: {e}")
        return jsonify({
            'error': 'Internal server error',
            'message': str(e)
        }), 500

def get_customer_stats(customer_id):
    """Get customer statistics for ML model"""
    try:
        # Query database for customer statistics
        query = """
        SELECT 
            AVG(t.amount) as amount_mean,
            STDDEV(t.amount) as amount_std,
            MAX(t.amount) as amount_max,
            COUNT(t.transaction_id) as transaction_count,
            SUM(CASE WHEN fa.alert_id IS NOT NULL THEN 1 ELSE 0 END) / COUNT(t.transaction_id) as fraud_rate
        FROM 
            transactions t
        LEFT JOIN 
            fraud_alerts fa ON t.transaction_id = fa.transaction_id
        WHERE 
            t.customer_id = %s
        """
        
        result = db.execute_raw_query(query, {'customer_id': customer_id})
        
        if not result or len(result) == 0:
            # Return default stats if customer not found
            return {
                'amount_mean': 0,
                'amount_std': 0,
                'amount_max': 0,
                'transaction_count': 0,
                'fraud_rate': 0
            }
        
        # Extract and return stats
        stats = result[0]
        return {
            'amount_mean': stats['amount_mean'] or 0,
            'amount_std': stats['amount_std'] or 0,
            'amount_max': stats['amount_max'] or 0,
            'transaction_count': stats['transaction_count'] or 0,
            'fraud_rate': stats['fraud_rate'] or 0
        }
        
    except Exception as e:
        logger.error(f"Error getting customer stats: {e}")
        # Return default stats on error
        return {
            'amount_mean': 0,
            'amount_std': 0,
            'amount_max': 0,
            'transaction_count': 0,
            'fraud_rate': 0
        }

def store_transaction(transaction, is_fraudulent, confidence, alert_type):
    """Store transaction and fraud alert in database"""
    try:
        # Store transaction
        db_transaction = {
            'transaction_id': transaction['transaction_id'],
            'customer_id': transaction['customer_id'],
            'merchant': transaction.get('merchant', ''),
            'amount': transaction['amount'],
            'transaction_time': transaction.get('transaction_time', datetime.now().isoformat()),
            'location': transaction.get('location', ''),
            'card_present': transaction.get('card_present', False),
            'currency': transaction.get('currency', 'USD')
        }
        
        db.add_transaction(db_transaction)
        
        # If fraudulent, store alert
        if is_fraudulent:
            alert = {
                'transaction_id': transaction['transaction_id'],
                'alert_type': alert_type,
                'confidence_score': confidence,
                'notes': f"Detected by: {alert_type}"
            }
            db.add_fraud_alert(alert)
            
    except Exception as e:
        logger.error(f"Error storing transaction: {e}")

def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Fraud detection API server')
    parser.add_argument('--config', default='config/config.yml', help='Path to config file')
    parser.add_argument('--host', default='0.0.0.0', help='Host to bind to')
    parser.add_argument('--port', type=int, default=5000, help='Port to bind to')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    
    args = parser.parse_args()
    
    # Load config
    global config
    config = load_config(args.config)
    
    # Get API config
    api_config = config.get('api', {})
    host = api_config.get('host', args.host)
    port = api_config.get('port', args.port)
    debug = api_config.get('debug', args.debug)
    
    # Run app
    app.run(host=host, port=port, debug=debug)

if __name__ == "__main__":
    main()
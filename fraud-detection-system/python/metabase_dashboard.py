#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Metabase Dashboard Generator for Fraud Detection System
Creates dashboard templates for Metabase
"""

import json
import requests
import yaml
import logging
import time
import sys

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("MetabaseDashboard")

class MetabaseClient:
    """Client for interacting with Metabase API"""
    
    def __init__(self, base_url="http://localhost:3000", username="admin@admin.com", password="admin"):
        """Initialize with Metabase URL and credentials"""
        self.base_url = base_url.rstrip("/")
        self.username = username
        self.password = password
        self.session_id = None
        self.database_id = None
    
    def login(self):
        """Login to Metabase and get session ID"""
        url = f"{self.base_url}/api/session"
        payload = {
            "username": self.username,
            "password": self.password
        }
        
        try:
            response = requests.post(url, json=payload)
            response.raise_for_status()
            result = response.json()
            
            if "id" in result:
                self.session_id = result["id"]
                logger.info("Successfully logged into Metabase")
                return True
            else:
                logger.error("Failed to get session ID")
                return False
                
        except Exception as e:
            logger.error(f"Login failed: {e}")
            return False
    
    def _get_headers(self):
        """Get headers with session ID for authenticated requests"""
        if not self.session_id:
            raise ValueError("Not logged in. Call login() first.")
            
        return {
            "X-Metabase-Session": self.session_id,
            "Content-Type": "application/json"
        }
    
    def get_database_id(self, db_name="fraud_detection"):
        """Get database ID by name"""
        url = f"{self.base_url}/api/database"
        
        try:
            response = requests.get(url, headers=self._get_headers())
            response.raise_for_status()
            databases = response.json()
            
            for db in databases:
                if db["name"].lower() == db_name.lower():
                    self.database_id = db["id"]
                    logger.info(f"Found database ID: {self.database_id}")
                    return self.database_id
            
            logger.warning(f"Database '{db_name}' not found")
            return None
            
        except Exception as e:
            logger.error(f"Failed to get database ID: {e}")
            return None
    
    def create_card(self, card_data):
        """Create a card (question/visualization)"""
        url = f"{self.base_url}/api/card"
        
        try:
            response = requests.post(url, json=card_data, headers=self._get_headers())
            response.raise_for_status()
            result = response.json()
            
            if "id" in result:
                logger.info(f"Created card: {result['name']} (ID: {result['id']})")
                return result
            else:
                logger.error("Failed to create card")
                return None
                
        except Exception as e:
            logger.error(f"Failed to create card: {e}")
            return None
    
    def create_dashboard(self, dashboard_data):
        """Create a dashboard"""
        url = f"{self.base_url}/api/dashboard"
        
        try:
            response = requests.post(url, json=dashboard_data, headers=self._get_headers())
            response.raise_for_status()
            result = response.json()
            
            if "id" in result:
                logger.info(f"Created dashboard: {result['name']} (ID: {result['id']})")
                return result
            else:
                logger.error("Failed to create dashboard")
                return None
                
        except Exception as e:
            logger.error(f"Failed to create dashboard: {e}")
            return None
    
    def add_card_to_dashboard(self, dashboard_id, card_id, position):
        """Add a card to a dashboard at the specified position"""
        url = f"{self.base_url}/api/dashboard/{dashboard_id}/cards"
        
        payload = {
            "cardId": card_id,
            "row": position["row"],
            "col": position["col"],
            "sizeX": position["sizeX"],
            "sizeY": position["sizeY"]
        }
        
        try:
            response = requests.post(url, json=payload, headers=self._get_headers())
            response.raise_for_status()
            result = response.json()
            
            if "id" in result:
                logger.info(f"Added card {card_id} to dashboard {dashboard_id}")
                return result
            else:
                logger.error("Failed to add card to dashboard")
                return None
                
        except Exception as e:
            logger.error(f"Failed to add card to dashboard: {e}")
            return None

def create_fraud_dashboard(client):
    """Create fraud detection dashboard in Metabase"""
    # Ensure logged in
    if not client.session_id:
        if not client.login():
            return False
    
    # Get database ID
    if not client.database_id:
        if not client.get_database_id():
            return False
    
    # Create dashboard
    dashboard = client.create_dashboard({
        "name": "Fraud Detection Dashboard",
        "description": "Real-time monitoring of fraud detection metrics"
    })
    
    if not dashboard:
        return False
    
    dashboard_id = dashboard["id"]
    
    # Create cards
    cards = []
    
    # Card 1: Fraud Rate Over Time
    fraud_rate_card = client.create_card({
        "name": "Fraud Rate Over Time",
        "display": "line",
        "visualization_settings": {
            "graph.dimensions": ["day"],
            "graph.metrics": ["fraud_rate"],
            "graph.y_axis.title_text": "Fraud Rate (%)",
            "graph.x_axis.title_text": "Date"
        },
        "dataset_query": {
            "type": "native",
            "native": {
                "query": """
                SELECT 
                    DATE(t.transaction_time) as day,
                    COUNT(*) as transaction_count,
                    SUM(CASE WHEN fa.alert_id IS NOT NULL THEN 1 ELSE 0 END) as fraud_count,
                    ROUND((SUM(CASE WHEN fa.alert_id IS NOT NULL THEN 1 ELSE 0 END) * 100.0 / COUNT(*)), 2) as fraud_rate
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
            },
            "database": client.database_id
        },
        "description": "Daily fraud rate percentage"
    })
    
    if fraud_rate_card:
        cards.append({
            "id": fraud_rate_card["id"],
            "position": {"row": 0, "col": 0, "sizeX": 8, "sizeY": 4}
        })
    
    # Card 2: Fraud by Merchant Type
    merchant_card = client.create_card({
        "name": "Fraud by Merchant",
        "display": "bar",
        "visualization_settings": {
            "graph.dimensions": ["merchant"],
            "graph.metrics": ["fraud_count"],
            "graph.y_axis.title_text": "Fraud Count",
            "graph.x_axis.title_text": "Merchant"
        },
        "dataset_query": {
            "type": "native",
            "native": {
                "query": """
                SELECT 
                    t.merchant,
                    COUNT(*) as transaction_count,
                    SUM(CASE WHEN fa.alert_id IS NOT NULL THEN 1 ELSE 0 END) as fraud_count,
                    ROUND((SUM(CASE WHEN fa.alert_id IS NOT NULL THEN 1 ELSE 0 END) * 100.0 / COUNT(*)), 2) as fraud_rate
                FROM 
                    transactions t
                LEFT JOIN 
                    fraud_alerts fa ON t.transaction_id = fa.transaction_id
                GROUP BY 
                    t.merchant
                ORDER BY 
                    fraud_count DESC
                LIMIT 10
                """
            },
            "database": client.database_id
        },
        "description": "Merchants with highest fraud counts"
    })
    
    if merchant_card:
        cards.append({
            "id": merchant_card["id"],
            "position": {"row": 0, "col": 8, "sizeX": 8, "sizeY": 4}
        })
    
    # Card 3: Recent Fraud Alerts
    alerts_card = client.create_card({
        "name": "Recent Fraud Alerts",
        "display": "table",
        "visualization_settings": {},
        "dataset_query": {
            "type": "native",
            "native": {
                "query": """
                SELECT 
                    fa.alert_id,
                    t.transaction_id,
                    t.customer_id,
                    c.name as customer_name,
                    t.amount,
                    t.merchant,
                    t.location,
                    fa.alert_type,
                    fa.confidence_score,
                    fa.alert_time
                FROM 
                    fraud_alerts fa
                JOIN 
                    transactions t ON fa.transaction_id = t.transaction_id
                LEFT JOIN 
                    customers c ON t.customer_id = c.customer_id
                ORDER BY 
                    fa.alert_time DESC
                LIMIT 20
                """
            },
            "database": client.database_id
        },
        "description": "Most recent fraud alerts"
    })
    
    if alerts_card:
        cards.append({
            "id": alerts_card["id"],
            "position": {"row": 4, "col": 0, "sizeX": 16, "sizeY": 6}
        })
    
    # Card 4: Fraud Detection Performance
    performance_card = client.create_card({
        "name": "Fraud Detection Performance",
        "display": "table",
        "visualization_settings": {},
        "dataset_query": {
            "type": "native",
            "native": {
                "query": """
                SELECT 
                    model_name,
                    version,
                    ROUND(accuracy * 100, 2) as accuracy_pct,
                    ROUND(precision * 100, 2) as precision_pct,
                    ROUND(recall * 100, 2) as recall_pct,
                    ROUND(f1_score * 100, 2) as f1_score_pct,
                    training_date
                FROM 
                    model_performance
                ORDER BY 
                    training_date DESC
                LIMIT 10
                """
            },
            "database": client.database_id
        },
        "description": "ML model performance metrics"
    })
    
    if performance_card:
        cards.append({
            "id": performance_card["id"],
            "position": {"row": 10, "col": 0, "sizeX": 8, "sizeY": 4}
        })
    
    # Card 5: Amount Distribution for Fraud vs. Non-Fraud
    amount_card = client.create_card({
        "name": "Transaction Amount Distribution",
        "display": "bar",
        "visualization_settings": {
            "graph.dimensions": ["amount_bucket"],
            "graph.metrics": ["fraud_count", "normal_count"],
            "graph.y_axis.title_text": "Transaction Count",
            "graph.x_axis.title_text": "Amount Range"
        },
        "dataset_query": {
            "type": "native",
            "native": {
                "query": """
                WITH amount_buckets AS (
                    SELECT 
                        t.transaction_id,
                        CASE 
                            WHEN t.amount < 100 THEN 'Under $100'
                            WHEN t.amount < 500 THEN '$100-$500'
                            WHEN t.amount < 1000 THEN '$500-$1000'
                            WHEN t.amount < 5000 THEN '$1000-$5000'
                            ELSE 'Over $5000'
                        END as amount_bucket,
                        CASE WHEN fa.alert_id IS NOT NULL THEN 1 ELSE 0 END as is_fraud
                    FROM 
                        transactions t
                    LEFT JOIN 
                        fraud_alerts fa ON t.transaction_id = fa.transaction_id
                )
                SELECT 
                    amount_bucket,
                    SUM(CASE WHEN is_fraud = 1 THEN 1 ELSE 0 END) as fraud_count,
                    SUM(CASE WHEN is_fraud = 0 THEN 1 ELSE 0 END) as normal_count
                FROM 
                    amount_buckets
                GROUP BY 
                    amount_bucket
                ORDER BY 
                    CASE 
                        WHEN amount_bucket = 'Under $100' THEN 1
                        WHEN amount_bucket = '$100-$500' THEN 2
                        WHEN amount_bucket = '$500-$1000' THEN 3
                        WHEN amount_bucket = '$1000-$5000' THEN 4
                        ELSE 5
                    END
                """
            },
            "database": client.database_id
        },
        "description": "Distribution of transaction amounts for fraudulent vs. legitimate transactions"
    })
    
    if amount_card:
        cards.append({
            "id": amount_card["id"],
            "position": {"row": 10, "col": 8, "sizeX": 8, "sizeY": 4}
        })
    
    # Add cards to dashboard
    for card in cards:
        client.add_card_to_dashboard(dashboard_id, card["id"], card["position"])
    
    logger.info(f"Dashboard creation complete. Access at {client.base_url}/dashboard/{dashboard_id}")
    return True

def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Create Metabase dashboard for fraud detection')
    parser.add_argument('--url', default='http://localhost:3000', help='Metabase URL')
    parser.add_argument('--username', default='admin@admin.com', help='Metabase username')
    parser.add_argument('--password', default='admin', help='Metabase password')
    parser.add_argument('--wait', type=int, default=0, help='Wait seconds before connecting (for startup)')
    
    args = parser.parse_args()
    
    # Wait if specified (useful when starting containers)
    if args.wait > 0:
        logger.info(f"Waiting {args.wait} seconds for Metabase to start...")
        time.sleep(args.wait)
    
    # Create client and dashboard
    client = MetabaseClient(
        base_url=args.url,
        username=args.username,
        password=args.password
    )
    
    try:
        if create_fraud_dashboard(client):
            logger.info("Dashboard created successfully!")
            return 0
        else:
            logger.error("Failed to create dashboard")
            return 1
    except Exception as e:
        logger.error(f"Error creating dashboard: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
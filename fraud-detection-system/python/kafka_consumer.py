#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Kafka Consumer for Fraud Detection System
Consumes transaction messages from Kafka and processes them
"""

import json
import yaml
import logging
import sys
import time
import signal
from datetime import datetime
from kafka import KafkaConsumer, KafkaProducer
from database import DatabaseManager

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("consumer.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("KafkaConsumer")

class TransactionConsumer:
    """Consumes transaction messages from Kafka"""
    
    def __init__(self, config_path="config/config.yml"):
        """Initialize the consumer with configuration"""
        self.config = self._load_config(config_path)
        self.consumer = self._create_consumer()
        self.producer = self._create_producer()
        self.db = DatabaseManager(config_path)
        self.running = True
        
        # Register signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _load_config(self, config_path):
        """Load configuration from YAML file"""
        try:
            with open(config_path, 'r') as config_file:
                return yaml.safe_load(config_file)
        except Exception as e:
            logger.error(f"Failed to load config: {e}")
            # Default configuration
            return {
                "kafka": {
                    "bootstrap_servers": "localhost:9092",
                    "group_id": "fraud-detection-group",
                    "topics": {
                        "transactions": "transactions",
                        "fraud_alerts": "fraud-alerts",
                        "processed_transactions": "processed-transactions"
                    }
                }
            }
    
    def _create_consumer(self):
        """Create and return a Kafka consumer"""
        try:
            kafka_config = self.config.get("kafka", {})
            bootstrap_servers = kafka_config.get("bootstrap_servers", "localhost:9092")
            group_id = kafka_config.get("group_id", "fraud-detection-group")
            transaction_topic = kafka_config.get("topics", {}).get("transactions", "transactions")
            
            consumer = KafkaConsumer(
                transaction_topic,
                bootstrap_servers=bootstrap_servers,
                group_id=group_id,
                auto_offset_reset='latest',
                value_deserializer=lambda m: json.loads(m.decode('utf-8')),
                # Enable auto commit for simplicity, but consider manual commit for production
                enable_auto_commit=True,
                auto_commit_interval_ms=5000
            )
            
            logger.info(f"Created Kafka consumer for topic: {transaction_topic}")
            return consumer
            
        except Exception as e:
            logger.error(f"Failed to create Kafka consumer: {e}")
            raise
    
    def _create_producer(self):
        """Create and return a Kafka producer for sending processed transactions and alerts"""
        try:
            kafka_config = self.config.get("kafka", {})
            bootstrap_servers = kafka_config.get("bootstrap_servers", "localhost:9092")
            
            producer = KafkaProducer(
                bootstrap_servers=bootstrap_servers,
                value_serializer=lambda v: json.dumps(v).encode('utf-8')
            )
            
            logger.info(f"Created Kafka producer")
            return producer
            
        except Exception as e:
            logger.error(f"Failed to create Kafka producer: {e}")
            return None
    
    def _signal_handler(self, sig, frame):
        """Handle termination signals"""
        logger.info(f"Received signal {sig}, shutting down...")
        self.running = False
    
    def process_transaction(self, transaction):
        """
        Process a transaction and apply basic fraud detection rules
        
        In a real system, this would be handled by Flink, but this simple
        implementation demonstrates the concept.
        """
        # Make a copy to avoid modifying the original
        processed = transaction.copy()
        
        # Extract transaction details
        amount = processed.get('amount', 0)
        location = processed.get('location', '')
        transaction_time = processed.get('transaction_time', '')
        customer_id = processed.get('customer_id', '')
        
        # Check for obvious fraud signals
        rules_config = self.config.get("rules", {})
        amount_threshold = rules_config.get("amount_threshold", 5000)
        
        # Flag for potential fraud
        is_fraudulent = False
        alert_type = None
        confidence_score = 0.0
        
        # Simple rule: Large transaction amount
        if amount > amount_threshold:
            is_fraudulent = True
            alert_type = "large_amount"
            confidence_score = min(0.5 + (amount / amount_threshold - 1) * 0.1, 0.95)
        
        # Add ML model prediction flag (would be set by ML model)
        if 'is_fraudulent' in processed and processed['is_fraudulent']:
            is_fraudulent = True
            alert_type = processed.get('fraud_pattern', 'unknown')
            confidence_score = 0.9  # High confidence for simulated fraud
        
        # Add fraud detection results to the transaction
        processed['fraud_detected'] = is_fraudulent
        processed['alert_type'] = alert_type
        processed['confidence_score'] = confidence_score
        processed['processing_time'] = datetime.now().isoformat()
        
        return processed
    
    def store_transaction(self, transaction):
        """Store transaction in the database"""
        try:
            # Store in database (simplify the transaction for storage)
            db_transaction = {
                'transaction_id': transaction.get('transaction_id'),
                'customer_id': transaction.get('customer_id'),
                'merchant': transaction.get('merchant'),
                'amount': transaction.get('amount'),
                'transaction_time': transaction.get('transaction_time'),
                'location': transaction.get('location'),
                'card_present': transaction.get('card_present'),
                'currency': transaction.get('currency')
            }
            
            # Add to database
            self.db.add_transaction(db_transaction)
            
            # If fraudulent, create an alert
            if transaction.get('fraud_detected'):
                alert = {
                    'transaction_id': transaction.get('transaction_id'),
                    'alert_type': transaction.get('alert_type'),
                    'confidence_score': transaction.get('confidence_score'),
                    'notes': f"Detected by rule: {transaction.get('alert_type')}"
                }
                self.db.add_fraud_alert(alert)
                
        except Exception as e:
            logger.error(f"Error storing transaction: {e}")
    
    def send_to_kafka(self, transaction):
        """Send processed transaction to appropriate Kafka topics"""
        if not self.producer:
            return
            
        try:
            # Topics from config
            kafka_config = self.config.get("kafka", {})
            processed_topic = kafka_config.get("topics", {}).get(
                "processed_transactions", "processed-transactions"
            )
            alert_topic = kafka_config.get("topics", {}).get(
                "fraud_alerts", "fraud-alerts"
            )
            
            # Send to processed transactions topic
            self.producer.send(processed_topic, transaction)
            
            # If fraudulent, also send to fraud alerts topic
            if transaction.get('fraud_detected'):
                self.producer.send(alert_topic, transaction)
                
        except Exception as e:
            logger.error(f"Error sending to Kafka: {e}")
    
    def run(self):
        """Main consumer loop"""
        logger.info("Starting Kafka consumer...")
        
        try:
            # Process messages
            while self.running:
                # Poll for messages with timeout
                messages = self.consumer.poll(timeout_ms=1000)
                
                if not messages:
                    continue
                
                # Process each message
                for topic_partition, records in messages.items():
                    for record in records:
                        try:
                            # Extract transaction
                            transaction = record.value
                            
                            logger.debug(f"Received transaction: {transaction['transaction_id']}")
                            
                            # Process transaction
                            processed = self.process_transaction(transaction)
                            
                            # Store in database
                            self.store_transaction(processed)
                            
                            # Send to Kafka
                            self.send_to_kafka(processed)
                            
                            if processed.get('fraud_detected'):
                                logger.info(f"Fraud detected: {processed['transaction_id']} "
                                           f"({processed['alert_type']}, "
                                           f"Score: {processed['confidence_score']:.2f})")
                                
                        except Exception as e:
                            logger.error(f"Error processing message: {e}")
                
        except Exception as e:
            logger.error(f"Consumer error: {e}")
            
        finally:
            # Cleanup
            logger.info("Shutting down consumer...")
            if self.consumer:
                self.consumer.close()
            if self.producer:
                self.producer.flush()
                self.producer.close()
            logger.info("Consumer shutdown complete")

def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Kafka consumer for transaction processing')
    parser.add_argument('--config', default='config/config.yml', help='Path to config file')
    
    args = parser.parse_args()
    
    # Create and run consumer
    consumer = TransactionConsumer(config_path=args.config)
    consumer.run()

if __name__ == "__main__":
    main()
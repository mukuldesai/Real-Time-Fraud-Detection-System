#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Database Utility for Fraud Detection System
Handles database connections and operations
"""

import os
import yaml
import logging
import psycopg2
from psycopg2.extras import RealDictCursor
from sqlalchemy import create_engine, MetaData, Table, Column, Integer, String, Float, DateTime, Boolean, Text, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Database")

Base = declarative_base()

# Define ORM Models
class Customer(Base):
    __tablename__ = 'customers'
    
    customer_id = Column(String(36), primary_key=True)
    name = Column(String(100))
    email = Column(String(100))
    created_at = Column(DateTime, default=datetime.now)
    updated_at = Column(DateTime, default=datetime.now, onupdate=datetime.now)
    
    # Relationships
    transactions = relationship("Transaction", back_populates="customer")
    profile = relationship("CustomerProfile", back_populates="customer", uselist=False)
    
    def __repr__(self):
        return f"<Customer(id={self.customer_id}, name={self.name})>"

class Transaction(Base):
    __tablename__ = 'transactions'
    
    transaction_id = Column(String(36), primary_key=True)
    customer_id = Column(String(36), ForeignKey('customers.customer_id'))
    merchant = Column(String(100))
    amount = Column(Float, nullable=False)
    transaction_time = Column(DateTime, nullable=False)
    location = Column(String(100))
    card_present = Column(Boolean)
    currency = Column(String(3))
    created_at = Column(DateTime, default=datetime.now)
    
    # Relationships
    customer = relationship("Customer", back_populates="transactions")
    fraud_alerts = relationship("FraudAlert", back_populates="transaction")
    
    def __repr__(self):
        return f"<Transaction(id={self.transaction_id}, amount={self.amount})>"

class FraudAlert(Base):
    __tablename__ = 'fraud_alerts'
    
    alert_id = Column(Integer, primary_key=True)
    transaction_id = Column(String(36), ForeignKey('transactions.transaction_id'))
    alert_time = Column(DateTime, default=datetime.now)
    alert_type = Column(String(50))
    confidence_score = Column(Float)
    is_confirmed = Column(Boolean, default=False)
    notes = Column(Text)
    
    # Relationships
    transaction = relationship("Transaction", back_populates="fraud_alerts")
    
    def __repr__(self):
        return f"<FraudAlert(id={self.alert_id}, type={self.alert_type})>"

class CustomerProfile(Base):
    __tablename__ = 'customer_profiles'
    
    profile_id = Column(Integer, primary_key=True)
    customer_id = Column(String(36), ForeignKey('customers.customer_id'))
    avg_transaction_amount = Column(Float)
    avg_transaction_frequency = Column(String)  # Stored as interval string
    common_merchants = Column(Text)  # Stored as JSON
    common_locations = Column(Text)  # Stored as JSON
    last_updated = Column(DateTime, default=datetime.now)
    
    # Relationships
    customer = relationship("Customer", back_populates="profile")
    
    def __repr__(self):
        return f"<CustomerProfile(customer_id={self.customer_id})>"

class ModelPerformance(Base):
    __tablename__ = 'model_performance'
    
    id = Column(Integer, primary_key=True)
    model_name = Column(String(100))
    version = Column(String(20))
    accuracy = Column(Float)
    precision = Column(Float)
    recall = Column(Float)
    f1_score = Column(Float)
    training_date = Column(DateTime)
    notes = Column(Text)
    
    def __repr__(self):
        return f"<ModelPerformance(model={self.model_name}, version={self.version})>"

class DatabaseManager:
    """Manages database connections and operations for the fraud detection system"""
    
    def __init__(self, config_path="config/config.yml"):  # Change from ../config/config.yml
        """Initialize database manager with configuration"""
        self.config = self._load_config(config_path)
        self.engine = self._create_engine()
        self.session_maker = sessionmaker(bind=self.engine)
        
    def _load_config(self, config_path):
        """Load configuration from YAML file"""
        try:
            with open(config_path, 'r') as config_file:
                return yaml.safe_load(config_file)
        except Exception as e:
            logger.error(f"Failed to load config: {e}")
            # Default configuration
            return {
                "database": {
                    "type": "postgresql",
                    "host": "localhost",
                    "port": 5432,
                    "username": "postgres",
                    "password": "postgres",
                    "database": "fraud_detection"
                }
            }
    
    def _create_engine(self):
        """Create SQLAlchemy engine based on configuration"""
        db_config = self.config.get("database", {})
        
        # Get connection parameters
        host = db_config.get("host", "localhost")
        port = db_config.get("port", 5432)
        username = db_config.get("username", "postgres")
        password = db_config.get("password", "postgres")
        database = db_config.get("database", "fraud_detection")
        
        # Create a more explicit connection string
        # Be sure to URL-encode the password in case it contains special characters
        import urllib.parse
        encoded_password = urllib.parse.quote_plus(password)
        connection_string = f"postgresql+psycopg2://{username}:{encoded_password}@{host}:{port}/{database}"
        
        logger.info(f"Connecting to database: postgresql+psycopg2://{username}:******@{host}:{port}/{database}")
        
        try:
            # Add echo=True to see SQL queries for debugging
            return create_engine(connection_string, echo=False, pool_pre_ping=True)
        except Exception as e:
            logger.error(f"Failed to create database engine: {e}")
            raise
    
    def create_tables(self):
        """Create all database tables if they don't exist"""
        try:
            Base.metadata.create_all(self.engine)
            logger.info("Database tables created successfully")
        except Exception as e:
            logger.error(f"Failed to create database tables: {e}")
            raise
    
    def get_session(self):
        """Get a new database session"""
        return self.session_maker()
    
    def add_customer(self, customer_data):
        """Add a new customer to the database"""
        session = self.get_session()
        try:
            customer = Customer(**customer_data)
            session.add(customer)
            session.commit()
            logger.info(f"Added customer: {customer.customer_id}")
            return customer
        except Exception as e:
            session.rollback()
            logger.error(f"Failed to add customer: {e}")
            raise
        finally:
            session.close()
    
    def add_transaction(self, transaction_data):
        """Add a new transaction to the database"""
        session = self.get_session()
        try:
            # Handle datetime conversion if it's a string
            if isinstance(transaction_data.get('transaction_time'), str):
                transaction_data['transaction_time'] = datetime.fromisoformat(
                    transaction_data['transaction_time'].replace('Z', '+00:00')
                )
                
            transaction = Transaction(**transaction_data)
            session.add(transaction)
            session.commit()
            logger.debug(f"Added transaction: {transaction.transaction_id}")
            return transaction
        except Exception as e:
            session.rollback()
            logger.error(f"Failed to add transaction: {e}")
            raise
        finally:
            session.close()
    
    def add_fraud_alert(self, alert_data):
        """Add a new fraud alert to the database"""
        session = self.get_session()
        try:
            alert = FraudAlert(**alert_data)
            session.add(alert)
            session.commit()
            logger.info(f"Added fraud alert: {alert.alert_id}")
            return alert
        except Exception as e:
            session.rollback()
            logger.error(f"Failed to add fraud alert: {e}")
            raise
        finally:
            session.close()
    
    def get_customer_transactions(self, customer_id, limit=100):
        """Get recent transactions for a customer"""
        session = self.get_session()
        try:
            transactions = session.query(Transaction).\
                filter(Transaction.customer_id == customer_id).\
                order_by(Transaction.transaction_time.desc()).\
                limit(limit).all()
            return transactions
        except Exception as e:
            logger.error(f"Failed to fetch customer transactions: {e}")
            raise
        finally:
            session.close()
    
    def get_recent_fraud_alerts(self, limit=100):
        """Get recent fraud alerts"""
        session = self.get_session()
        try:
            alerts = session.query(FraudAlert).\
                order_by(FraudAlert.alert_time.desc()).\
                limit(limit).all()
            return alerts
        except Exception as e:
            logger.error(f"Failed to fetch recent fraud alerts: {e}")
            raise
        finally:
            session.close()
    
    def execute_raw_query(self, query, params=None):
        """Execute a raw SQL query"""
        conn = self.engine.raw_connection()
        try:
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            cursor.execute(query, params or {})
            results = cursor.fetchall()
            cursor.close()
            conn.commit()
            return results
        except Exception as e:
            conn.rollback()
            logger.error(f"Failed to execute query: {e}")
            raise
        finally:
            conn.close()

# Example usage
if __name__ == "__main__":
    # This will create the database tables when run directly
    db_manager = DatabaseManager()
    db_manager.create_tables()
    print("Database initialized successfully!")
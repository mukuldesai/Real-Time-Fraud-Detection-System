#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
ML Model Predictor for Fraud Detection System
Loads and applies ML models to detect fraud in transactions
"""

import os
import yaml
import json
import logging
import numpy as np
import pandas as pd
from datetime import datetime
import joblib

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ModelPredictor")

class FraudModelPredictor:
    """Applies trained ML models to detect fraud"""
    
    def __init__(self, config_path="config/config.yml"):
        """Initialize with configuration"""
        self.config = self._load_config(config_path)
        self.model = None
        self.feature_columns = None
        self.load_model()
        
    def _load_config(self, config_path):
        """Load configuration from YAML file"""
        try:
            with open(config_path, 'r') as config_file:
                return yaml.safe_load(config_file)
        except Exception as e:
            logger.error(f"Failed to load config: {e}")
            # Default configuration
            return {
                "model": {
                    "path": "../models",
                    "default_model": "xgboost_fraud_model.joblib",
                    "threshold": 0.7
                }
            }
    
    def load_model(self):
        """Load the trained model and feature information"""
        model_config = self.config.get("model", {})
        model_dir = model_config.get("path", "../models")
        model_file = model_config.get("default_model", "xgboost_fraud_model.joblib")
        model_path = os.path.join(model_dir, model_file)
        
        # Load feature columns
        feature_path = os.path.join(model_dir, "feature_columns.json")
        
        try:
            # Load model
            if os.path.exists(model_path):
                self.model = joblib.load(model_path)
                logger.info(f"Model loaded from {model_path}")
            else:
                logger.warning(f"Model file not found: {model_path}")
            
            # Load feature columns
            if os.path.exists(feature_path):
                with open(feature_path, 'r') as f:
                    self.feature_columns = json.load(f)
                logger.info(f"Feature columns loaded from {feature_path}")
            else:
                logger.warning(f"Feature columns file not found: {feature_path}")
                # Default feature columns if file not found
                self.feature_columns = [
                    'amount', 'amount_log', 'hour', 'day_of_week', 'is_weekend',
                    'card_present', 'amount_mean', 'amount_std', 'amount_max',
                    'transaction_count', 'fraud_rate', 'amount_diff_ratio'
                ]
                
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            self.model = None
    
    def prepare_transaction_features(self, transaction, customer_stats=None):
        """
        Prepare features for a single transaction
        
        Args:
            transaction: Dictionary with transaction data
            customer_stats: Optional dictionary with customer statistics
                            (if not provided, will use defaults)
        
        Returns:
            pandas.DataFrame: DataFrame with features for prediction
        """
        # Create a dataframe with a single row
        df = pd.DataFrame([transaction])
        
        # Add customer statistics if provided
        if customer_stats:
            for key, value in customer_stats.items():
                df[key] = value
        else:
            # Default values when customer stats not available
            df['amount_mean'] = transaction.get('amount', 0)
            df['amount_std'] = 0
            df['amount_max'] = transaction.get('amount', 0)
            df['transaction_count'] = 1
            df['fraud_rate'] = 0
        
        # Convert transaction_time to datetime if it's a string
        if 'transaction_time' in df and df['transaction_time'].dtype == 'object':
            df['transaction_time'] = pd.to_datetime(df['transaction_time'])
        
        # Extract time features
        if 'transaction_time' in df:
            df['hour'] = df['transaction_time'].dt.hour
            df['day_of_week'] = df['transaction_time'].dt.dayofweek
            df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        
        # Amount features
        if 'amount' in df:
            df['amount_log'] = np.log1p(df['amount'])
            
            if 'amount_mean' in df and df['amount_mean'].iloc[0] > 0:
                df['amount_diff_ratio'] = df['amount'] / df['amount_mean']
            else:
                df['amount_diff_ratio'] = 1.0
        
        # Card present feature
        if 'card_present' in df and df['card_present'].dtype == bool:
            df['card_present'] = df['card_present'].astype(int)
        
        # Ensure all required columns are present
        for col in self.feature_columns:
            if col not in df.columns:
                df[col] = 0  # Default value for missing features
        
        # Select only the required features in the correct order
        return df[self.feature_columns]
    
    def predict(self, transaction, customer_stats=None):
        """
        Predict if a transaction is fraudulent
        
        Args:
            transaction: Dictionary with transaction data
            customer_stats: Optional dictionary with customer statistics
        
        Returns:
            tuple: (is_fraudulent, confidence_score, alert_type)
        """
        if self.model is None:
            logger.warning("No model loaded, cannot make prediction")
            return False, 0.0, None
        
        try:
            # Prepare features
            features = self.prepare_transaction_features(transaction, customer_stats)
            
            # Make prediction
            prediction_proba = self.model.predict_proba(features)[0, 1]
            
            # Get threshold from config
            threshold = self.config.get("model", {}).get("threshold", 0.7)
            
            # Determine if fraudulent
            is_fraudulent = prediction_proba >= threshold
            
            return is_fraudulent, prediction_proba, "ml_model"
            
        except Exception as e:
            logger.error(f"Error during prediction: {e}")
            return False, 0.0, None

# For testing
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
    
    # Customer statistics (in a real system, these would come from the database)
    customer_stats = {
        'amount_mean': 500,
        'amount_std': 250,
        'amount_max': 2000,
        'transaction_count': 20,
        'fraud_rate': 0.05
    }
    
    # Create predictor and test
    predictor = FraudModelPredictor()
    is_fraud, score, alert_type = predictor.predict(test_transaction, customer_stats)
    
    print(f"Transaction evaluation:")
    print(f"  Fraud detected: {is_fraud}")
    print(f"  Confidence score: {score:.4f}")
    print(f"  Alert type: {alert_type}")
#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
ML Model Trainer for Fraud Detection System
Trains and saves machine learning models for fraud detection
"""

import os
import yaml
import json
import logging
import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, precision_recall_curve
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
import xgboost as xgb
import joblib

from database import DatabaseManager

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("ModelTrainer")

class FraudModelTrainer:
    """Trains ML models for fraud detection"""
    
    # Change this initialization method
    def __init__(self, config_path="config/config.yml"):  # Change from ../config/config.yml
        """Initialize with configuration"""
        self.config = self._load_config(config_path)
        self.db = DatabaseManager(config_path)
        
        # Ensure model directory exists
        model_dir = self.config.get("model", {}).get("path", "models")  # Remove the leading ../
        os.makedirs(model_dir, exist_ok=True)
        
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
                    "default_model": "xgboost_fraud_model.joblib"
                }
            }
    
    def load_training_data(self, data_path=None):
        """
        Load training data from CSV file or database
        
        In a real system, you would typically extract data from your database.
        For simplicity, this example can also load from a CSV file.
        """
        if data_path and os.path.exists(data_path):
            # Load from CSV
            logger.info(f"Loading training data from {data_path}")
            return pd.read_csv(data_path)
            
        else:
            # Load from database
            logger.info("Loading training data from database")
            
            # In a real system, you'd use a more complex query tailored to your schema
            # This is a simplified example
            query = """
            SELECT 
                t.transaction_id,
                t.customer_id,
                t.merchant,
                t.amount,
                t.transaction_time,
                t.location,
                t.card_present,
                t.currency,
                CASE WHEN fa.alert_id IS NOT NULL THEN true ELSE false END as is_fraud
            FROM 
                transactions t
            LEFT JOIN 
                fraud_alerts fa ON t.transaction_id = fa.transaction_id
            ORDER BY 
                t.transaction_time DESC
            LIMIT 10000
            """
            
            results = self.db.execute_raw_query(query)
            
            if not results:
                logger.warning("No training data found in database")
                return None
                
            # Convert to DataFrame
            df = pd.DataFrame(results)
            logger.info(f"Loaded {len(df)} transactions from database")
            return df
    
    def prepare_features(self, df):
        """Extract and transform features for model training"""
        logger.info("Preparing features for model training")
        
        if df is None or len(df) == 0:
            logger.error("No data available for feature preparation")
            return None, None, None
        
        # Convert transaction_time to datetime if it's a string
        if df['transaction_time'].dtype == 'object':
            df['transaction_time'] = pd.to_datetime(df['transaction_time'])
        
        # Time-based features
        df['hour'] = df['transaction_time'].dt.hour
        df['day_of_week'] = df['transaction_time'].dt.dayofweek
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        
        # Amount-based features
        df['amount_log'] = np.log1p(df['amount'])  # log transformation for skewed amount
        
        # Card presence feature (already binary)
        if 'card_present' in df.columns:
            df['card_present'] = df['card_present'].astype(int)
        
        # Customer-based aggregations
        # Calculate statistics per customer
        customer_stats = df.groupby('customer_id').agg({
            'amount': ['mean', 'std', 'max', 'count'],
            'is_fraud': ['mean']  # Fraud rate per customer
        })
        
        # Flatten column names
        customer_stats.columns = ['_'.join(col).strip() for col in customer_stats.columns.values]
        customer_stats = customer_stats.reset_index()
        
        # Rename for clarity
        customer_stats.rename(columns={
            'amount_count': 'transaction_count',
            'is_fraud_mean': 'fraud_rate'
        }, inplace=True)
        
        # Merge back to the original dataframe
        df = pd.merge(df, customer_stats, on='customer_id', how='left')
        
        # Calculate how much this transaction differs from the customer's mean amount
        df['amount_diff_ratio'] = df['amount'] / df['amount_mean']
        
        # Replace infinity and NaN with appropriate values
        df['amount_diff_ratio'].replace([np.inf, -np.inf], 10.0, inplace=True)
        df.fillna(0, inplace=True)
        
        # Define features and target variable
        feature_cols = [
            'amount', 'amount_log', 'hour', 'day_of_week', 'is_weekend',
            'card_present', 'amount_mean', 'amount_std', 'amount_max',
            'transaction_count', 'fraud_rate', 'amount_diff_ratio'
        ]
        
        # Add location and merchant features if available
        if 'location' in df.columns and df['location'].notna().sum() > 0:
            # Extract country from location for simplicity
            # In a real system, you might use geolocation features
            df['country'] = df['location'].str.split(',').str[-1].str.strip()
            feature_cols.append('country')
        
        # Select only available columns
        feature_cols = [col for col in feature_cols if col in df.columns]
        
        # Split features and target
        X = df[feature_cols]
        y = df['is_fraud'].astype(int)
        
        # For categorical variables, we'd typically use one-hot encoding
        # For simplicity, we're not doing that here for all variables
        
        # Return processed data
        return X, y, feature_cols
    
    def train_model(self, X, y, model_type='xgboost'):
        """Train a fraud detection model"""
        if X is None or y is None:
            logger.error("No data available for model training")
            return None
            
        logger.info(f"Training {model_type} model...")
        
        # Split data into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Handle class imbalance with SMOTE
        # Only apply if there are enough samples of minority class
        if len(y_train) > 100 and y_train.sum() >= 10:
            logger.info("Applying SMOTE for class imbalance")
            smote = SMOTE(random_state=42)
            
            # For categorical variables, we'd need to be more careful with SMOTE
            # This is a simplified implementation
            try:
                X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
                logger.info(f"SMOTE applied. Training samples: {len(X_train_resampled)}")
                X_train, y_train = X_train_resampled, y_train_resampled
            except Exception as e:
                logger.warning(f"SMOTE failed, proceeding with original data: {e}")
        
        # Identify numeric and categorical columns
        numeric_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
        categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
        
        # Create preprocessing pipeline
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), numeric_cols),
                ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
            ]
        )
        
        # Choose model based on type
        if model_type == 'random_forest':
            model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=10,
                random_state=42,
                class_weight='balanced'
            )
        elif model_type == 'logistic':
            model = LogisticRegression(
                class_weight='balanced',
                max_iter=1000,
                random_state=42
            )
        else:  # xgboost
            model = xgb.XGBClassifier(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                objective='binary:logistic',
                scale_pos_weight=sum(y_train == 0) / sum(y_train == 1),  # Balance classes
                random_state=42
            )
        
        # Create pipeline
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', model)
        ])
        
        # Train model
        pipeline.fit(X_train, y_train)
        
        # Evaluate model
        y_pred = pipeline.predict(X_test)
        y_prob = pipeline.predict_proba(X_test)[:, 1]
        
        # Print evaluation metrics
        logger.info("\nModel Evaluation:")
        logger.info(classification_report(y_test, y_pred))
        
        # Calculate AUC
        auc = roc_auc_score(y_test, y_prob)
        logger.info(f"ROC AUC: {auc:.4f}")
        
        # Plot confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Confusion Matrix')
        plt.savefig('../models/confusion_matrix.png')
        
        # Calculate and store model performance
        results = {
            'model_type': model_type,
            'accuracy': (y_pred == y_test).mean(),
            'precision': (y_pred & y_test).sum() / y_pred.sum() if y_pred.sum() > 0 else 0,
            'recall': (y_pred & y_test).sum() / y_test.sum() if y_test.sum() > 0 else 0,
            'f1_score': classification_report(y_test, y_pred, output_dict=True)['1']['f1-score'],
            'auc': auc,
            'training_date': datetime.now().isoformat(),
            'test_size': len(y_test),
            'fraud_ratio': y.mean()
        }
        
        # Save results
        with open(f'../models/{model_type}_performance.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        # Store in database if available
        try:
            self.db.execute_raw_query(
                """
                INSERT INTO model_performance 
                (model_name, version, accuracy, precision, recall, f1_score, training_date, notes)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    model_type, 
                    datetime.now().strftime('%Y%m%d'),
                    results['accuracy'],
                    results['precision'],
                    results['recall'],
                    results['f1_score'],
                    datetime.now(),
                    f"Training data size: {len(X)}, Fraud ratio: {y.mean():.4f}"
                )
            )
        except Exception as e:
            logger.warning(f"Failed to store model performance in database: {e}")
        
        return pipeline
    
    def save_model(self, model, model_type='xgboost'):
        """Save trained model to disk"""
        model_dir = self.config.get("model", {}).get("path", "../models")
        model_path = os.path.join(model_dir, f"{model_type}_fraud_model.joblib")
        
        logger.info(f"Saving model to {model_path}")
        joblib.dump(model, model_path)
        
        # Save a default model copy if specified
        default_model = self.config.get("model", {}).get("default_model")
        if default_model:
            default_path = os.path.join(model_dir, default_model)
            joblib.dump(model, default_path)
            logger.info(f"Saved copy as default model: {default_path}")
    
    def run(self, data_path=None, model_type='xgboost'):
        """Run the full model training pipeline"""
        # Load data
        df = self.load_training_data(data_path)
        
        # Prepare features
        X, y, feature_cols = self.prepare_features(df)
        
        # Save feature columns for future reference
        if feature_cols:
            with open('../models/feature_columns.json', 'w') as f:
                json.dump(feature_cols, f)
        
        # Train model
        model = self.train_model(X, y, model_type)
        
        # Save model
        if model:
            self.save_model(model, model_type)
            logger.info(f"Model training completed successfully")
            return True
        else:
            logger.error("Model training failed")
            return False

def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Train fraud detection model')
    parser.add_argument('--config', default='config/config.yml', help='Path to config file')  # Change from ../config/config.yml
    parser.add_argument('--data', help='Path to training data CSV (optional)')
    parser.add_argument('--model', default='xgboost', choices=['xgboost', 'random_forest', 'logistic'],
                       help='Model type to train')
    
    args = parser.parse_args()
    
    # Create and run trainer
    trainer = FraudModelTrainer(config_path=args.config)
    success = trainer.run(data_path=args.data, model_type=args.model)
    
    if success:
        print("Model training completed successfully")
    else:
        print("Model training failed")
        exit(1)

if __name__ == "__main__":
    main()
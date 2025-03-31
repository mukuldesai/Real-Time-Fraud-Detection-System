import pandas as pd
import psycopg2

# Load the CSV file with sample transactions
df = pd.read_csv('../data/sample_transactions.csv')

# Connect to PostgreSQL
conn = psycopg2.connect(
    dbname="fraud_detection",
    user="postgres",
    password="abcd@1234",
    host="localhost"
)
cursor = conn.cursor()

# For each transaction, insert into database
for _, row in df.iterrows():
    # Construct the INSERT statement
    cursor.execute("""
    INSERT INTO transactions (
        transaction_id, customer_id, merchant, amount, 
        transaction_time, location, card_present, currency
    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
    """, (
        row['transaction_id'],
        row['customer_id'], 
        row['merchant'],
        row['amount'],
        row['transaction_time'],
        row['location'],
        row['card_present'],
        row['currency']
    ))
    
    # If it's fraudulent, also create an alert
    if row.get('is_fraudulent', False):
        cursor.execute("""
        INSERT INTO fraud_alerts (
            transaction_id, alert_type, confidence_score, notes
        ) VALUES (%s, %s, %s, %s)
        """, (
            row['transaction_id'],
            row.get('fraud_pattern', 'unknown'),
            0.85,  # Default confidence score
            f"Fraud pattern: {row.get('fraud_pattern', 'unknown')}"
        ))

# Commit changes and close connection
conn.commit()
cursor.close()
conn.close()
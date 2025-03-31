-- Create Database (if not exists)
-- CREATE DATABASE fraud_detection;

-- Connect to the database
\c fraud_detection;

-- Customer table
CREATE TABLE IF NOT EXISTS customers (
    customer_id VARCHAR(36) PRIMARY KEY,
    name VARCHAR(100),
    email VARCHAR(100),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Transaction table
CREATE TABLE IF NOT EXISTS transactions (
    transaction_id VARCHAR(36) PRIMARY KEY,
    customer_id VARCHAR(36) REFERENCES customers(customer_id),
    merchant VARCHAR(100),
    amount DECIMAL(12, 2) NOT NULL,
    transaction_time TIMESTAMP NOT NULL,
    location VARCHAR(100),
    card_present BOOLEAN,
    currency VARCHAR(3),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Fraud alerts table
CREATE TABLE IF NOT EXISTS fraud_alerts (
    alert_id SERIAL PRIMARY KEY,
    transaction_id VARCHAR(36) REFERENCES transactions(transaction_id),
    alert_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    alert_type VARCHAR(50),
    confidence_score DECIMAL(5, 4),
    is_confirmed BOOLEAN DEFAULT FALSE,
    notes TEXT
);

-- Customer profiles table (for ML features)
CREATE TABLE IF NOT EXISTS customer_profiles (
    profile_id SERIAL PRIMARY KEY,
    customer_id VARCHAR(36) REFERENCES customers(customer_id),
    avg_transaction_amount DECIMAL(12, 2),
    avg_transaction_frequency INTERVAL,
    common_merchants TEXT[],
    common_locations TEXT[],
    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Model performance tracking
CREATE TABLE IF NOT EXISTS model_performance (
    id SERIAL PRIMARY KEY,
    model_name VARCHAR(100),
    version VARCHAR(20),
    accuracy DECIMAL(5, 4),
    precision DECIMAL(5, 4),
    recall DECIMAL(5, 4),
    f1_score DECIMAL(5, 4),
    training_date TIMESTAMP,
    notes TEXT
);

-- Indexes for better performance
CREATE INDEX IF NOT EXISTS idx_transaction_customer ON transactions(customer_id);
CREATE INDEX IF NOT EXISTS idx_transaction_time ON transactions(transaction_time);
CREATE INDEX IF NOT EXISTS idx_fraud_alerts_transaction ON fraud_alerts(transaction_id);

-- Views for analytics
CREATE OR REPLACE VIEW customer_transaction_stats AS
SELECT 
    c.customer_id,
    c.name,
    COUNT(t.transaction_id) AS transaction_count,
    AVG(t.amount) AS avg_amount,
    MAX(t.amount) AS max_amount,
    MIN(t.amount) AS min_amount,
    COUNT(fa.alert_id) AS fraud_alert_count
FROM customers c
LEFT JOIN transactions t ON c.customer_id = t.customer_id
LEFT JOIN fraud_alerts fa ON t.transaction_id = fa.transaction_id
GROUP BY c.customer_id, c.name;

-- Function to update customer profiles
CREATE OR REPLACE FUNCTION update_customer_profile()
RETURNS TRIGGER AS $$
BEGIN
    -- Update or insert customer profile
    INSERT INTO customer_profiles (
        customer_id, 
        avg_transaction_amount,
        avg_transaction_frequency,
        common_merchants,
        common_locations,
        last_updated
    )
    SELECT 
        t.customer_id,
        AVG(t.amount),
        (MAX(t.transaction_time) - MIN(t.transaction_time)) / COUNT(t.transaction_id),
        ARRAY(SELECT m FROM (
            SELECT t2.merchant as m, COUNT(*) as cnt 
            FROM transactions t2 
            WHERE t2.customer_id = t.customer_id 
            GROUP BY t2.merchant 
            ORDER BY cnt DESC 
            LIMIT 5
        ) sub),
        ARRAY(SELECT l FROM (
            SELECT t3.location as l, COUNT(*) as cnt 
            FROM transactions t3 
            WHERE t3.customer_id = t.customer_id 
            GROUP BY t3.location 
            ORDER BY cnt DESC 
            LIMIT 5
        ) sub),
        CURRENT_TIMESTAMP
    FROM transactions t
    WHERE t.customer_id = NEW.customer_id
    GROUP BY t.customer_id
    ON CONFLICT (customer_id) 
    DO UPDATE SET
        avg_transaction_amount = EXCLUDED.avg_transaction_amount,
        avg_transaction_frequency = EXCLUDED.avg_transaction_frequency,
        common_merchants = EXCLUDED.common_merchants,
        common_locations = EXCLUDED.common_locations,
        last_updated = CURRENT_TIMESTAMP;
    
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Trigger to update customer profile after transaction
CREATE TRIGGER update_profile_after_transaction
AFTER INSERT ON transactions
FOR EACH ROW
EXECUTE FUNCTION update_customer_profile();
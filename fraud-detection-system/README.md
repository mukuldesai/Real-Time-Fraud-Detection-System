# Real-Time Fraud Detection System

A comprehensive system for detecting fraudulent financial transactions using real-time stream processing, machine learning, and big data technologies.

## System Overview

This fraud detection system combines rule-based detection and machine learning to identify potentially fraudulent transactions in real-time. It uses Kafka for data streaming, Apache Flink for stream processing, and a variety of machine learning algorithms for anomaly detection.

![System Architecture](docs/architecture.png)

### Key Features

- **Real-time transaction processing** using Apache Kafka
- **Rule-based fraud detection** for immediate flagging of suspicious activity
- **Machine learning models** (Random Forest, XGBoost) for advanced pattern recognition
- **Interactive dashboards** for monitoring and investigation
- **REST API** for integration with other systems
- **Scalable architecture** designed for high throughput

## Project Structure

```
fraud-detection-system/
├── config/                 # Configuration files
│   ├── config.yml          # Main configuration
│   └── database_schema.sql # Database schema
├── data/                   # Data files
│   └── sample_data.csv     # Sample transaction data
├── deploy/                 # Deployment files
│   └── docker-compose.yml  # Docker composition
├── models/                 # ML models
├── python/                 # Python source code
│   ├── api.py              # REST API
│   ├── database.py         # Database utilities
│   ├── fraud_rules.py      # Rule-based detection
│   ├── kafka_consumer.py   # Kafka message consumer
│   ├── model_predictor.py  # ML model predictor
│   ├── model_trainer.py    # ML model trainer
│   └── transaction_generator.py # Data generator
├── dashboards/             # Dashboard definitions
├── docs/                   # Documentation
├── setup_environment.bat   # Setup script
├── start_system.bat        # Start script
└── stop_system.bat         # Stop script
```

## Installation and Setup

### Prerequisites

- Docker and Docker Compose
- Python 3.8 or higher
- Visual Studio or another code editor
- At least 8GB of RAM for running all containers

### Step 1: Clone the Repository

Clone or download this repository to your local machine, preferably to D drive:

```bash
mkdir D:\fraud-detection-system
cd D:\fraud-detection-system
```

### Step 2: Setup Environment

Run the setup script to create a virtual environment and install dependencies:

```bash
setup_environment.bat
```

This will:
- Create a Python virtual environment
- Install required dependencies
- Set up Kafka topics when containers are running

### Step 3: Start the System

Start all required containers and services:

```bash
start_system.bat
```

This script will:
- Start all Docker containers (Kafka, Flink, Postgres, etc.)
- Initialize the database
- Start the transaction generator
- Start the Kafka consumer
- Start the API server

### Step 4: Access Web Interfaces

Once started, you can access the following web interfaces:

- Kafka UI: http://localhost:8080
- Flink UI: http://localhost:8081
- PgAdmin: http://localhost:5050 (admin@admin.com / admin)
- Metabase: http://localhost:3000
- API Server: http://localhost:5000

## Usage Guide

### Generating Test Data

To generate sample transaction data for testing:

```bash
cd python
python generate_sample_data.py --customers 100 --transactions 10000 --fraud 0.01
```

### Testing the API

The API provides several endpoints:

- `GET /health` - Health check
- `POST /api/v1/fraud/check` - Check a transaction for fraud
- `GET /api/v1/fraud/alerts` - Get recent fraud alerts
- `GET /api/v1/fraud/stats` - Get fraud statistics
- `GET /api/v1/fraud/customer/{customer_id}` - Get customer profile

To test the API:

```bash
cd python
python test_api.py --count 5
```

### Training ML Models

To train or retrain the machine learning model:

```bash
cd python
python model_trainer.py --model xgboost
```

Available model types:
- `xgboost` (default)
- `random_forest`
- `logistic`

### Setting Up Metabase Dashboard

To create a Metabase dashboard for monitoring:

```bash
cd python
python metabase_dashboard.py --wait 60
```

Note: The `--wait` parameter gives Metabase time to initialize before attempting to create the dashboard.

## Development Guide

### Adding New Fraud Detection Rules

To add new rule-based detection patterns, modify `python/fraud_rules.py`. Each rule should implement:

1. A method to evaluate the rule
2. Return values: (is_fraudulent, confidence_score, alert_type)

### Extending the Machine Learning Model

To enhance the ML model:

1. Add new features in `model_trainer.py` in the `prepare_features` method
2. Update `feature_columns` list to include new features
3. Retrain the model

### Adding API Endpoints

To add new API endpoints, modify `python/api.py`:

1. Add a new route decorator and function
2. Implement the functionality
3. Return results as JSON

## Troubleshooting

### Common Issues

1. **Docker containers fail to start**
   - Ensure Docker is running
   - Check if required ports are already in use
   - Verify Docker has enough memory allocated

2. **Kafka connection issues**
   - Check if Kafka container is running: `docker ps`
   - Verify Kafka topics exist: `docker exec kafka kafka-topics --list --bootstrap-server kafka:29092`

3. **Database connection issues**
   - Check if Postgres container is running
   - Verify database schema was created successfully

### Logs

Log files are stored in the `logs` directory:
- `generator.log` - Transaction generator logs
- `consumer.log` - Kafka consumer logs
- `api.log` - API server logs

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Uses open-source libraries and tools
- Based on real-world fraud detection patterns and techniques
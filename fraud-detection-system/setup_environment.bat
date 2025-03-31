@echo off
echo Setting up Python virtual environment for Fraud Detection System...

REM Create virtual environment
python -m venv venv

REM Activate virtual environment
call venv\Scripts\activate.bat

REM Install requirements
pip install -r requirements.txt

REM Create Kafka topics
echo Setting up Kafka topics (will work when Kafka is running)...
docker exec kafka kafka-topics --create --topic transactions --bootstrap-server kafka:29092 --partitions 8 --replication-factor 1 --if-not-exists
docker exec kafka kafka-topics --create --topic fraud-alerts --bootstrap-server kafka:29092 --partitions 8 --replication-factor 1 --if-not-exists
docker exec kafka kafka-topics --create --topic processed-transactions --bootstrap-server kafka:29092 --partitions 8 --replication-factor 1 --if-not-exists

echo Environment setup complete!
echo To activate the environment, run: venv\Scripts\activate.bat
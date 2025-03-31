@echo off
echo Starting Fraud Detection System...

REM Activate virtual environment
call venv\Scripts\activate.bat

REM Create log directory
mkdir logs

REM Start Docker Compose (in detached mode)
echo Starting Docker containers...
docker-compose up -d

REM Wait for services to start
echo Waiting for services to start...
timeout /t 15

REM Initialize database (only if needed)
echo Creating database schema...
docker exec postgres psql -U postgres -d fraud_detection -f /docker-entrypoint-initdb.d/database_schema.sql

REM Start data generator in background
echo Starting transaction generator...
start "Transaction Generator" cmd /c "python python\transaction_generator.py --tps 5 --fraud 0.02 > logs\generator.log 2>&1"

REM Start Kafka consumer in background
echo Starting Kafka consumer...
start "Kafka Consumer" cmd /c "python python\kafka_consumer.py > logs\consumer.log 2>&1"

REM Start API server in background
echo Starting API server...
start "API Server" cmd /c "python python\api.py --debug > logs\api.log 2>&1"

echo Fraud Detection System started!
echo.
echo Web interfaces:
echo  - Kafka UI:      http://localhost:8080
echo  - Flink UI:      http://localhost:8081
echo  - PgAdmin:       http://localhost:5050 (admin@admin.com / admin)
echo  - Metabase:      http://localhost:3000
echo  - API Server:    http://localhost:5000
echo.
echo Log files are in the logs directory
echo.
echo To stop the system, run stop_system.bat
@echo off
echo Stopping Fraud Detection System...

REM Kill Python processes
echo Stopping Python applications...
taskkill /FI "WINDOWTITLE eq Transaction Generator*" /T /F
taskkill /FI "WINDOWTITLE eq Kafka Consumer*" /T /F
taskkill /FI "WINDOWTITLE eq API Server*" /T /F

REM Stop Docker Compose
echo Stopping Docker containers...
docker-compose down

echo Fraud Detection System stopped!
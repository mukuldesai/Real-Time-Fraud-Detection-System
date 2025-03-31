@echo off
echo Fraud Detection System - Project Launcher
echo =========================================
echo.

:menu
echo Select an option:
echo 1. Setup environment (first-time setup)
echo 2. Start all services
echo 3. Stop all services
echo 4. Generate sample data
echo 5. Run API tests
echo 6. Train ML model
echo 7. Create Metabase dashboard
echo 8. Open web interfaces in browser
echo 9. Exit
echo.

set /p choice=Enter your choice (1-9): 

if "%choice%"=="1" goto setup
if "%choice%"=="2" goto start
if "%choice%"=="3" goto stop
if "%choice%"=="4" goto generate
if "%choice%"=="5" goto test
if "%choice%"=="6" goto train
if "%choice%"=="7" goto dashboard
if "%choice%"=="8" goto browser
if "%choice%"=="9" goto end

echo Invalid choice. Please try again.
goto menu

:setup
echo Running environment setup...
call setup_environment.bat
echo.
echo Setup complete. Press any key to return to menu.
pause > nul
goto menu

:start
echo Starting all services...
call start_system.bat
echo.
echo System started. Press any key to return to menu.
pause > nul
goto menu

:stop
echo Stopping all services...
call stop_system.bat
echo.
echo System stopped. Press any key to return to menu.
pause > nul
goto menu

:generate
echo Generating sample data...
call venv\Scripts\activate.bat
cd python
python generate_sample_data.py --customers 100 --transactions 10000 --fraud 0.01
cd ..
echo.
echo Sample data generated. Press any key to return to menu.
pause > nul
goto menu

:test
echo Running API tests...
call venv\Scripts\activate.bat
cd python
python test_api.py --count 5
cd ..
echo.
echo API tests completed. Press any key to return to menu.
pause > nul
goto menu

:train
echo Training ML model...
call venv\Scripts\activate.bat
cd python
python model_trainer.py --model xgboost
cd ..
echo.
echo Model training completed. Press any key to return to menu.
pause > nul
goto menu

:dashboard
echo Creating Metabase dashboard...
call venv\Scripts\activate.bat
cd python
python metabase_dashboard.py --wait 10
cd ..
echo.
echo Dashboard creation initiated. Press any key to return to menu.
pause > nul
goto menu

:browser
echo Opening web interfaces in browser...
start http://localhost:8080
start http://localhost:8081
start http://localhost:5050
start http://localhost:3000
start http://localhost:5000/health
echo.
echo Browsers opened. Press any key to return to menu.
pause > nul
goto menu

:end
echo Thank you for using the Fraud Detection System.
exit /b 0
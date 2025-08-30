@echo off
echo ========================================
echo MT5 Training and Backtesting Launcher
echo ========================================
echo.

echo Choose an option:
echo 1. GPU Troubleshooting
echo 2. Simple 1-Minute Training (3 months)
echo 3. Full Training Pipeline (customizable)
echo 4. Backtest Trained Model
echo 5. Exit
echo.

set /p choice="Enter your choice (1-5): "

if "%choice%"=="1" goto gpu_troubleshoot
if "%choice%"=="2" goto simple_training
if "%choice%"=="3" goto full_training
if "%choice%"=="4" goto backtest
if "%choice%"=="5" goto exit
goto invalid

:gpu_troubleshoot
echo.
echo Running GPU Troubleshooter...
python GPU_Troubleshooter.py
pause
goto menu

:simple_training
echo.
echo Starting Simple 1-Minute Training...
echo This will train on 3 months of 1-minute data
echo.
set /p confirm="Continue? (y/n): "
if /i "%confirm%"=="y" (
    python MT5_1min_Training.py
) else (
    echo Training cancelled.
)
pause
goto menu

:full_training
echo.
echo Full Training Pipeline Configuration
echo ===================================
echo.
set /p timeframe="Timeframe (1m, 5m, 15m, 30m, 1h, 4h) [default: 1m]: "
if "%timeframe%"=="" set timeframe=1m

set /p training_months="Training months [default: 3]: "
if "%training_months%"=="" set training_months=3

set /p backtest_months="Backtest months [default: 1]: "
if "%backtest_months%"=="" set backtest_months=1

set /p timesteps="Training timesteps [default: 500000]: "
if "%timesteps%"=="" set timesteps=500000

echo.
echo Configuration:
echo   Timeframe: %timeframe%
echo   Training months: %training_months%
echo   Backtest months: %backtest_months%
echo   Timesteps: %timesteps%
echo.
set /p confirm="Start training? (y/n): "
if /i "%confirm%"=="y" (
    python MT5_GPU_Fixed_Training.py --timeframe %timeframe% --training_months %training_months% --backtest_months %backtest_months% --timesteps %timesteps%
) else (
    echo Training cancelled.
)
pause
goto menu

:backtest
echo.
echo Backtesting Configuration
echo ========================
echo.
set /p model_path="Model path (.zip file): "
if "%model_path%"=="" (
    echo No model path specified.
    pause
    goto menu
)

set /p timeframe="Timeframe [default: 1m]: "
if "%timeframe%"=="" set timeframe=1m

set /p backtest_months="Backtest months [default: 1]: "
if "%backtest_months%"=="" set backtest_months=1

set /p episodes="Number of episodes [default: 10]: "
if "%episodes%"=="" set episodes=10

echo.
echo Configuration:
echo   Model: %model_path%
echo   Timeframe: %timeframe%
echo   Backtest months: %backtest_months%
echo   Episodes: %episodes%
echo.
set /p confirm="Start backtesting? (y/n): "
if /i "%confirm%"=="y" (
    python MT5_Backtest_After_Training.py --model_path "%model_path%" --timeframe %timeframe% --backtest_months %backtest_months% --episodes %episodes%
) else (
    echo Backtesting cancelled.
)
pause
goto menu

:invalid
echo.
echo Invalid choice. Please enter a number between 1 and 5.
pause
goto menu

:menu
cls
echo ========================================
echo MT5 Training and Backtesting Launcher
echo ========================================
echo.

echo Choose an option:
echo 1. GPU Troubleshooting
echo 2. Simple 1-Minute Training (3 months)
echo 3. Full Training Pipeline (customizable)
echo 4. Backtest Trained Model
echo 5. Exit
echo.

set /p choice="Enter your choice (1-5): "

if "%choice%"=="1" goto gpu_troubleshoot
if "%choice%"=="2" goto simple_training
if "%choice%"=="3" goto full_training
if "%choice%"=="4" goto backtest
if "%choice%"=="5" goto exit
goto invalid

:exit
echo.
echo Goodbye!
pause
exit

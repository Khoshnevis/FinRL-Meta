#!/bin/bash

# MT5 Training and Backtesting Launcher for Linux/Mac

echo "========================================"
echo "MT5 Training and Backtesting Launcher"
echo "========================================"
echo

# Function to show menu
show_menu() {
    echo "Choose an option:"
    echo "1. GPU Troubleshooting"
    echo "2. Simple 1-Minute Training (3 months)"
    echo "3. Full Training Pipeline (customizable)"
    echo "4. Backtest Trained Model"
    echo "5. Exit"
    echo
}

# Function to run GPU troubleshooting
gpu_troubleshoot() {
    echo
    echo "Running GPU Troubleshooter..."
    python3 GPU_Troubleshooter.py
    echo
    read -p "Press Enter to continue..."
}

# Function to run simple training
simple_training() {
    echo
    echo "Starting Simple 1-Minute Training..."
    echo "This will train on 3 months of 1-minute data"
    echo
    read -p "Continue? (y/n): " confirm
    if [[ $confirm =~ ^[Yy]$ ]]; then
        python3 MT5_1min_Training.py
    else
        echo "Training cancelled."
    fi
    echo
    read -p "Press Enter to continue..."
}

# Function to run full training
full_training() {
    echo
    echo "Full Training Pipeline Configuration"
    echo "==================================="
    echo
    
    read -p "Timeframe (1m, 5m, 15m, 30m, 1h, 4h) [default: 1m]: " timeframe
    timeframe=${timeframe:-1m}
    
    read -p "Training months [default: 3]: " training_months
    training_months=${training_months:-3}
    
    read -p "Backtest months [default: 1]: " backtest_months
    backtest_months=${backtest_months:-1}
    
    read -p "Training timesteps [default: 500000]: " timesteps
    timesteps=${timesteps:-500000}
    
    echo
    echo "Configuration:"
    echo "  Timeframe: $timeframe"
    echo "  Training months: $training_months"
    echo "  Backtest months: $backtest_months"
    echo "  Timesteps: $timesteps"
    echo
    
    read -p "Start training? (y/n): " confirm
    if [[ $confirm =~ ^[Yy]$ ]]; then
        python3 MT5_GPU_Fixed_Training.py --timeframe $timeframe --training_months $training_months --backtest_months $backtest_months --timesteps $timesteps
    else
        echo "Training cancelled."
    fi
    echo
    read -p "Press Enter to continue..."
}

# Function to run backtesting
backtest() {
    echo
    echo "Backtesting Configuration"
    echo "========================"
    echo
    
    read -p "Model path (.zip file): " model_path
    if [ -z "$model_path" ]; then
        echo "No model path specified."
        echo
        read -p "Press Enter to continue..."
        return
    fi
    
    read -p "Timeframe [default: 1m]: " timeframe
    timeframe=${timeframe:-1m}
    
    read -p "Backtest months [default: 1]: " backtest_months
    backtest_months=${backtest_months:-1}
    
    read -p "Number of episodes [default: 10]: " episodes
    episodes=${episodes:-10}
    
    echo
    echo "Configuration:"
    echo "  Model: $model_path"
    echo "  Timeframe: $timeframe"
    echo "  Backtest months: $backtest_months"
    echo "  Episodes: $episodes"
    echo
    
    read -p "Start backtesting? (y/n): " confirm
    if [[ $confirm =~ ^[Yy]$ ]]; then
        python3 MT5_Backtest_After_Training.py --model_path "$model_path" --timeframe $timeframe --backtest_months $backtest_months --episodes $episodes
    else
        echo "Backtesting cancelled."
    fi
    echo
    read -p "Press Enter to continue..."
}

# Main loop
while true; do
    clear
    show_menu
    
    read -p "Enter your choice (1-5): " choice
    
    case $choice in
        1)
            gpu_troubleshoot
            ;;
        2)
            simple_training
            ;;
        3)
            full_training
            ;;
        4)
            backtest
            ;;
        5)
            echo
            echo "Goodbye!"
            exit 0
            ;;
        *)
            echo
            echo "Invalid choice. Please enter a number between 1 and 5."
            echo
            read -p "Press Enter to continue..."
            ;;
    esac
done

#!/usr/bin/env python3
"""
GPU Troubleshooter for FinRL-Meta MT5 Training

This script helps diagnose and fix GPU compatibility issues for MT5 training.
It provides detailed information about your GPU setup and suggests solutions.

Usage:
    python GPU_Troubleshooter.py
"""

import os
import sys
import subprocess
import platform
import warnings
warnings.filterwarnings('ignore')

def check_system_info():
    """Check basic system information"""
    print("🖥️  System Information")
    print("=" * 50)
    print(f"OS: {platform.system()} {platform.release()}")
    print(f"Architecture: {platform.machine()}")
    print(f"Python: {platform.python_version()}")
    print(f"Platform: {platform.platform()}")
    print()

def check_nvidia_gpu():
    """Check NVIDIA GPU information using nvidia-smi"""
    print("🔍 NVIDIA GPU Detection")
    print("=" * 50)
    
    try:
        # Try to run nvidia-smi
        result = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total,driver_version', '--format=csv,noheader,nounits'], 
                              capture_output=True, text=True, timeout=10)
        
        if result.returncode == 0:
            gpu_info = result.stdout.strip().split('\n')
            print("✅ NVIDIA GPU(s) detected:")
            for i, gpu in enumerate(gpu_info):
                if gpu.strip():
                    name, memory, driver = gpu.split(', ')
                    print(f"   GPU {i}: {name}")
                    print(f"   Memory: {memory} MB")
                    print(f"   Driver: {driver}")
            print()
            return True
        else:
            print("❌ nvidia-smi failed to run")
            print(f"Error: {result.stderr}")
            print()
            return False
            
    except FileNotFoundError:
        print("❌ nvidia-smi not found. NVIDIA drivers may not be installed.")
        print()
        return False
    except subprocess.TimeoutExpired:
        print("❌ nvidia-smi timed out")
        print()
        return False
    except Exception as e:
        print(f"❌ Error running nvidia-smi: {e}")
        print()
        return False

def check_cuda_installation():
    """Check CUDA installation"""
    print("🔍 CUDA Installation Check")
    print("=" * 50)
    
    # Check CUDA environment variables
    cuda_home = os.environ.get('CUDA_HOME') or os.environ.get('CUDA_PATH')
    if cuda_home:
        print(f"✅ CUDA_HOME: {cuda_home}")
    else:
        print("⚠️  CUDA_HOME not set")
    
    # Check if nvcc is available
    try:
        result = subprocess.run(['nvcc', '--version'], capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            version_line = result.stdout.split('\n')[3]  # CUDA version is usually on line 4
            print(f"✅ CUDA Compiler (nvcc): {version_line}")
        else:
            print("❌ nvcc failed to run")
    except FileNotFoundError:
        print("❌ nvcc not found. CUDA toolkit may not be installed.")
    except Exception as e:
        print(f"❌ Error checking nvcc: {e}")
    
    print()

def check_pytorch_cuda():
    """Check PyTorch CUDA support"""
    print("🔍 PyTorch CUDA Support")
    print("=" * 50)
    
    try:
        import torch
        print(f"✅ PyTorch version: {torch.__version__}")
        
        if torch.cuda.is_available():
            print("✅ CUDA is available in PyTorch")
            print(f"   CUDA version: {torch.version.cuda}")
            print(f"   cuDNN version: {torch.backends.cudnn.version()}")
            print(f"   Number of GPUs: {torch.cuda.device_count()}")
            
            # Test GPU functionality
            try:
                test_tensor = torch.zeros(1).cuda()
                gpu_name = torch.cuda.get_device_name(0)
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
                
                print(f"✅ GPU test successful:")
                print(f"   GPU Name: {gpu_name}")
                print(f"   GPU Memory: {gpu_memory:.1f} GB")
                
                # Check for known problematic GPUs
                problematic_gpus = ["RTX 5090", "RTX 4090 Ti", "RTX 4090 Super"]
                for problematic in problematic_gpus:
                    if problematic in gpu_name:
                        print(f"⚠️  Known problematic GPU detected: {problematic}")
                        print("   This GPU may have compatibility issues with current PyTorch")
                        print("   Consider using CPU or updating to latest PyTorch version")
                
                # Test memory allocation
                try:
                    test_tensor = torch.randn(1000, 1000).cuda()
                    del test_tensor
                    torch.cuda.empty_cache()
                    print("✅ GPU memory allocation test passed")
                except Exception as e:
                    print(f"❌ GPU memory allocation test failed: {e}")
                    print("   This may indicate GPU memory issues")
                
            except Exception as e:
                print(f"❌ GPU functionality test failed: {e}")
                print("   PyTorch CUDA support may be broken")
                
        else:
            print("❌ CUDA is not available in PyTorch")
            print("   This may be due to:")
            print("   1. PyTorch installed without CUDA support")
            print("   2. CUDA version mismatch")
            print("   3. Driver compatibility issues")
            
    except ImportError:
        print("❌ PyTorch not installed")
    except Exception as e:
        print(f"❌ Error checking PyTorch: {e}")
    
    print()

def check_stable_baselines3():
    """Check Stable-Baselines3 installation"""
    print("🔍 Stable-Baselines3 Check")
    print("=" * 50)
    
    try:
        import stable_baselines3
        print(f"✅ Stable-Baselines3 version: {stable_baselines3.__version__}")
        
        # Check if it can import torch
        try:
            from stable_baselines3.common.utils import get_device
            print("✅ Stable-Baselines3 torch integration working")
        except Exception as e:
            print(f"❌ Stable-Baselines3 torch integration error: {e}")
            
    except ImportError:
        print("❌ Stable-Baselines3 not installed")
    except Exception as e:
        print(f"❌ Error checking Stable-Baselines3: {e}")
    
    print()

def provide_solutions():
    """Provide solutions for common GPU issues"""
    print("🔧 Common Solutions for GPU Issues")
    print("=" * 50)
    
    print("1. PyTorch CUDA Installation Issues:")
    print("   - Uninstall current PyTorch: pip uninstall torch torchvision torchaudio")
    print("   - Install CUDA-compatible PyTorch:")
    print("     pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
    print("   - For CUDA 12.1: pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121")
    print()
    
    print("2. Driver Compatibility Issues:")
    print("   - Update NVIDIA drivers to latest version")
    print("   - Ensure driver version supports your CUDA version")
    print("   - Check NVIDIA website for driver compatibility matrix")
    print()
    
    print("3. RTX 5090/4090 Ti Compatibility:")
    print("   - These GPUs use new architecture (sm_120) not yet fully supported")
    print("   - Try PyTorch nightly builds: pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu121")
    print("   - Consider using CPU for now (slower but stable)")
    print("   - Monitor PyTorch releases for official support")
    print()
    
    print("4. Memory Issues:")
    print("   - Reduce batch size in training configuration")
    print("   - Use gradient accumulation")
    print("   - Monitor GPU memory usage with nvidia-smi")
    print()
    
    print("5. Environment Variables:")
    print("   - Set CUDA_VISIBLE_DEVICES to specific GPU: export CUDA_VISIBLE_DEVICES=0")
    print("   - Set CUDA_LAUNCH_BLOCKING=1 for debugging: export CUDA_LAUNCH_BLOCKING=1")
    print()

def check_mt5_requirements():
    """Check MT5-specific requirements"""
    print("🔍 MT5 Requirements Check")
    print("=" * 50)
    
    try:
        import MetaTrader5 as mt5
        print("✅ MetaTrader5 package installed")
        
        # Check MT5 connection
        if mt5.initialize():
            print("✅ MT5 terminal connection successful")
            terminal_info = mt5.terminal_info()
            if terminal_info:
                print(f"   Terminal: {getattr(terminal_info, 'name', 'Unknown')}")
                print(f"   Build: {getattr(terminal_info, 'build', 'Unknown')}")
                print(f"   Connected: {getattr(terminal_info, 'connected', 'Unknown')}")
        else:
            print("❌ MT5 terminal connection failed")
            print("   Make sure MetaTrader 5 terminal is running")
        
        mt5.shutdown()
        
    except ImportError:
        print("❌ MetaTrader5 package not installed")
        print("   Install with: pip install MetaTrader5")
    except Exception as e:
        print(f"❌ Error checking MT5: {e}")
    
    print()

def run_diagnostic():
    """Run complete diagnostic"""
    print("🚀 GPU Troubleshooter for FinRL-Meta MT5 Training")
    print("=" * 70)
    print()
    
    check_system_info()
    check_nvidia_gpu()
    check_cuda_installation()
    check_pytorch_cuda()
    check_stable_baselines3()
    check_mt5_requirements()
    provide_solutions()
    
    print("📋 Summary")
    print("=" * 50)
    print("If you're experiencing GPU issues:")
    print("1. Check the solutions above")
    print("2. Try running training with CPU first: --device cpu")
    print("3. Update your PyTorch and CUDA installations")
    print("4. Check NVIDIA driver compatibility")
    print("5. For RTX 5090/4090 Ti, consider waiting for official PyTorch support")
    print()
    print("For more help, check:")
    print("- PyTorch documentation: https://pytorch.org/get-started/locally/")
    print("- NVIDIA driver downloads: https://www.nvidia.com/drivers/")
    print("- FinRL-Meta issues: https://github.com/AI4Finance-Foundation/FinRL-Meta/issues")

if __name__ == "__main__":
    run_diagnostic()

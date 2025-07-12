import torch
import subprocess
import sys

def run_cmd(command):
    try:
        output = subprocess.check_output(command, shell=True, stderr=subprocess.STDOUT, text=True)
        return output.strip()
    except subprocess.CalledProcessError as e:
        return f"Command '{command}' failed with error:\n{e.output.strip()}"
    except FileNotFoundError:
        return f"Command not found: {command.split()[0]}"

def check_gpu_env():
    print("--- GPU Environment Check ---")

    # 1. NVIDIA Driver and nvidia-smi
    print("\n1. Checking NVIDIA driver and nvidia-smi...")
    nvidia_smi_output = run_cmd("nvidia-smi")
    print(nvidia_smi_output)

    # 2. NVCC (CUDA Compiler)
    print("\n2. Checking nvcc (CUDA Compiler)...")
    nvcc_output = run_cmd("nvcc --version")
    print(nvcc_output)

    # 3. PyTorch and CUDA
    print("\n3. Checking PyTorch setup...")
    print(f"   Python version: {sys.version}")
    try:
        import torch
        print(f"   PyTorch version: {torch.__version__}")
        cuda_available = torch.cuda.is_available()
        print(f"   Is CUDA available?: {cuda_available}")

        if cuda_available:
            print(f"   CUDA version (from PyTorch): {torch.version.cuda}")
            
            # 4. cuDNN version
            try:
                cudnn_version = torch.backends.cudnn.version()
                print(f"   cuDNN version: {cudnn_version}")
            except Exception as e:
                print(f"   Could not determine cuDNN version: {e}")

            device_count = torch.cuda.device_count()
            print(f"   Number of GPUs found: {device_count}")
            for i in range(device_count):
                print(f"   GPU {i}: {torch.cuda.get_device_name(i)}")
                print(f"     - Compute Capability: {torch.cuda.get_device_capability(i)}")
        else:
            print("   PyTorch was not built with CUDA support.")

    except ImportError:
        print("   PyTorch is not installed.")
    except Exception as e:
        print(f"   An error occurred while checking PyTorch: {e}")

    print("\n--- End of Check ---")

if __name__ == "__main__":
    check_gpu_env() 
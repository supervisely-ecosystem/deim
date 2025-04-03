import psutil
import os
import torch

def print_ram_usage(message: str):
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    
    output = f"\n------------------------- {message} -------------------------\n"
    output += f"RSS (physical memory): {memory_info.rss / 1024**2:.1f} MB\n"
    output += f"VMS (virtual memory): {memory_info.vms / 1024**2:.1f} MB\n"
    output += f"System RAM usage: {psutil.virtual_memory().percent}%\n"
    output += f"Available system RAM: {psutil.virtual_memory().available / 1024**2:.1f} MB\n"
    
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            gpu_memory_allocated = torch.cuda.memory_allocated(i) / 1024**2
            gpu_memory_reserved = torch.cuda.memory_reserved(i) / 1024**2
            output += f"\nGPU {i}:\n"
            output += f"Allocated memory: {gpu_memory_allocated:.1f} MB\n"
            output += f"Reserved memory: {gpu_memory_reserved:.1f} MB\n"
    
    print(output)
    with open("ram_usage.log", "a") as f:
        f.write(output)
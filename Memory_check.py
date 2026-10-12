import psutil

# Total memory
total_mem = psutil.virtual_memory().total / (1024**3)  # in GB
# Available memory
available_mem = psutil.virtual_memory().available / (1024**3)  # in GB
# Used memory
used_mem = psutil.virtual_memory().used / (1024**3)  # in GB

print(f"Total Memory: {total_mem:.2f} GB")
print(f"Available Memory: {available_mem:.2f} GB")
print(f"Used Memory: {used_mem:.2f} GB")
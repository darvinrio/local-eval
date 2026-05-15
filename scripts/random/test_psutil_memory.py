"""
scripts/random/test_psutil_memory.py

Verify if the memory stats from psutil are close to expected values.
"""

import psutil

mem = psutil.virtual_memory()
print(f"Total: {mem.total / (1024**3):.2f} GiB")
print(f"Available: {mem.available / (1024**3):.2f} GiB")
print(f"Used: {mem.used / (1024**3):.2f} GiB")
print("______")
print(mem)

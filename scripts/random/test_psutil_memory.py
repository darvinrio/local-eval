"""
scripts/random/test_psutil_memory.py

Verify if the memory stats from psutil are close to expected values.
"""

import psutil

mem = psutil.virtual_memory()
print(f"Total: {mem.total / 1e9:.2f} GB")
print(f"Available: {mem.available / 1e9:.2f} GB")
print(f"Used: {mem.used / 1e9:.2f} GB")
print("______")
print(mem)

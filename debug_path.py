import os
from pathlib import Path

target = r"C:\Users\hungz\Student\look_painting\fashion-mnist\data\synthetic\images"
p = Path(target)
print(f"Target: {target}")
print(f"Exists: {p.exists()}")
print(f"Is Dir: {p.is_dir()}")
print(f"Absolute: {p.absolute()}")

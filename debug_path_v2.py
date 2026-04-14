import os
from pathlib import Path
import json

# Giả lập dữ liệu nhận được từ JSON request
json_data = '{"path": "C:\\\\Users\\\\hungz\\\\Student\\\\look_painting\\\\fashion-mnist\\\\data\\\\synthetic\\\\images"}'
data = json.loads(json_data)
path_str = data['path']

p1 = Path(path_str)
p2 = Path(path_str.replace('\\', '/'))

print(f"Path Str: {path_str}")
print(f"P1 exists: {p1.exists()}")
print(f"P2 exists: {p2.exists()}")
print(f"Current Dir: {os.getcwd()}")

import os
import re

def check_asyncio_imports(directory):
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".py"):
                path = os.path.join(root, file)
                with open(path, "r", encoding="utf-8", errors="ignore") as f:
                    content = f.read()
                    if "asyncio." in content:
                        # Check if asyncio is imported
                        if not re.search(r"^\s*import asyncio\b", content, re.MULTILINE) and \
                           not re.search(r"^\s*from asyncio import\b", content, re.MULTILINE):
                            print(f"File missing asyncio import: {path}")

if __name__ == "__main__":
    check_asyncio_imports("synapse")

import sys
import traceback

def check_import(module_name, entity_name=None):
    print(f"Checking {module_name}...", end=" ", flush=True)
    try:
        if entity_name:
            # from module import entity
            module = __import__(module_name, fromlist=[entity_name])
            entity = getattr(module, entity_name)
            print(f"OK (imported {entity_name} from {module_name})")
        else:
            # import module
            __import__(module_name)
            print("OK")
        return True
    except ImportError as e:
        print(f"FAILED")
        print(f"  --> Error: {e}")
        return False
    except Exception as e:
        print(f"ERROR: {e}")
        traceback.print_exc()
        return False

def main():
    print("=== Synapse Dependency Diagnostic ===")
    print(f"Python version: {sys.version}")
    print("-" * 40)
    
    dependencies = [
        ("torch", None),
        ("torchvision", None),
        ("transformers", "AutoTokenizer"),
        ("transformers", "AutoProcessor"),
        ("timm", None),
        ("sentencepiece", None),
        ("cv2", None),
        ("PIL", "Image"),
        ("grpc", None),
        ("numpy", None),
        ("accelerate", None),
    ]
    
    missing = []
    for mod, entity in dependencies:
        if not check_import(mod, entity):
            missing.append(mod)
            
    print("-" * 40)
    if not missing:
        print("✅ All core dependencies are successfully imported!")
    else:
        print(f"❌ Missing dependencies: {', '.join(missing)}")
        print("\nSuggested fix:")
        print(f"pip install {' '.join(missing)}")
        if "cv2" in missing:
            print("Note: For Ubuntu VM, use: pip install opencv-python-headless")
        if "torchvision" in missing:
            print("Note: torchvision is critical for AutoProcessor.")

if __name__ == "__main__":
    main()

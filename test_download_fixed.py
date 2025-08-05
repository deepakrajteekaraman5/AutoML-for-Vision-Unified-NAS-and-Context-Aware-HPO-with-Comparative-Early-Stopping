# test_download_correct.py
import sys
from pathlib import Path

# Find the correct path
current_dir = Path(__file__).parent
template_dir = current_dir / "automl-exam-ss25-vision-freiburg-template"
src_path = template_dir / "src"

print(f"Current directory: {current_dir}")
print(f"Template directory exists: {template_dir.exists()}")
print(f"Src path: {src_path}")
print(f"Src path exists: {src_path.exists()}")

if src_path.exists():
    sys.path.insert(0, str(src_path))
    
    try:
        from automl.datasets import FlowersDataset
        print("✓ Import successful!")
        
        print("Testing dataset download...")
        dataset = FlowersDataset(root="data", download=True)
        print(f"✓ Download successful!")
        print(f"Dataset length: {len(dataset)}")
        
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
else:
    print("Please run this command to show me your directory structure:")
    print("dir (on Windows) or ls -la (on Git Bash)")
# tests/__init__.py
"""
Test suite for AutoML Pipeline
"""

import sys
from pathlib import Path

# Add the src directory to Python path for all tests
def setup_test_environment():
    """Setup Python path for testing"""
    current_dir = Path(__file__).parent
    project_root = current_dir.parent
    template_dir = project_root / "automl-exam-ss25-vision-freiburg-template"
    src_path = template_dir / "src"
    
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))
    
    return src_path

# Automatically setup when tests module is imported
setup_test_environment()
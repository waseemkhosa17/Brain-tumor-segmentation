import os
import sys

def setup_environment():
    """Setup Python path for the project"""
    project_root = os.path.dirname(os.path.abspath(__file__))
    src_path = os.path.join(project_root, 'src')
    
    # Add to Python path
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    if src_path not in sys.path:
        sys.path.insert(0, src_path)
    
    print(f"Added to path: {project_root}")
    print(f"Added to path: {src_path}")
    
    # Create necessary directories
    dirs = [
        'models',
        'data/raw', 'data/processed',
        'outputs/predictions', 'outputs/visualizations',
        'uploads'
    ]
    
    for dir_path in dirs:
        os.makedirs(dir_path, exist_ok=True)
        print(f"Created directory: {dir_path}")
    
    print("Setup completed!")

if __name__ == "__main__":
    setup_environment()
import os
import sys

def test_imports():
    """Test if all imports work"""
    try:
        # Add to path
        current_dir = os.path.dirname(os.path.abspath(__file__))
        sys.path.insert(0, current_dir)
        sys.path.insert(0, os.path.join(current_dir, 'src'))
        
        # Test imports
        from src.model_nnunet import nnUNet3D
        from src.dataset import BraTSDataset3D
        from src.losses import CombinedLoss
        from src.utils import calculate_metrics
        
        print("✅ All imports successful!")
        
        # Test model creation
        model = nnUNet3D()
        print(f"✅ Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
        
        # Test dataset
        dataset = BraTSDataset3D('./data/raw')
        print(f"✅ Dataset created with {len(dataset)} samples")
        
        return True
        
    except Exception as e:
        print(f"❌ Import error: {e}")
        return False

def test_flask_app():
    """Test if Flask app can start"""
    try:
        from app.app import app
        print("✅ Flask app imported successfully!")
        return True
    except Exception as e:
        print(f"❌ Flask app error: {e}")
        return False

if __name__ == "__main__":
    print("Testing Brain Tumor Segmentation App...")
    print("=" * 50)
    
    imports_ok = test_imports()
    flask_ok = test_flask_app()
    
    print("=" * 50)
    if imports_ok and flask_ok:
        print("🎉 All tests passed! You can run the app with:")
        print("   python app/app.py")
    else:
        print("❌ Some tests failed. Please check the errors above.")
                                         
                                                  

import os
import json
import torch
import shutil
import tempfile
from unittest.mock import patch
from train_disease_classifier import prepare_dataset_for_training, train_model

def test_prepare_dataset():
    """Test dataset preparation function."""
    print("Testing dataset preparation...")

                                                                    
    result = prepare_dataset_for_training()
    assert result == True, "Dataset preparation should succeed"

                                              
    data_dir = os.path.join(os.path.dirname(__file__), 'dataset')
    assert os.path.exists(data_dir), "Dataset directory should be created"

                                           
    subdirs = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
    assert len(subdirs) > 0, "Should have class subdirectories"

    print("✓ Dataset preparation test passed")

def test_train_model_quick():
    """Test model training with minimal epochs for speed."""
    print("Testing model training (quick version)...")

                                             
    with patch('train_disease_classifier.EPOCHS', 1):
        train_model()

                                       
    model_path = os.path.join(os.path.dirname(__file__), 'disease_model.pt')
    assert os.path.exists(model_path), "Model file should be created"

                                               
    class_map_path = os.path.join(os.path.dirname(__file__), 'class_to_name.json')
    assert os.path.exists(class_map_path), "Class mapping file should be created"

                                        
    with open(class_map_path, 'r') as f:
        class_map = json.load(f)
    assert isinstance(class_map, dict), "Class mapping should be a dictionary"
    assert len(class_map) > 0, "Class mapping should not be empty"

                                
    try:
        model = torch.load(model_path, map_location='cpu')
        assert model is not None, "Model should load successfully"
    except Exception as e:
        assert False, f"Model loading failed: {e}"

    print("✓ Model training test passed")

def cleanup_test_files():
    """Clean up test-generated files."""
    print("Cleaning up test files...")

    files_to_remove = [
        os.path.join(os.path.dirname(__file__), 'disease_model.pt'),
        os.path.join(os.path.dirname(__file__), 'class_to_name.json')
    ]

    data_dir = os.path.join(os.path.dirname(__file__), 'dataset')
    if os.path.exists(data_dir):
        shutil.rmtree(data_dir)

    for file_path in files_to_remove:
        if os.path.exists(file_path):
            os.remove(file_path)

    print("✓ Cleanup completed")

if __name__ == "__main__":
    print("🧬 MODEL TRAINING TEST SUITE")
    print("=" * 40)

    try:
        test_prepare_dataset()
        test_train_model_quick()
        print("\n🎉 ALL TRAINING TESTS PASSED!")
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        raise
    finally:
        cleanup_test_files()

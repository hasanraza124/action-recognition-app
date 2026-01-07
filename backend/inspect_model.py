import h5py
import os

MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'download', 'best_model.h5')

print(f"Inspecting model: {MODEL_PATH}")
print(f"File exists: {os.path.exists(MODEL_PATH)}")
print(f"File size: {os.path.getsize(MODEL_PATH) / (1024*1024):.2f} MB\n")

try:
    with h5py.File(MODEL_PATH, 'r') as f:
        print("HDF5 File Structure:")
        print("=" * 50)
        
        def print_structure(name, obj):
            if isinstance(obj, h5py.Dataset):
                print(f"  Dataset: {name} - Shape: {obj.shape}, Dtype: {obj.dtype}")
            elif isinstance(obj, h5py.Group):
                print(f"  Group: {name}")
        
        f.visititems(print_structure)
        
        # Check for model config
        if 'model_config' in f.attrs:
            import json
            config = json.loads(f.attrs['model_config'])
            print("\n" + "=" * 50)
            print("Model Configuration:")
            print("=" * 50)
            print(f"Class name: {config.get('class_name', 'Unknown')}")
            if 'config' in config:
                conf = config['config']
                print(f"Name: {conf.get('name', 'Unknown')}")
                if 'layers' in conf:
                    print(f"Number of layers: {len(conf['layers'])}")
                    print("\nLayers:")
                    for i, layer in enumerate(conf['layers'][:5]):  # Show first 5
                        print(f"  {i+1}. {layer.get('class_name', 'Unknown')}")
                    if len(conf['layers']) > 5:
                        print(f"  ... and {len(conf['layers']) - 5} more layers")
                        
except Exception as e:
    print(f"Error inspecting model: {e}")

print("\n" + "=" * 50)
print("Attempting to load with TensorFlow...")
print("=" * 50)

try:
    import tensorflow as tf
    from tensorflow import keras
    
    # Try different loading methods
    methods = [
        ("Standard load", lambda: keras.models.load_model(MODEL_PATH)),
        ("Load without compile", lambda: keras.models.load_model(MODEL_PATH, compile=False)),
    ]
    
    for method_name, load_func in methods:
        try:
            print(f"\nTrying: {method_name}...")
            model = load_func()
            print(f"✓ SUCCESS with {method_name}!")
            print(f"  Input shape: {model.input_shape}")
            print(f"  Output shape: {model.output_shape}")
            print(f"  Total layers: {len(model.layers)}")
            break
        except Exception as e:
            print(f"✗ Failed: {str(e)[:100]}")
            
except ImportError:
    print("TensorFlow not available for loading test")

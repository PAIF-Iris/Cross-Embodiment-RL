import h5py
import numpy as np

path = "/Users/paif_iris/Desktop/metaworld/robosuite_demo_data/Door/Panda/raw/Door_demo_act_norm.hdf5"

with h5py.File(path, "r") as f:
    # Pick one demo to inspect its structure
    demo_name = "demo_1"
    demo = f["data"][demo_name]
    
    print(f"=== Structure of {demo_name} ===\n")
    
    # Show all keys/datasets in this demo
    print("Keys in demo:")
    for key in demo.keys():
        print(f"  {key}")
    
    print("\n" + "="*50 + "\n")
    
    # Show details of each dataset
    print("Dataset details:\n")
    for key in demo.keys():
        item = demo[key]
        if isinstance(item, h5py.Dataset):
            print(f"{key}:")
            print(f"  Shape: {item.shape}")
            print(f"  Dtype: {item.dtype}")
            print(f"  First few values: {item[:min(3, len(item))]}")
            print()
        elif isinstance(item, h5py.Group):
            print(f"{key}: (group with {len(item.keys())} items)")
            for subkey in item.keys():
                subitem = item[subkey]
                if isinstance(subitem, h5py.Dataset):
                    print(f"  {subkey}: shape={subitem.shape}, dtype={subitem.dtype}")
            print()
    
    print("="*50)
    print(f"\nTotal number of demos: {len(f['data'].keys())}")
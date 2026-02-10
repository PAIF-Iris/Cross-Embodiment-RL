import h5py
import numpy as np
import matplotlib.pyplot as plt

path = "/Users/paif_iris/Desktop/metaworld/robosuite_demo_data/Door/Panda/raw/Door_demo_act_norm.hdf5"
with h5py.File(path, "r") as f:
    actions = f["policy/actions"][:]

print("Extracted action array shape:", actions.shape)
print("First 5 actions:")
print(actions[:5])

with h5py.File(path, "r") as f:
    print("Top-level keys:")
    for k in f.keys():
        print("  ", k)
with h5py.File(path, "r") as f:
    if "policy" in f:
        for k in f["policy"].keys():
            print("policy:", k)
with h5py.File(path, "r") as f:
    if "env" in f:
        for k in f["env"].keys():
            print("env:", k)

with h5py.File(path, "r") as f:
    actions = f["policy/actions"]
    print("Action shape:", actions.shape)
    print("First action vector:", actions[0])
    print("Action dtype:", actions.dtype)

with h5py.File(path, "r") as f:
    env = f["env/state"]
    print("State shape:", env.shape)
    print("First state vector:", env[0])
    print("State dtype:", env.dtype)

with h5py.File(path, "r") as f:
    env = f["env/cam0_video"]
    video = env['frames']
    print("Camera video shape:", video.shape)




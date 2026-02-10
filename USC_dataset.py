
import h5py
import numpy as np

path = "/Users/paif_iris/Desktop/metaworld/test_data_jaco_play.h5"
data = {}
with h5py.File("/Users/paif_iris/Desktop/metaworld/test_data_jaco_play.h5", "r") as F:
  for key in F.keys():
    data[key] = np.array(F[key])

print("Dataset shapes:")
for key in data:
  print(f"{key}: {data[key].shape}")


with h5py.File(path, "r") as f:
    print("Top-level keys:")
    for k in f.keys():
        print("  ", k)

with h5py.File(path, "r") as f:
    env = f["ee_cartesian_pos_ob"]
    print("State shape:", env.shape)
    print("First state vector:", env[0])
    print("State dtype:", env.dtype)


with h5py.File(path, "r") as f:
    env = f["prompts"]
    print("State shape:", env.shape)
    print("First state vector:", env[0])
    print("State dtype:", env.dtype)
  
with h5py.File(path, "r") as f:
    env = f["reward"]
    print("reward shape:", env.shape)
    print(" reward:", env)
    for i in range(239):
      if (env[i] != 0):
         print("Index:", i, "Reward:", env[i])
      else:
        print(" reward:", env[i])

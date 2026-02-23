import gymnasium as gym
import metaworld
import h5py
import numpy as np
import pandas as pd
import time

path = "/Users/paif_iris/Desktop/metaworld/episode_000001.parquet"
df = pd.read_parquet(path)

render_mode = 'human'
env = gym.make('Meta-World/MT1', env_name="pick-place-v3", render_mode=render_mode, seed=42)



def map_action_7d_to_4d(action_7d, scale_position=10):
    # Scale position deltas
    x = action_7d[0] * scale_position
    y = action_7d[1] * scale_position
    z = action_7d[2] * scale_position
    
    # Handle gripper - convert to binary open/close
    # Original gripper might be continuous [0,1] or [-1,1]
    gripper_raw = action_7d[6]
    
    # Convert to MetaWorld's binary gripper: -1 = close, 1 = open
    if gripper_raw == 0:
        gripper = -1.0  # Open gripper
    elif gripper_raw == -1:
        gripper = 1.0  # Close gripper
    else:
        gripper = -1.0   # Open gripper
    
    # Clip position deltas to MetaWorld's range
    action_4d = np.array([x, y, z, gripper], dtype=np.float32)
    action_4d[:3] = np.clip(action_4d[:3], -1.0, 1.0)  # Clip positions only
    
    return action_4d


# Metaworld
# - EE_XYZ_Start = [0.0045842, 0.60138811, 0.19514345]
# - EE_At_Object = [2.22111578e-02, 6.45388010e-01, 0.02]

# Robot
# - EE_XYZ_Start = [0.378831, 0.015589, 0.684344]
# - EE_XYZ_At_Object = [0.486089, 0.27058, 0.13536]
# - Object_Start = [0.485241, 0.270474, -0.005]

# Metaworld = s * Isaac + t
# Metaworld(EE_XYZ_Start) = s * Robot(EE_XYZ_Start) + t

# 0.02221116 = s​ * 0.485241 + t
# 0.64538801 = s1 * 0.270474 + t1

# change in metaworld = s * change in isaac
# s = d metaworld / d isaac
# s = (obj(metaword) - Metaworld(start)) / (obj(robot) - Robot(start))

# Δx_I = 0.485241 − 0.378831 = 0.106410
# Δx_M = 0.02221116 − 0.0045842 = 0.01762696
# s_x = 0.01762696 / 0.106410 = 0.1657

# Δy_I = 0.270474 − 0.015589 = 0.254885
# Δy_M = 0.64538801 − 0.60138811 = 0.044000
# s_y = 0.044000 / 0.254885 = 0.1726
# s roughly 0.17

EE_XYZ_Start = df['observation.state'].iloc[0][56:59]
EE_XYZ_End = df['observation.state'].iloc[len(df)-1][56:59]
meta_obj_start = np.array([0, 0.6, 0.02])
print("EE_XYZ_Start", EE_XYZ_Start)
print("EE_XYZ_End", EE_XYZ_End)

s = 0.17

relative_expert = EE_XYZ_End - EE_XYZ_Start
relative_meta = s * relative_expert

meta_eef_start = meta_obj_start - relative_meta

print("EE start in metaworld:", meta_eef_start)

# 1. Set EE start BEFORE reset
env.unwrapped.hand_init_pos = np.array([
    meta_eef_start[0],
    meta_eef_start[1],
    meta_eef_start[2]
])
print("hand_init_pos", env.unwrapped.hand_init_pos)

# 2. Reset (this applies hand_init_pos)
obs, info = env.reset()
print("observation after reset:", obs)

# 4. Set object AFTER reset
env.unwrapped._set_obj_xyz(meta_obj_start)

# 5. (Optional) refresh obs
#obs = env.unwrapped._get_obs()
print("hand_init_pos", env.unwrapped.hand_init_pos)
print("object pos after reset:", env.unwrapped._get_pos_objects())


for i in range(68):
    action_7d = df['action'].iloc[i]
        
    # Map action
    action_4d = map_action_7d_to_4d(action_7d)
    print("action:", action_4d)
    
    # Step environment
    observation, reward, terminated, truncated, info = env.step(action_4d)
    print(f"Step {i+1}: observation={observation}")

    env.render()
    time.sleep(0.1)
    print (i)

env.close()
print(type(env.unwrapped))
print(dir(env.unwrapped))
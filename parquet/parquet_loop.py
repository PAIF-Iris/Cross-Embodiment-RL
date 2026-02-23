import math

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



def map_action_7d_to_4d(action_7d, mask, scale_position=1, scale_action=1):
    normal = math.sqrt(sum(x**2 for x in action_7d))

    if normal == 0:
        return np.zeros(4, dtype=np.float32)

    # Scale position 
    x = abs(action_7d[0]) / normal * scale_position * scale_action * mask[0]+ 0.0001
    y = abs(action_7d[1]) / normal * scale_position * scale_action * mask[1]+ 0.0001
    z = abs(action_7d[2]) / normal * scale_position * scale_action * mask[2]+ 0.0001
    gripper_raw = action_7d[3]
    
    if gripper_raw == 0:
        gripper = -1.0  # Open gripper
    elif gripper_raw == -1:
        gripper = 1.0  # Close gripper
    else:
        gripper = -1.0   # Open gripper
    
    action_4d = np.array([x, y, z, gripper], dtype=np.float32)
    
    return action_4d

def action_determine(current_pos, target_pos, observation,threshold=0.005):
    diff = target_pos - current_pos
    mask = np.zeros(3, dtype=np.float32)
    for i in range(3):
        if (i == 2 and observation[2] <= 0.047):
            mask[i] = 0.0
        elif (diff[i] > threshold):
            mask[i] = 1.0
        elif (diff[i] < -threshold):
            mask[i] = -1.0
        else:            
            mask[i] = 0.0

    return mask

EE_XYZ_Start = df['observation.state'].iloc[0][56:59]
EE_XYZ_End = df['observation.state'].iloc[39][56:59]
meta_obj_start_adjusted = np.array([0, 0.6, 0.02])
meta_obj_start = np.array([0, 0.6, 0.02])
print("EE_XYZ_Start", EE_XYZ_Start)
print("EE_XYZ_End", EE_XYZ_End)

relative_expert = EE_XYZ_End - EE_XYZ_Start
meta_eef_start = meta_obj_start_adjusted - relative_expert

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
print("start is obs[0:3]", obs[0:3])
print("start is", meta_eef_start)

target_pos = meta_eef_start
for i in range(68):
    action_7d = df['observation.state'].iloc[i+1][56:59] - df['observation.state'].iloc[i][56:59]
    action_7d = np.append(action_7d, df['action'].iloc[i][6]) 
    current_pos = obs[0:3]
    target_pos = target_pos + action_7d[0:3]
    print("target_pos:", target_pos)
    mask = action_determine(current_pos, target_pos, obs)
    
    while np.any(mask != 0):
        action_4d = map_action_7d_to_4d(action_7d, mask)
        print("action:", action_4d)
        
        obs, reward, terminated, truncated, info = env.step(action_4d)
        print(f"Step {i+1}: observation={obs[0:3]}")

        current_pos = obs[0:3]
        env.render()
        mask = action_determine(current_pos, target_pos, obs)
        print(f"mask: {mask}")
    
        
    time.sleep(0.1)
    print (i)

env.close()
print(type(env.unwrapped))
print(dir(env.unwrapped))
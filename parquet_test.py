import gymnasium as gym
import metaworld
import h5py
import numpy as np
import pandas as pd
import time

path = "/Users/paif_iris/Desktop/metaworld/episode_000042.parquet"
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



# s = 0.17

# observation, info = env.reset()
# start_ee_pos = observation[0:3]
# x =  s*(0.486089 - 0.378831) + start_ee_pos[0]
# y =  s*(0.27058 - 0.015589) + start_ee_pos[1]
# z = s*(-0.005 - 0.684344) + start_ee_pos[2]
# print("Setting object to:", x, y, z)
# env.unwrapped._set_obj_xyz(np.array([x, y, 0.02]))  # (-0.6~0.6, 0.35~0.95, 0.02)
# print("Original hand_init_pos:", env.unwrapped.hand_init_pos)
# env.unwrapped.hand_init_pos = np.array([
#     env.unwrapped.hand_init_pos[0],
#     env.unwrapped.hand_init_pos[1],
#     0.1   # your desired lower Z
# ])
# print("Modified hand_init_pos:", env.unwrapped.hand_init_pos)
# # observation[2] -= 0.01
# # start_ee_pos = observation[0:3]
# # obs, info = env.reset()
# print("Start end-effector XYZ:", start_ee_pos)



# # s = 1

# # x = 0-(0.486089 - 0.378831)
# # y = 0.6-(0.27058  - 0.015589)
# # z = 0.02-( -0.005 - 0.684344)

# env.unwrapped.hand_init_pos = np.array([
#     x,
#     y,
#     z
# ])
# print("end-effector start", x, y, z)

# obs, info = env.reset()
# start_ee_pos = obs[0:3]

# print("EE start after reset:", start_ee_pos)

# #3. Compute object position RELATIVE to EE
# x = start_ee_pos[0] + s * (0.486089 - 0.378831)
# y = start_ee_pos[1] + s * (0.27058  - 0.015589)


# #4. Set object AFTER reset
# env.unwrapped._set_obj_xyz(np.array([x, y, 0.02]))

# #env.unwrapped._set_obj_xyz(np.array([0, 0.6, 0.02]))


# print("Setting object to:", x, y)

s = 0.17

# 1. Set EE start BEFORE reset
env.unwrapped.hand_init_pos = np.array([
    env.unwrapped.hand_init_pos[0],
    env.unwrapped.hand_init_pos[1],
    0.10   # lower Z
])

# 2. Reset (this applies hand_init_pos)
obs, info = env.reset()
start_ee_pos = obs[0:3]

print("EE start after reset:", start_ee_pos)

# 3. Compute object position RELATIVE to EE
x = start_ee_pos[0] + s * (0.486089 - 0.378831)
y = start_ee_pos[1] + s * (0.27058  - 0.015589)

print("Setting object to:", x, y)

# 4. Set object AFTER reset
env.unwrapped._set_obj_xyz(np.array([x, y, 0.02]))

# 5. (Optional) refresh obs
obs = env.unwrapped._get_obs()



for i in range(68):
    action_7d = df['action'].iloc[i]
        
    # Map action
    action_4d = map_action_7d_to_4d(action_7d)
    print("action:", action_4d)
    
    # Step environment
    observation, reward, terminated, truncated, info = env.step(action_4d)
    #print(f"Step {i+1}: observation={observation}")

    env.render()
    time.sleep(0.1)
    print (i)

env.close()
print(type(env.unwrapped))
print(dir(env.unwrapped))
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



def map_action_7d_to_4d(action_7d, EEF_isaac, object_isaac, scale_position=1.0):
    relative_pos_isaac = object_isaac - EEF_isaac
    object_metaworld = env.unwrapped._get_pos_objects()
    print("Object position in MetaWorld:", object_metaworld)

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

x = 0.378831 - 0.485241 + 0.6
y = 0.015589 - 0.270474 + 0
z = 0.684344 + 0.005  + 0.2

env.unwrapped.hand_init_pos = np.array([
    x,
    y,
    z
])
print("end-effector start", x, y, z)

obs, info = env.reset()
start_ee_pos = obs[0:3]

print("EE start after reset:", start_ee_pos)

# 3. Compute object position RELATIVE to EE
# x = start_ee_pos[0] + s * (0.486089 - 0.378831)
# y = start_ee_pos[1] + s * (0.27058  - 0.015589)


# 4. Set object AFTER reset
# env.unwrapped._set_obj_xyz(np.array([x, y, 0.02]))

env.unwrapped._set_obj_xyz(np.array([0, 0.6, 0.02]))

print("Setting object to:", x, y)


for i in range(68):
    action_7d = df['action'].iloc[i]
    observation_EEF_isaac = df['observation.state'].iloc[i][56:59]
    observation_object_isaac = df['observation.state'].iloc[i][0:3]
        
    # Map action
    action_4d = map_action_7d_to_4d(action_7d, observation_EEF_isaac, observation_object_isaac)
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
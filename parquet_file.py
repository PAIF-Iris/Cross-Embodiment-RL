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


obs, info = env.reset()

# ========== METAWORLD TARGET POSITIONS ==========
METAWORLD_OBJ_INITIAL = np.array([0.0, 0.6, 0.02])  # Fixed object position in MetaWorld

# ========== ISAAC SIM INITIAL STATE (from first frame) ==========
# Extract initial state from first row of parquet
first_state = df['observation.state'].iloc[0]

# EE position is at indices 54:57 based on the schema
ISAAC_EE_INITIAL = np.array([
    first_state[56],  # eef_pos_x
    first_state[57],  # eef_pos_y
    first_state[58]   # eef_pos_z
])

# Object position - using box0 (indices 0:3)
ISAAC_OBJ_INITIAL = np.array([
    first_state[0],   # box0_pos_x
    first_state[1],   # box0_pos_y
    first_state[2]    # box0_pos_z
])

# Compute initial relative position (EE - object) in Isaac Sim
ISAAC_INITIAL_RELATIVE = ISAAC_OBJ_INITIAL - ISAAC_EE_INITIAL
print(f"Isaac Sim object initial: {ISAAC_OBJ_INITIAL}")
print(f"Isaac Sim EE initial: {ISAAC_EE_INITIAL}")
print(f"Isaac Sim initial relative (obj - EE): {ISAAC_INITIAL_RELATIVE}")

def map_gripper(gripper_raw):
    """
    Convert Isaac Sim gripper value to MetaWorld's binary gripper.
    MetaWorld: -1 = close, 1 = open
    Isaac Sim: gripper > 0 means open, gripper < 0 means close
    """
    if gripper_raw == 0:
        gripper = -1.0  # Open gripper
    elif gripper_raw == -1:
        gripper = 1.0  # Close gripper
    else:
        gripper = -1.0   # Open gripper
    return gripper


metaworld_ee_initial = METAWORLD_OBJ_INITIAL - ISAAC_INITIAL_RELATIVE
print(f"Setting MetaWorld EE initial to: {metaworld_ee_initial}")

env.unwrapped.hand_init_pos = np.array([
    metaworld_ee_initial[0],
    metaworld_ee_initial[1],
    metaworld_ee_initial[2]
])

# 2. Reset environment (this applies hand_init_pos)
obs, info = env.reset()

# 3. Verify EE position after reset
metaworld_ee_after_reset = obs[0:3]
print(f"MetaWorld EE after reset: {metaworld_ee_after_reset}")

# 4. Set object to fixed position
env.unwrapped._set_obj_xyz(METAWORLD_OBJ_INITIAL)
print(f"MetaWorld object set to: {METAWORLD_OBJ_INITIAL}")

# 5. Refresh observation after setting object
obs = env.unwrapped._get_obs()
metaworld_ee_current = obs[0:3]
# TODO: Verify the correct index for object position in MetaWorld obs
metaworld_obj_current = obs[4:7]  # This may need adjustment
print(f"MetaWorld EE current: {metaworld_ee_current}")
print(f"MetaWorld object current: {metaworld_obj_current}")

print(f"Starting trajectory execution...")

# ========== TRAJECTORY EXECUTION ==========
for i in range(68):
    # Get Isaac Sim state at this timestep
    current_state = df['observation.state'].iloc[i]
    # if i ==0:
    #     metaworld_obj_current = obs[4:7]
    #     print("obs:", metaworld_obj_current)
    # else:
    #     metaworld_obj_current = observation[4:7]
    #     print("observation:", metaworld_obj_current)
    metaworld_obj_current = METAWORLD_OBJ_INITIAL
    print("observation:", metaworld_obj_current)
    
    # Extract current EE position in Isaac Sim
    isaac_ee_current = np.array([
        current_state[56],  # eef_pos_x
        current_state[57],  # eef_pos_y
        current_state[58]   # eef_pos_z
    ])
    
    # Extract current object position in Isaac Sim
    isaac_obj_current = np.array([
        current_state[0],   # box0_pos_x
        current_state[1],   # box0_pos_y
        current_state[2]    # box0_pos_z
    ])
    
    # Extract gripper action
    action_7d = df['action'].iloc[i]
    gripper_raw = action_7d[6]
    
    current_obs = env.unwrapped._get_obs()
    metaworld_ee_current = current_obs[0:3]
    
    isaac_current_relative = isaac_obj_current - isaac_ee_current
    target_ee_position = metaworld_obj_current - metaworld_ee_current - isaac_current_relative
    
    # Compute delta (action) for MetaWorld
    delta_xyz = target_ee_position - metaworld_ee_current
    
    # Clip deltas to valid range
    #delta_xyz = np.clip(delta_xyz, -1.0, 1.0)
    
    # Get gripper action
    gripper = map_gripper(gripper_raw)
    
    # Construct 4D action for MetaWorld
    action_4d = np.array([delta_xyz[0], delta_xyz[1], delta_xyz[2], gripper], dtype=np.float32)
    
    print(f"Step {i}:")
    print(f"  Isaac EE: {isaac_ee_current}, obj: {isaac_obj_current}, relative: {isaac_current_relative}")
    print(f"  MetaWorld target: {target_ee_position}, current: {metaworld_ee_current}")
    print(f"  Action: {action_4d}")

    # Step environment
    observation, reward, terminated, truncated, info = env.step(action_4d)
    
    env.render()
    time.sleep(0.1)

env.close()
print("Trajectory execution complete!")
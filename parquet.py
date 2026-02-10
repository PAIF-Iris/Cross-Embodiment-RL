# import gymnasium as gym
# import metaworld
# import pandas as pd
# import numpy as np
# import time

# # Load the parquet file
# parquet_path = "/Users/paif_iris/Desktop/metaworld/episode_000000.parquet"  # UPDATE THIS PATH
# df = pd.read_parquet(parquet_path)

# print("="*70)
# print("TRAJECTORY DATA LOADED")
# print("="*70)
# print(f"Total timesteps: {len(df)}")
# print(f"Episode indices: {df['episode_index'].unique()}")
# print(f"Task indices: {df['task_index'].unique()}")
# print(f"Duration: {df['timestamp'].iloc[-1]:.2f} seconds")
# print()

# # Inspect the state vector to understand its structure
# print("="*70)
# print("STATE VECTOR ANALYSIS")
# print("="*70)
# first_state = df['observation.state'].iloc[0]
# print(f"State dimension: {len(first_state)}")
# print(f"\nFirst state breakdown:")
# print(f"  [0:7]   Robot joints?: {first_state[0:7]}")
# print(f"  [7:14]  Joint vels?:   {first_state[7:14]}")
# print(f"  [14:21] EE pose?:      {first_state[14:21]}")
# print(f"  [21:24] Object pos?:   {first_state[21:24]}")
# print(f"  [24:28] Object quat?:  {first_state[24:28]}")
# print()

# # Create MetaWorld environment
# # UPDATE THIS to match your task!
# task_name = "pick-place-v3"  # Options: "pick-place-v3", "door-open-v3", "push-v3", etc.
# render_mode = 'human'

# print("="*70)
# print("CREATING METAWORLD ENVIRONMENT")
# print("="*70)
# print(f"Task: {task_name}")
# env = gym.make('Meta-World/MT1', env_name=task_name, render_mode=render_mode)
# observation, info = env.reset()

# # Inspect MetaWorld observation structure
# print(f"\nMetaWorld observation shape: {observation.shape}")
# print(f"First observation: {observation[:10]}...")  # Show first 10 dims
# print()

# def map_action_7d_to_4d(action_7d, scale_position=1.0):
#     """
#     Map 7D action to 4D MetaWorld action.
    
#     7D: [dx, dy, dz, droll, dpitch, dyaw, gripper]
#     4D: [x, y, z, gripper]
    
#     Args:
#         action_7d: 7D action array
#         scale_position: Scale factor for position deltas (if original actions are too large/small)
#     """
#     x = action_7d[0] * scale_position
#     y = action_7d[1] * scale_position
#     z = action_7d[2] * scale_position
#     gripper = action_7d[6]
    
#     # Clip to reasonable range for MetaWorld
#     action_4d = np.array([x, y, z, gripper], dtype=np.float32)
#     action_4d = np.clip(action_4d, -1.0, 1.0)
    
#     return action_4d

# def analyze_state_structure(state_81d):
#     """
#     Analyze and print the state structure to help identify object positions.
#     """
#     print("\nDetailed state analysis:")
    
#     # Look for position-like values (typically in range [-2, 2] for robot workspace)
#     position_candidates = []
#     for i in range(len(state_81d) - 2):
#         xyz = state_81d[i:i+3]
#         # Check if this looks like a 3D position
#         if np.all(np.abs(xyz) < 2.0):  # Reasonable workspace bounds
#             position_candidates.append((i, xyz))
    
#     print(f"Found {len(position_candidates)} candidate 3D positions:")
#     for idx, pos in position_candidates[:10]:  # Show first 10
#         print(f"  Index {idx:2d}: [{pos[0]:7.4f}, {pos[1]:7.4f}, {pos[2]:7.4f}]")
    
#     return position_candidates

# def extract_object_info(state_81d, object_pos_idx=21):
#     """
#     Extract object position and other info from state.
    
#     Args:
#         state_81d: The 81D state vector
#         object_pos_idx: Starting index for object position (default: 21)
#     """
#     # Extract object position
#     obj_pos = state_81d[object_pos_idx:object_pos_idx+3]
    
#     # Try to extract object orientation (quaternion)
#     obj_quat = state_81d[object_pos_idx+3:object_pos_idx+7]
    
#     return {
#         'position': obj_pos,
#         'quaternion': obj_quat
#     }

# def set_metaworld_object_position(env, obj_pos, obj_name='door'):
#     """
#     Attempt to set object position in MetaWorld environment.
    
#     This is tricky because MetaWorld doesn't always expose easy APIs for this.
#     """
#     try:
#         # Method 1: Direct model manipulation (may work for some environments)
#         if hasattr(env.unwrapped, 'model') and hasattr(env.unwrapped, 'data'):
#             model = env.unwrapped.model
#             data = env.unwrapped.data
            
#             # Try to find the object body
#             try:
#                 body_id = model.body(obj_name).id
#                 data.body(obj_name).xpos[:] = obj_pos
#                 print(f"✓ Set {obj_name} position to {obj_pos}")
#                 return True
#             except:
#                 pass
            
#             # Alternative: try joint positions
#             try:
#                 joint_id = model.joint(obj_name).id
#                 data.joint(obj_name).qpos[:3] = obj_pos
#                 print(f"✓ Set {obj_name} joint position to {obj_pos}")
#                 return True
#             except:
#                 pass
        
#         print(f"✗ Could not set object position (environment may not support it)")
#         return False
        
#     except Exception as e:
#         print(f"✗ Error setting object position: {e}")
#         return False

# # Analyze first state
# print("="*70)
# print("ANALYZING FIRST STATE")
# print("="*70)
# analyze_state_structure(first_state)

# # Try to set initial object position
# obj_info = extract_object_info(first_state)
# print(f"\nExtracted object info:")
# print(f"  Position: {obj_info['position']}")
# print(f"  Quaternion: {obj_info['quaternion']}")

# set_metaworld_object_position(env, obj_info['position'])

# # Replay trajectory
# print("\n" + "="*70)
# print("STARTING TRAJECTORY REPLAY")
# print("="*70)
# print(f"Total steps: {len(df)}")
# print("Press Ctrl+C to stop\n")

# action_scale = 1.0  # Adjust if actions seem too large/small

# try:
#     for i in range(len(df)):
#         # Get data from demonstration
#         action_7d = df['action'].iloc[i]
#         state_81d = df['observation.state'].iloc[i]
#         timestamp = df['timestamp'].iloc[i]
        
#         # Map action
#         action_4d = map_action_7d_to_4d(action_7d, scale_position=action_scale)
        
#         # Step environment
#         observation, reward, terminated, truncated, info = env.step(action_4d)
        
#         # Print progress
#         if i % 20 == 0:
#             print(f"[{i:4d}/{len(df)}] t={timestamp:.2f}s | "
#                   f"action: [{action_4d[0]:6.3f}, {action_4d[1]:6.3f}, {action_4d[2]:6.3f}, {action_4d[3]:6.3f}] | "
#                   f"reward: {reward:.3f}")
        
#         # Render
#         env.render()
        
#         # Small delay for visualization
#         time.sleep(0.01)
        
#         # Check termination
#         if terminated or truncated:
#             print(f"\n⚠ Episode ended early at step {i}/{len(df)}")
#             break
            
# except KeyboardInterrupt:
#     print("\n\n⚠ Interrupted by user")

# print("\n" + "="*70)
# print("REPLAY COMPLETE")
# print("="*70)
# env.close()


import gymnasium as gym
import metaworld
import pandas as pd
import numpy as np
import time

# Load the parquet file
parquet_path = "/Users/paif_iris/Desktop/metaworld/episode_000001.parquet"  # UPDATE THIS PATH
df = pd.read_parquet(parquet_path)

print("="*70)
print("TRAJECTORY DATA LOADED")
print("="*70)
print(f"Total timesteps: {len(df)}")
print(f"Episode indices: {df['episode_index'].unique()}")
print(f"Task indices: {df['task_index'].unique()}")
print(f"Duration: {df['timestamp'].iloc[-1]:.2f} seconds")
print()

# Inspect the state vector to understand its structure
print("="*70)
print("STATE VECTOR ANALYSIS")
print("="*70)
first_state = df['observation.state'].iloc[0]
print(f"State dimension: {len(first_state)}")
print(f"\nFirst state breakdown:")
print(f"  [0:7]   Robot joints?: {first_state[0:7]}")
print(f"  [7:14]  Joint vels?:   {first_state[7:14]}")
print(f"  [14:21] EE pose?:      {first_state[14:21]}")
print(f"  [21:24] Object pos?:   {first_state[21:24]}")
print(f"  [24:28] Object quat?:  {first_state[24:28]}")
print()

# Create MetaWorld environment
# UPDATE THIS to match your task!
task_name = "pick-place-v3"  # Options: "pick-place-v3", "door-open-v3", "push-v3", etc.
render_mode = 'human'

print("="*70)
print("CREATING METAWORLD ENVIRONMENT")
print("="*70)
print(f"Task: {task_name}")
env = gym.make('Meta-World/MT1', env_name=task_name, render_mode=render_mode)
observation, info = env.reset()

# Inspect MetaWorld observation structure
print(f"\nMetaWorld observation shape: {observation.shape}")
print(f"First observation: {observation[:10]}...")  # Show first 10 dims
print()

def map_action_7d_to_4d(action_7d, scale_position=1.0, gripper_threshold=0.5):
    """
    Map 7D action to 4D MetaWorld action.
    
    7D: [dx, dy, dz, droll, dpitch, dyaw, gripper]
    4D: [x, y, z, gripper]
    
    Args:
        action_7d: 7D action array
        scale_position: Scale factor for position deltas (if original actions are too large/small)
        gripper_threshold: Threshold to convert gripper to -1/1 (open/close)
    """
    # Scale position deltas
    x = action_7d[0] * scale_position
    y = action_7d[1] * scale_position
    z = action_7d[2] * scale_position
    
    # Handle gripper - convert to binary open/close
    # Original gripper might be continuous [0,1] or [-1,1]
    gripper_raw = action_7d[6]
    
    # Convert to MetaWorld's binary gripper: -1 = close, 1 = open
    if gripper_raw < gripper_threshold:
        gripper = -1.0  # Close gripper
    else:
        gripper = 1.0   # Open gripper
    
    # Clip position deltas to MetaWorld's range
    action_4d = np.array([x, y, z, gripper], dtype=np.float32)
    action_4d[:3] = np.clip(action_4d[:3], -1.0, 1.0)  # Clip positions only
    
    return action_4d

def analyze_state_structure(state_81d):
    """
    Analyze and print the state structure to help identify object positions.
    """
    print("\nDetailed state analysis:")
    
    # Look for position-like values (typically in range [-2, 2] for robot workspace)
    position_candidates = []
    for i in range(len(state_81d) - 2):
        xyz = state_81d[i:i+3]
        # Check if this looks like a 3D position
        if np.all(np.abs(xyz) < 2.0):  # Reasonable workspace bounds
            position_candidates.append((i, xyz))
    
    print(f"Found {len(position_candidates)} candidate 3D positions:")
    for idx, pos in position_candidates[:10]:  # Show first 10
        print(f"  Index {idx:2d}: [{pos[0]:7.4f}, {pos[1]:7.4f}, {pos[2]:7.4f}]")
    
    return position_candidates

def extract_object_info(state_81d, object_pos_idx=21):
    """
    Extract object position and other info from state.
    
    Args:
        state_81d: The 81D state vector
        object_pos_idx: Starting index for object position (default: 21)
    """
    # Extract object position
    obj_pos = state_81d[object_pos_idx:object_pos_idx+3]
    
    # Try to extract object orientation (quaternion)
    obj_quat = state_81d[object_pos_idx+3:object_pos_idx+7]
    print("obj_pos:", obj_pos)
    print("obj_quat:", obj_quat)
    
    return {
        'position': obj_pos,
        'quaternion': obj_quat
    }

def set_metaworld_object_position(env, obj_pos, obj_name='door'):
    """
    Attempt to set object position in MetaWorld environment.
    
    This is tricky because MetaWorld doesn't always expose easy APIs for this.
    """
    try:
        # Method 1: Direct model manipulation (may work for some environments)
        if hasattr(env.unwrapped, 'model') and hasattr(env.unwrapped, 'data'):
            model = env.unwrapped.model
            data = env.unwrapped.data
            
            # Try to find the object body
            try:
                body_id = model.body(obj_name).id
                data.body(obj_name).xpos[:] = obj_pos
                print(f"✓ Set {obj_name} position to {obj_pos}")
                return True
            except:
                pass
            
            # Alternative: try joint positions
            try:
                joint_id = model.joint(obj_name).id
                data.joint(obj_name).qpos[:3] = obj_pos
                print(f"✓ Set {obj_name} joint position to {obj_pos}")
                return True
            except:
                pass
        
        print(f"✗ Could not set object position (environment may not support it)")
        return False
        
    except Exception as e:
        print(f"✗ Error setting object position: {e}")
        return False

# Analyze first state
print("="*70)
print("ANALYZING FIRST STATE")
print("="*70)
analyze_state_structure(first_state)

# Try to set initial object position
obj_info = extract_object_info(first_state)
print(f"\nExtracted object info:")
print(f"  Position: {obj_info['position']}")
print(f"  Quaternion: {obj_info['quaternion']}")

set_metaworld_object_position(env, obj_info['position'])

# Replay trajectory
print("\n" + "="*70)
print("ACTION STATISTICS ANALYSIS")
print("="*70)

# Analyze action magnitudes to determine proper scaling
all_actions = np.array([df['action'].iloc[i] for i in range(len(df))])
pos_actions = all_actions[:, :3]  # position deltas
gripper_actions = all_actions[:, 6]  # gripper

action_scale = 20.0  
print(f"Position deltas (dx, dy, dz):")
print(f"  Mean abs: {np.mean(np.abs(pos_actions), axis=0)}")
print(f"  Max abs:  {np.max(np.abs(pos_actions), axis=0)}")
print(f"  Min:      {np.min(pos_actions, axis=0)}")
print(f"  Max:      {np.max(pos_actions, axis=0)}")
print(f"\nGripper actions:")
print(f"  Min: {gripper_actions.min():.4f}")
print(f"  Max: {gripper_actions.max():.4f}")
print(f"  Mean: {gripper_actions.mean():.4f}")
print(f"  Unique values: {np.unique(gripper_actions)[:10]}")  # Show first 10 unique values

print("\n" + "="*70)
print("STARTING TRAJECTORY REPLAY")
print("="*70)
print(f"Total steps: {len(df)}")
print(f"Action scale factor: {action_scale}")
print("Press Ctrl+C to stop\n")

action_scale = 20.0  # Scale up the tiny normalized actions for MetaWorld
# Try values between 10-50 if movement is still too small/large

try:
    for i in range(len(df)):
        # Get data from demonstration
        action_7d = df['action'].iloc[i]
        state_81d = df['observation.state'].iloc[i]
        timestamp = df['timestamp'].iloc[i]
        
        # Map action
        action_4d = map_action_7d_to_4d(action_7d, scale_position=action_scale)
        
        # Step environment
        observation, reward, terminated, truncated, info = env.step(action_4d)
        
        # Print progress
        if i % 20 == 0:
            print(f"[{i:4d}/{len(df)}] t={timestamp:.2f}s | "
                  f"action: [{action_4d[0]:6.3f}, {action_4d[1]:6.3f}, {action_4d[2]:6.3f}, {action_4d[3]:6.3f}] | "
                  f"reward: {reward:.3f}")
        
        # Render
        env.render()
        
        # Small delay for visualization
        time.sleep(0.01)
        
        # Check termination
        if terminated or truncated:
            print(f"\n⚠ Episode ended early at step {i}/{len(df)}")
            break
            
except KeyboardInterrupt:
    print("\n\n⚠ Interrupted by user")

print("\n" + "="*70)
print("REPLAY COMPLETE")
print("="*70)
env.close()
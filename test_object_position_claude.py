import gymnasium as gym
import metaworld
import h5py
import numpy as np

def infer_object_position_from_dataset(h5_path):
    """Infer object position from where gripper closes before first reward"""
    with h5py.File(h5_path, "r") as f:
        ee_positions = f["ee_cartesian_pos_ob"][:, 0:3]
        rewards = f["reward"][:]
        actions = f["actions"][:]
        
        reward_indices = np.where(rewards == 1)[0]
        print(f"Skill completions at timesteps: {reward_indices}")
        
        if len(reward_indices) == 0:
            return None, None
        
        first_reward_idx = reward_indices[0]
        
        # Find last gripper close before first reward
        grasp_candidates = []
        for i in range(min(first_reward_idx, len(actions))):
            if actions[i, 3] == 2:  # Gripper close
                grasp_candidates.append(i)
        
        if grasp_candidates:
            grasp_idx = grasp_candidates[-1]
            inferred_object_pos = ee_positions[grasp_idx].copy()
            print(f"Inferred object at grasp timestep {grasp_idx}: {inferred_object_pos}")
        else:
            grasp_idx = max(0, first_reward_idx - 10)
            inferred_object_pos = ee_positions[grasp_idx].copy()
            print(f"Inferred object at timestep {grasp_idx}: {inferred_object_pos}")
        
        # Goal is where object ends up
        last_reward_idx = reward_indices[-1]
        inferred_goal_pos = ee_positions[last_reward_idx].copy()
        print(f"Inferred goal at timestep {last_reward_idx}: {inferred_goal_pos}")
        
        return inferred_object_pos, inferred_goal_pos

path = "/Users/paif_iris/Desktop/metaworld/test_data_jaco_play.h5"
render_mode = 'human'

print("="*60)
print("Inferring positions from Jaco dataset")
print("="*60)

jaco_obj_pos, jaco_goal_pos = infer_object_position_from_dataset(path)

# Create environment
env = gym.make('Meta-World/MT1', env_name="pick-place-v3", render_mode=render_mode)
observation, info = env.reset(seed=42)

# Get the unwrapped environment
env_unwrapped = env.unwrapped

print(f"\nMetaWorld default positions:")
print(f"  Object: {observation[3:6]}")
print(f"  Goal: {observation[6:9]}")

print("\n" + "="*60)
print("Setting object position")
print("="*60)

# Set desired positions on MetaWorld table
# MetaWorld table is roughly X:[-0.3, 0.3], Y:[0.4, 0.8], Z:[0.02 for surface]
desired_obj_xyz = np.array([0.0, 0.6, 0.02])  # Center of table
desired_goal_xyz = np.array([0.1, 0.7, 0.15])  # Raised goal position

print(f"Target object position: {desired_obj_xyz}")
print(f"Target goal position: {desired_goal_xyz}")

# Method 1: Try MetaWorld's built-in method
try:
    env_unwrapped._set_obj_xyz(desired_obj_xyz)
    print("✓ Used _set_obj_xyz()")
except Exception as e:
    print(f"✗ _set_obj_xyz() failed: {e}")

# Method 2: Set goal
try:
    env_unwrapped._target_pos = desired_goal_xyz
    env_unwrapped.goal = desired_goal_xyz
    print("✓ Set goal via _target_pos and goal attributes")
except Exception as e:
    print(f"✗ Goal setting failed: {e}")

# Method 3: Direct qpos manipulation
# Find where object position is stored in qpos
print(f"\nMuJoCo model info:")
print(f"  Total qpos size: {env_unwrapped.model.nq}")
print(f"  Object body name: 'obj'")

# Get object body ID
obj_body_id = env_unwrapped.model.body('obj').id
print(f"  Object body ID: {obj_body_id}")

# Find object joint (free joint for the object)
# Look for a joint connected to the object
for i in range(env_unwrapped.model.njnt):
    jnt_bodyid = env_unwrapped.model.jnt_bodyid[i]
    if jnt_bodyid == obj_body_id:
        jnt_type = env_unwrapped.model.jnt_type[i]
        jnt_qposadr = env_unwrapped.model.jnt_qposadr[i]
        print(f"  Found object joint {i}: type={jnt_type}, qpos_start={jnt_qposadr}")
        
        # For free joint (type 0), qpos is [x, y, z, qw, qx, qy, qz] (7 values)
        if jnt_type == 0:  # Free joint
            print(f"  Setting object position directly in qpos[{jnt_qposadr}:{jnt_qposadr+3}]")
            env_unwrapped.data.qpos[jnt_qposadr:jnt_qposadr+3] = desired_obj_xyz
            # Keep quaternion as is (identity or current)
            # qpos[jnt_qposadr+3:jnt_qposadr+7] is the quaternion
            
# Forward the simulation to apply changes
env_unwrapped.model.forward(env_unwrapped.data)

# Get updated observation
observation = env_unwrapped._get_obs()

print(f"\nAfter setting positions:")
print(f"  Actual object: {observation[3:6]}")
print(f"  Actual goal: {observation[6:9]}")

obj_match = np.allclose(observation[3:6], desired_obj_xyz, atol=0.02)
goal_match = np.allclose(observation[6:9], desired_goal_xyz, atol=0.02)

print(f"  Object match: {obj_match}")
print(f"  Goal match: {goal_match}")

# Load actions
with h5py.File(path, "r") as f:
    actions = f["actions"][:]
    rewards = f["reward"][:]

def convert_jaco_action(jaco_action):
    mw_action = jaco_action[:3].copy()
    gripper_cmd = int(jaco_action[3])
    gripper = 1.0 if gripper_cmd == 0 else (-1.0 if gripper_cmd == 2 else 0.0)
    return np.append(mw_action, gripper)

print("\n" + "="*60)
print("Running trajectory (first 80 steps)")
print("="*60)

for i in range(min(len(actions), 80)):
    mw_action = convert_jaco_action(actions[i])
    observation, reward, terminated, truncated, info = env.step(mw_action)
    
    ee_pos = observation[:3]
    obj_pos = observation[3:6]
    goal_pos = observation[6:9]
    
    if i % 10 == 0 or rewards[i] == 1:
        print(f"Step {i:3d}: "
              f"EE=[{ee_pos[0]:+.2f},{ee_pos[1]:+.2f},{ee_pos[2]:+.2f}], "
              f"Obj=[{obj_pos[0]:+.2f},{obj_pos[1]:+.2f},{obj_pos[2]:+.2f}], "
              f"EE→Obj={np.linalg.norm(ee_pos - obj_pos):.3f}m, "
              f"Grip={mw_action[3]:+.1f}, "
              f"MW_R={reward:.2f}, Jaco_R={rewards[i]}")
    
    env.render()
    
    if terminated or truncated:
        print(f"Episode ended at step {i}")
        break

env.close()
print("\nDone!")
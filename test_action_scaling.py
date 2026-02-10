# test_action_scaling.py
import gymnasium as gym
import metaworld
import h5py
import numpy as np
import time

DATASET_FILE = "/Users/paif_iris/Desktop/metaworld/hdf5/berkeley_sawyer_traj973.hdf5"
# Test different scaling factors
SCALING_FACTORS = [1.0, 5.0, 10.0, 20.0, 10000.0]

print("="*80)
print("ACTION SCALING EXPERIMENT")
print("="*80)

env = gym.make('Meta-World/MT1', env_name="pick-place-v3", render_mode='human')

for scale_factor in SCALING_FACTORS:
    print(f"\n{'='*80}")
    print(f"Testing with scale factor: {scale_factor}x")
    print(f"{'='*80}")
    
    observation, info = env.reset()
    
    with h5py.File(DATASET_FILE, "r") as f:
        actions = f["policy/actions"][:]
        
        total_reward = 0
        total_movement = 0
        
        for i in range(min(30, len(actions))):  # Play first 30 steps
            robonet_action = actions[i]
            
            # Scale the action
            scaled_action = robonet_action * scale_factor
            
            # Clip to valid range
            metaworld_action = np.clip(scaled_action, -1.0, 1.0)
            
            # Track how much clipping occurred
            clipped = np.any(scaled_action != metaworld_action)
            
            # Get robot position before action
            pos_before = env.unwrapped._get_pos_objects()
            
            # Execute action
            observation, reward, terminated, truncated, info = env.step(metaworld_action)
            
            # Get robot position after action
            pos_after = env.unwrapped._get_pos_objects()
            movement = np.linalg.norm(pos_after - pos_before)
            
            total_reward += reward
            total_movement += movement
            
            if i < 5:  # Print first 5 steps
                print(f"  Step {i+1}: "
                      f"orig={np.linalg.norm(robonet_action):.4f}, "
                      f"scaled={np.linalg.norm(scaled_action):.4f}, "
                      f"movement={movement:.4f}, "
                      f"{'CLIPPED' if clipped else 'ok'}")
            
            env.render()
            time.sleep(0.05)
            
            if terminated or truncated:
                break
        
        avg_movement = total_movement / min(30, len(actions))
      
    time.sleep(2)  # Pause between tests

env.close()
import gymnasium as gym
import metaworld
import h5py
import numpy as np
import time
import glob
from pathlib import Path

dataset_dir = "/Users/paif_iris/Desktop/metaworld/dataset/downloads/extracted/"
hdf5_files = glob.glob(f"{dataset_dir}/**/hdf5/*.hdf5", recursive=True)

print(f"Found {len(hdf5_files)} RoboNet trajectory files")

render_mode = 'human'
env = gym.make('Meta-World/MT1', env_name="reach-v3", render_mode=render_mode)

print(f"\nMetaWorld Environment: reach-v3")
print(f"  Action space: {env.action_space}")
print(f"  Observation space: {env.observation_space}")

# Configuration
NUM_TRAJECTORIES = 700 
ADD_DELAY = 0.03  
RESET_ENV = True 

observation, info = env.reset()
total_reward = 0
total_steps = 0

for traj_idx, traj_path in enumerate(hdf5_files[:NUM_TRAJECTORIES]):
    print(f"\n{'='*60}")
    print(f"Playing trajectory {traj_idx + 1}/{NUM_TRAJECTORIES}")
    print(f"File: {Path(traj_path).name}")
    print(f"{'='*60}")
    
    try:
        with h5py.File(traj_path, "r") as f:
    
            actions = f["policy/actions"][:]
            num_steps = len(actions)
            print(f"  Trajectory length: {num_steps} steps")
            
            traj_reward = 0
            
            for i in range(num_steps):
                robonet_action = actions[i]
                
                # Clip to valid range
                metaworld_action = np.clip(
                    robonet_action,
                    env.action_space.low,
                    env.action_space.high
                )
                
                observation, reward, terminated, truncated, info = env.step(metaworld_action)
                traj_reward += reward
                total_reward += reward
                total_steps += 1
                
                # Render
                env.render()
                
                # Add delay
                time.sleep(ADD_DELAY)
                
                # Check if episode ended
                if terminated or truncated:
                    print(f"    Episode ended at step {i+1}/{num_steps}")
                    if RESET_ENV:
                        observation, info = env.reset()
                    break
            
            print(f"  Trajectory reward: {traj_reward:.3f}")
            print(f"  Success: {info.get('success', 0.0)}")
            
            # Reset between trajectories
            if RESET_ENV and (traj_idx < NUM_TRAJECTORIES - 1):
                observation, info = env.reset()
                time.sleep(0.5)  # Brief pause between trajectories
    
    except Exception as e:
        print(f"  ⚠️  Error: {e}")
        continue


env.close()
import math

import numpy as np
import pandas as pd
import gymnasium as gym
import metaworld
import time

path = "/Users/paif_iris/Desktop/metaworld/parquet/episode_001046.parquet"
df = pd.read_parquet(path)

render_mode = 'human'
env = gym.make('Meta-World/MT1', env_name="pick-place-v3", render_mode=render_mode, seed=42)

EE_XYZ_Start = df['observation.state'].iloc[0][56:59]
EE_XYZ_End = df['observation.state'].iloc[45][56:59]

meta_obj_start = np.array([0.3, 0.6, 0.02])
meta_obj_start_adjusted = np.array([0.3, 0.6, 0.02])

relative_expert = EE_XYZ_End - EE_XYZ_Start
meta_eef_start  = meta_obj_start_adjusted - relative_expert

print("EE_XYZ_Start:", EE_XYZ_Start)
print("EE_XYZ_End:", EE_XYZ_End)
print("EE start metaworld:", meta_eef_start)

def reset_and_restore(env, ee_pos, obj_pos):
    env.unwrapped.hand_init_pos = ee_pos.copy()
    obs, info = env.reset()
    env.unwrapped._set_obj_xyz(obj_pos)
    return obs

obs = reset_and_restore(env, meta_eef_start, meta_obj_start)
print("obs[0:3] after reset:", obs[0:3])
print("object pos:", env.unwrapped._get_pos_objects())


def move_to_target(env, target_pos, gripper, obj_pos, settle_steps=2):
    for _ in range(settle_steps):

        env.unwrapped.data.mocap_pos[0][:] = target_pos
        action = np.array([0.0, 0.0, 0.0, gripper], dtype=np.float32)
        obs, reward, terminated, truncated, info = env.step(action)
        print("observation:", obs[0:3])
        env.render()
        time.sleep(0.1)
        
        if truncated:
            print("  [truncated] resetting and restoring position...")
            obs = reset_and_restore(env, target_pos, obj_pos)
            env.unwrapped.data.mocap_pos[0][:] = target_pos

        if terminated:
            print("  [terminated] task completed!")
            return obs, reward, terminated, truncated, info

    return obs, reward, terminated, truncated, info

target_pos = meta_eef_start.copy()
obj_pos = meta_obj_start.copy()

for i in range(len(df)-1):
    delta = df['observation.state'].iloc[i+1][56:59] - df['observation.state'].iloc[i][56:59]
    gripper_raw = df['action'].iloc[i][6]
    gripper = 1.0 if gripper_raw == -1 else -1.0

    target_pos = target_pos + delta

    print(f"Step {i}: target={target_pos}, gripper={gripper}")

    obs, reward, terminated, truncated, info = move_to_target(
        env,
        target_pos=target_pos,
        gripper=gripper,
        obj_pos=obj_pos
    )

    print(f"  -> actual EE pos: {obs[0:3]}")

env.close()
import gymnasium as gym
import metaworld
import h5py
import numpy as np

# -----------------------------
# Config
# -----------------------------
path = "/Users/paif_iris/Desktop/metaworld/test_data_jaco_play.h5"
render_mode = "human"

ENV_NAME = "pick-place-v3"

# -----------------------------
# Load demonstration
# -----------------------------
with h5py.File(path, "r") as f:
    actions = np.array(f["actions"])                     # (T, 4)
    rewards = np.array(f["reward"])                      # (T,)
    ee_pos = np.array(f["ee_cartesian_pos_ob"])[:, :3]   # (T, 3)

# -----------------------------
# Infer key timesteps
# -----------------------------
reward_idxs = np.where(rewards == 1)[0]
assert len(reward_idxs) >= 2, "Need at least grasp + goal reward"

t_grasp = reward_idxs[0]
t_goal = reward_idxs[-1]

object_pos = ee_pos[t_grasp]
goal_pos = ee_pos[t_goal]

print("Inferred object position:", object_pos)
print("Inferred goal position:", goal_pos)

# -----------------------------
# Create MetaWorld env
# -----------------------------
env = gym.make(
    "Meta-World/MT1",
    env_name=ENV_NAME,
    render_mode=render_mode
)

obs, info = env.reset()

# -----------------------------
# Inject inferred task geometry
# -----------------------------
# MetaWorld stores object init position here
env.unwrapped.obj_init_pos = object_pos.copy()

# Some envs also require resetting internal state
env.unwrapped._target_pos = goal_pos.copy()

# Reset again to apply
obs, info = env.reset()

# -----------------------------
# Replay policy
# -----------------------------
for t in range(len(actions)):
    action = actions[t]

    obs, reward, terminated, truncated, info = env.step(action)
    env.render()

    if terminated or truncated:
        break

env.close()

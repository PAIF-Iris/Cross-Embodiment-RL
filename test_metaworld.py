import gymnasium as gym
import metaworld
import h5py
path = "/Users/paif_iris/Desktop/metaworld/test_data_jaco_play.h5"

render_mode = 'human'
env = gym.make('Meta-World/MT1', env_name="pick-place-v3", render_mode=render_mode)

observation, info = env.reset()
for i in range(2380):
    with h5py.File(path, "r") as f:
        actions = f["actions"]
        a = actions[i]
    print("action:", a)
    observation, reward, terminated, truncated, info = env.step(a)
    env.render()

env.close()
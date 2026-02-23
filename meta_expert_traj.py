import gymnasium as gym
import metaworld
from metaworld.policies.sawyer_reach_v3_policy import SawyerReachV3Policy
import time
import numpy as np

env = gym.make('Meta-World/MT1', env_name='reach-v3', render_mode='human', seed=42)

obs, info = env.reset()

policy = SawyerReachV3Policy()

done = False

while not done:
    a = policy.get_action(obs)
    #print("Action:", a)
    obs, _, _, _, info = env.step(np.array([0.1, 0.1, 0.1, 0.1]))
    print("Observation:", obs)
    env.render()
    time.sleep(0.01)
    done = int(info['success']) == 1
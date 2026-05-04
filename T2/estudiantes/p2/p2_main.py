from dwpt_rl import DWPTContinuousShareEnv, DWPTParams
import numpy as np
import matplotlib.pyplot as plt

params = DWPTParams()

env = DWPTContinuousShareEnv(params=params, seed=0, system_knowledge='full')




obs = env.reset(variable=False)
infos = []

episodes = 500

for ep in range(episodes):
    terminated = False
    while not terminated:
        action = [np.random.uniform(-1, 1) for _ in range(2)]
        obs_dict, terminated = env.step(action)

        print(obs_dict)


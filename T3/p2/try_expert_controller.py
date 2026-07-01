from scripts.fermentation_env import WineFermentationEnv
from scripts.expert_controller import WineExpertController
import matplotlib.pyplot as plt
import numpy as np
from scripts.quality_functions import maceration_extraction, get_wine_quality_score

from scripts.expert_controller_v2 import SmoothController

import os

path = 'results/p2' 
os.makedirs(path, exist_ok=True)

# controlador experto o experto suave
input = input("Elija el controlador a usar [1]-Experto    [2]-Experto Suave: ")
controller = "Experto" if input == "1" else "Experto Suave"

# Randomize initial conditions
S0 = np.random.uniform(210, 260)
T0 = np.random.uniform(15, 22)
N0 = np.random.uniform(150, 280)

# Instantiate the environment
env = WineFermentationEnv(S0=S0, T0=T0, N0=N0)

# Instantiate the Expert Controller
if controller == "Experto":
    ctrl = WineExpertController()
    ctrl.reset()
else:
    ctrl = SmoothController()
    ctrl.reset()

obs  = env.reset(S0=S0, T0=T0, N0=N0)
done = False

obs_list = [obs.copy()]
act_list = []

while not done:
    t      = float(obs[7])
    action = ctrl(obs, t)
    obs, done, info = env.step(action)
    act_list.append(action.copy())
    obs_list.append(obs.copy())

obs_arr = np.array(obs_list[:-1], dtype=np.float32)
act_arr = np.array(act_list,  dtype=np.float32)
names = ['Res Sugar', 'Etanol', 'Viable yeast', 'Dead yeast', 'Neast-assimilable Nitrogen', 'Disolved CO2', 'T', 't']
act_names = ['T_setpoint', 'Nitrogen']

# Get trajectory
trajectory = env.get_trajectory()

# Maceration extraction
profile = maceration_extraction(trajectory)
print(profile)

################ Ground Truth of preferences --- DO NOT USE FOR TRAINING, JUST FOR EVALUATION --- ########
score = get_wine_quality_score(profile)
print('Score:', score)

# cond iniciales
conds = 'S0=' + str(round(S0, 3)) + '_T0=' + str(round(T0, 3)) + '_N0=' + str(round(N0, 3))
print(conds)

# guardar score en results
np.save(os.path.join(path, 'score_' + controller + '.npy'), np.array(score))

plt.figure(figsize=[15, 8])
plt.title('Observations for ' + conds)
for i in range(obs_arr.shape[1]):
    plt.subplot(2, 4, i+1)
    plt.plot(obs_arr[:, i], label=names[i])
    plt.legend()
    plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(path, 'obs_' + controller + '.png'))

plt.figure(figsize=[15, 8])
plt.title('Actions for ' + conds)
for i in range(act_arr.shape[1]):
    plt.subplot(2, 1, i+1)
    plt.plot(act_arr[:, i], label=act_names[i])
    plt.legend()
    plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(path, 'act_' + controller + '.png'))
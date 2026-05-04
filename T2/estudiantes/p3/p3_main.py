import numpy as np
import matplotlib.pyplot as plt
from practice_env import make_person_params, make_env


# Option A: deterministic reference person.
#person_mode = "fixed"

# Option B: one random but reproducible person for the whole training run.
person_mode = "random_once"

# Option C: new random person every episode. Use eval_episodes > 1.
#person_mode = "random_each_episode"

# With these you can generate many persons with similar characteristics
seed = 0
params = make_person_params(
        person_mode=person_mode,
        seed=seed,
        athletic_level='low', # low, medium, high
        recovery_speed='high', # low, medium, high
        fitness_retention='high', # low, medium, high
    )

env = make_env(params=params, seed=seed)



actions = [1, 0.1, 0.01, 0.5, 2, 3, 2.5]
performance = {}
measured_performance = {}
for i in range(len(actions)):
    done = False
    truncated = False
    measured_performance[i] = []
    performance[i] = []
    env.reset()
    while not (done or truncated):
        action = [actions[i]]
        obs, reward, done, truncated, info = env.step(action)
        performance[i].append(env.true_performance)


plt.figure()
for i in range(len(actions)):
    plt.plot(performance[i], label="Action {}".format(actions[i]))
plt.legend()
plt.grid()
plt.tight_layout()

plt.show()
from evacuation import EvacuationEnv
from collections import defaultdict
import random
import numpy as np
import matplotlib.pyplot as plt
import os

results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(results_dir, exist_ok=True)


edges = [
        ("A", "B", 2.6, 3, 0), ("B", "C", 2.4, 2, 0), ("C", "D", 2.2, 2, 0), ("D", "S1", 2.0, 2, 0),
        ("E", "F", 2.8, 4, 0), ("F", "G", 2.8, 4, 0), ("G", "H", 2.8, 4, 0), ("H", "S2", 2.5, 3, 0),
        ("E", "I", 3.2, 5, 0), ("I", "J", 3.2, 5, 0), ("J", "H", 3.2, 5, 0),
        ("A", "E", 3.0, 2, 0), ("B", "F", 2.6, 2, 0), ("C", "G", 2.6, 2, 0), ("D", "H", 3.0, 2, 0),
        ("B", "E", 2.8, 2, 0), ("C", "F", 2.6, 1, 0),("C", "H", 3.4, 2, 0),
        ("F", "I", 2.8, 3, 0), ("G", "J", 2.8, 3, 0), ]


positions = {  # For plotting
        "A": (0, 2),
        "B": (2, 2),
        "C": (4, 2),
        "D": (6, 2),

        "E": (0, 0),
        "F": (2, 0),
        "G": (4, 0),
        "H": (6, 0),

        "I": (1, -2),
        "J": (5, -2),

        "S1": (8, 2),
        "S2": (8, 0),
    }


## setup
env = EvacuationEnv(n_agents=50, s0=["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]*5,
                        edges=edges, node_positions=positions)
# print(env.get_neighbors('A'))

###---------------- política epsilon-greedy ----------------###
def epsilon_greedy_policy(q_table, state, valid_actions, epsilon):
    if not valid_actions:
        return 0

    # If evaluation graph has higher node degree than the training graph,
    # expand this state's Q-vector so valid action indexes are safe.
    max_valid_action = max(valid_actions)
    if max_valid_action >= len(q_table[state]):
        old_q = q_table[state]
        pad = max_valid_action + 1 - len(old_q)
        q_table[state] = np.pad(old_q, (0, pad), mode="constant")

    if random.random() < epsilon:
        return random.choice(valid_actions)
    else:
        q_values = [q_table[state][action] for action in valid_actions]
        max_q = max(q_values)
        best_actions = [action for action in valid_actions if q_table[state][action] == max_q]
        return random.choice(best_actions)


###---------------- Q-Learning ----------------###
def train_q_learning(env, n_episodes=1000, max_steps=60,
                     alpha=0.1, gamma=0.95,
                     eps_start=0.2, eps_end=0.01, eps_decay=0.995):
    q_table = defaultdict(lambda: np.zeros(env.max_degree))
    rewards_hist, saved_hist = [], []
    epsilon = eps_start

    for ep in range(n_episodes):
        states = env.reset()
        ep_reward = 0

        for _ in range(max_steps):

            if all(env.done_list):
                break

            valid = env.get_valid_actions_for_current_state()
            # cada agente elige acción con eplison greedy propio
            actions = [epsilon_greedy_policy(q_table, state, valid_actions, epsilon) for state, valid_actions in zip(states, valid)]

            next_states, rewards, _, _, _, done_list, _, _ = env.step(actions)
            next_valid = env.get_valid_actions_for_current_state()

            for i in range(env.n_agents):
                if rewards[i] is None:
                    continue

                # calcular target OFF-policy
                if done_list[i]:
                    target = rewards[i]
                else:
                    target = rewards[i] + gamma * max(q_table[next_states[i]][a] for a in next_valid[i])

                # update Q-table
                q_table[states[i]][actions[i]] += alpha * (target - q_table[states[i]][actions[i]])

                ep_reward += rewards[i]

            states = next_states
        
        rewards_hist.append(ep_reward)
        saved_hist.append(sum(env.done_list))
        epsilon = max(epsilon * eps_decay, eps_end) 

    return q_table, rewards_hist, saved_hist


###---------------- SARSA ----------------###
def train_sarsa(env, n_episodes=1000, max_steps=60,
                 alpha=0.1, gamma=0.95,
                 eps_start=0.2, eps_end=0.01, eps_decay=0.995):
    q_table = defaultdict(lambda: np.zeros(env.max_degree))
    rewards_hist, saved_hist = [], []
    epsilon = eps_start

    for ep in range(n_episodes):
        states = env.reset()
        
        valid = env.get_valid_actions_for_current_state()
        # cada agente elige acción con eplison greedy propio
        actions = [epsilon_greedy_policy(q_table, state, valid_actions, epsilon) for state, valid_actions in zip(states, valid)]
        
        ep_reward = 0.0

        for _ in range(max_steps):

            if all(env.done_list):
                break

            next_states, rewards, _, _, _, done_list, _, _ = env.step(actions)
            next_valid = env.get_valid_actions_for_current_state()

            # on policy
            # next_actions con epsilon greedy para cada agente
            next_actions = [epsilon_greedy_policy(q_table, next_states[i], next_valid[i], epsilon) for i in range(env.n_agents)]

            for i in range(env.n_agents):
                if rewards[i] is None:
                    continue

                # calcular target on-policy

                if done_list[i]:
                    target = rewards[i]
                else:
                    next_action = next_actions[i]
                    target = rewards[i] + gamma * q_table[next_states[i]][next_action] # NO MAX!

                # update Q-table
                q_table[states[i]][actions[i]] += alpha * (target - q_table[states[i]][actions[i]])

                ep_reward += rewards[i]

            states = next_states
            actions = next_actions

        rewards_hist.append(ep_reward)
        saved_hist.append(sum(env.done_list))
        epsilon = max(epsilon * eps_decay, eps_end)

    return q_table, rewards_hist, saved_hist


###---------------- PLOTS ----------------###
def moving_average(data, window_size=50):
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

def plot_progress(q_rewards, q_saved, s_rewards, s_saved):
    # plot rewards
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(q_rewards, label="Q-Learning Rewards", alpha=0.25)
    axes[0].plot(moving_average(q_rewards), label="Q-Learning MA", linewidth=2)
    axes[0].plot(s_rewards, label="SARSA Rewards", alpha=0.25)
    axes[0].plot(moving_average(s_rewards), label="SARSA MA", linewidth=2)
    axes[0].set_title("Episodic Rewards")
    axes[0].set_xlabel("Episode")
    axes[0].set_ylabel("Total Reward")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    # plot saved
    axes[1].plot(q_saved, label="Q-Learning Saved", alpha=0.25)
    axes[1].plot(moving_average(q_saved), label="Q-Learning Saved MA", linewidth=2)
    axes[1].plot(s_saved, label="SARSA Saved", alpha=0.25)
    axes[1].plot(moving_average(s_saved), label="SARSA Saved MA", linewidth=2)
    axes[1].set_title("People Saved")
    axes[1].set_xlabel("Episode")
    axes[1].set_ylabel("Number of People Saved")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    fig.tight_layout()
    plt.show()

    fig.savefig(results_dir + "/evacuation_training_progress.png", dpi=150)

def compare_algorithms(q_rewards, q_saved, s_rewards, s_saved, window_size=50):
    # última window de recompensa y personas salvadas
    q_rewards_avg = np.mean(q_rewards[-window_size:])
    s_rewards_avg = np.mean(s_rewards[-window_size:])
    q_saved_avg = np.mean(q_saved[-window_size:])
    s_saved_avg = np.mean(s_saved[-window_size:])

    print("=== Q-Learning vs SARSA final ===")
    print(f"Q-Learning: Final reward = {q_rewards_avg:.2f}, People saved = {q_saved_avg:.2f}")
    print(f"SARSA: Final reward = {s_rewards_avg:.2f}, People saved = {s_saved_avg:.2f}")


def evaluate_policy_over_time(env, q_table, max_time_seconds=300, selected_agents=None):
    # Evaluacion sin exploracion: politica greedy segun la Q-table aprendida
    if selected_agents is None:
        selected_agents = [0, 3, 6, 9]

    states = env.reset()
    times = [0]
    saved_over_time = [sum(env.done_list)]

    trajectories = {i: [states[i][0]] for i in selected_agents}

    while env.global_time < max_time_seconds and not all(env.done_list):
        valid = env.get_valid_actions_for_current_state()
        actions = [
            epsilon_greedy_policy(q_table, state, valid_actions, epsilon=0.0)
            for state, valid_actions in zip(states, valid)
        ]

        next_states, _, _, _, global_time, done_list, _, _ = env.step(actions)
        states = next_states

        times.append(global_time)
        saved_over_time.append(sum(done_list))

        for i in selected_agents:
            trajectories[i].append(states[i][0])

    return times, saved_over_time, trajectories


def plot_saved_vs_time(times, saved_over_time, title_suffix):
    fig, ax = plt.subplots(1, 1, figsize=(8, 4))
    ax.plot(times, saved_over_time, marker="o", linewidth=1.5)
    ax.set_title(f"People saved vs time ({title_suffix})")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Saved people")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(results_dir, "evacuation_saved_vs_time.png"), dpi=150)


def plot_trajectories(
    trajectories,
    edges=None,
    positions=None,
    ax=None,
    title="Agent trajectories",
    save_path = None,
):
    if edges is None:
        edges = globals()["edges"]
    if positions is None:
        positions = globals()["positions"]

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    else:
        fig = ax.figure

    max_capacity = max(capacity for _, _, _, capacity, _ in edges) # lineas más anchas com más cap
    for u, v, _, capacity, _ in edges:
        x1, y1 = positions[u]
        x2, y2 = positions[v]
        width = 1.0 + 5.0 * (capacity / max_capacity)
        ax.plot([x1, x2], [y1, y2], color="0.75", linewidth=width, alpha=0.65, zorder=0)

        # tag
        mx, my = (x1 + x2) / 2, (y1 + y2) / 2
        ax.text(mx, my + 0.08, f"C={capacity}", fontsize=7, color="0.35", ha="center", zorder=1)

    # nodos base
    for node, (x, y) in positions.items():
        color = "lightgreen" if node.startswith("S") else "lightblue"
        ax.scatter(x, y, s=300, c=color, edgecolors="k")
        ax.text(x, y + 0.18, node, ha="center", fontsize=9)

    # trayectorias de agentes seleccionados
    selected_ids = list(trajectories.keys())
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(selected_ids), 1)))
    offset_scale = 0.09 # offset

    for idx, (agent_id, path_nodes) in enumerate(trajectories.items()):
        xy = np.array([positions[n] for n in path_nodes], dtype=float)

        # offset
        dx = ((idx % 5) - 2) * offset_scale
        dy = ((idx // 5) - 0.5) * offset_scale
        xy_plot = xy + np.array([dx, dy])
        color = colors[idx % len(colors)]

        ax.plot(
            xy_plot[:, 0],
            xy_plot[:, 1],
            color=color,
            linewidth=1.8,
            alpha=0.9,
            label=f"Agent {agent_id}",
            zorder=3,
        )
        ax.scatter(
            xy_plot[0, 0],
            xy_plot[0, 1],
            marker="s",
            s=90,
            color=color,
            edgecolors="k",
            zorder=5,
        )
        end_marker = "*" if path_nodes[-1].startswith("S") else "D" # dif si llegó o no
        end_size = 180 if end_marker == "*" else 90
        ax.scatter(
            xy_plot[-1, 0],
            xy_plot[-1, 1],
            marker=end_marker,
            s=end_size,
            color=color,
            edgecolors="k",
            zorder=6,
        )

        ax.text(
            xy_plot[0, 0],
            xy_plot[0, 1] - 0.17,
            f"start {agent_id}",
            fontsize=7,
            color=color,
            ha="center",
            zorder=7,
        )
        ax.text(
            xy_plot[-1, 0],
            xy_plot[-1, 1] + 0.17,
            f"end {agent_id}",
            fontsize=7,
            color=color,
            ha="center",
            zorder=7,
        )

    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=150)
    else:
        fig.savefig(os.path.join(results_dir, "agent_trajectories.png"), dpi=150)


if __name__ == "__main__":
    random.seed(42)
    np.random.seed(42)

    # Q-Learning env
    env_q = EvacuationEnv(n_agents=50, s0=["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]*5,
                        edges=edges, node_positions=positions)
    q_table, q_rewards, q_saved = train_q_learning(env_q, n_episodes=1000)

    # SARSA env
    env_s = EvacuationEnv(n_agents=50, s0=["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]*5,
                        edges=edges, node_positions=positions)
    s_table, s_rewards, s_saved = train_sarsa(env_s, n_episodes=1000)

    plot_progress(q_rewards, q_saved, s_rewards, s_saved)
    compare_algorithms(q_rewards, q_saved, s_rewards, s_saved, window_size=50)

    # Simulacion entrenada de 300 s
    env_eval = EvacuationEnv(
        n_agents=50,
        s0=["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"] * 5,
        edges=edges,
        node_positions=positions,
    )

    selected_agents = np.random.choice(50, size=10, replace=False)  # 10 agentes
    times, saved_over_time, trajectories = evaluate_policy_over_time(
        env_eval,
        q_table, # Q-learning mejor q sars
        max_time_seconds=300,
        selected_agents=selected_agents,
    )

    plot_saved_vs_time(times, saved_over_time, title_suffix="Q-Learning")
    plot_trajectories(trajectories)

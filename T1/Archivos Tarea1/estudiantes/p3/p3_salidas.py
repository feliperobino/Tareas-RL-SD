import itertools
import os
import random

import matplotlib.pyplot as plt
import numpy as np

from evacuation import EvacuationEnv

from p3 import train_q_learning, evaluate_policy_over_time
from p3 import plot_trajectories

EvacuationEnv.plot_graph = lambda self, G, safe_nodes=None: None # para q no popupee todas las runs

results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(results_dir, exist_ok=True)

BASE_EDGES = [
    ("A", "B", 2.6, 3, 0), ("B", "C", 2.4, 2, 0), ("C", "D", 2.2, 2, 0),
    ("E", "F", 2.8, 4, 0), ("F", "G", 2.8, 4, 0), ("G", "H", 2.8, 4, 0),
    ("E", "I", 3.2, 5, 0), ("I", "J", 3.2, 5, 0), ("J", "H", 3.2, 5, 0),
    ("A", "E", 3.0, 2, 0), ("B", "F", 2.6, 2, 0), ("C", "G", 2.6, 2, 0), ("D", "H", 3.0, 2, 0),
    ("B", "E", 2.8, 2, 0), ("C", "F", 2.6, 1, 0), ("C", "H", 3.4, 2, 0),
    ("F", "I", 2.8, 3, 0), ("G", "J", 2.8, 3, 0),
]

BASE_POSITIONS = {
    "A": (0, 2), "B": (2, 2), "C": (4, 2), "D": (6, 2),
    "E": (0, 0), "F": (2, 0), "G": (4, 0), "H": (6, 0),
    "I": (1, -2), "J": (5, -2),
}

START_NODES = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]
S0 = START_NODES * 5

# parámetros de salidas iguales a base
S1_LENGTH, S1_CAPACITY = 2.0, 2
S2_LENGTH, S2_CAPACITY = 2.5, 3

# helper para unir salidas a grafo base
def build_config_edges(s1_attach, s2_attach):
    edges = list(BASE_EDGES)
    edges.append((s1_attach, "S1", S1_LENGTH, S1_CAPACITY, 0))
    edges.append((s2_attach, "S2", S2_LENGTH, S2_CAPACITY, 0))
    return edges


def build_config_positions(s1_attach, s2_attach):
    positions = dict(BASE_POSITIONS)
    x1, y1 = positions[s1_attach]
    x2, y2 = positions[s2_attach]

    # Graficar salidas en nodos attatched
    positions["S1"] = (x1 + 2.0, y1 + 0.4)
    positions["S2"] = (x2 + 2.0, y2 - 0.4)
    return positions

def make_env(s1_attach, s2_attach):
    return EvacuationEnv(
        n_agents=50,
        s0=S0,
        edges=build_config_edges(s1_attach, s2_attach),
        node_positions=build_config_positions(s1_attach, s2_attach),
    )

def evaluate_configuration_with_q_learning(s1_attach, s2_attach, n_episodes=200):
    # Reentrenar Q-learning para el grafo específico
    env_train = make_env(s1_attach, s2_attach)
    q_table, q_rewards, q_saved = train_q_learning(env_train, n_episodes=n_episodes)

    env_eval = make_env(s1_attach, s2_attach)
    times, saved, trajectories = evaluate_policy_over_time(
        env_eval,
        q_table,
        max_time_seconds=300,
        selected_agents=list(range(10)),
    )

    return {
        "config": (s1_attach, s2_attach),
        "policy": "Q-learning(retrained)",
        "times": times,
        "saved": saved,
        "trajectories": trajectories,
        "train_reward_last50": float(np.mean(q_rewards[-50:])),
        "train_saved_last50": float(np.mean(q_saved[-50:])),
        "score": float(saved[-1]), # rankiar por más salvados en última época
    }


def plot_best_vs_original(original_res, best_res):
    fig, ax = plt.subplots(1, 1, figsize=(9, 5))
    ax.plot(original_res["times"], original_res["saved"], linewidth=2, label=f"Original ({original_res['config']}, {original_res['policy']})")
    ax.plot(best_res["times"], best_res["saved"], linewidth=2, label=f"Best ({best_res['config']}, {best_res['policy']})")
    ax.set_title("Evacuated agents vs time (max 300s)")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Evacuated agents")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(results_dir, "config_best_vs_original.png"), dpi=150)


def plot_config_trajectory(res, rank_label):
    cfg = res["config"]
    s1_attach, s2_attach = cfg
    cfg_edges = build_config_edges(s1_attach, s2_attach)
    cfg_positions = build_config_positions(s1_attach, s2_attach)
    trajectories = res["trajectories"]

    fname = f"trajectory_{rank_label.lower().replace(' ', '_')}_S1_{s1_attach}_S2_{s2_attach}.png"
    plot_trajectories(
        trajectories,
        edges=cfg_edges,
        positions=cfg_positions,
        title=f"{rank_label} | S1->{s1_attach}, S2->{s2_attach} | Final evacuated={int(res['saved'][-1])}",
        save_path=os.path.join(results_dir, fname),
    )


def plot_saved_over_time_selected(config_results):
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    for res in config_results:
        s1, s2 = res["config"]
        label = f"S1->{s1}, S2->{s2} (final={int(res['saved'][-1])})"
        ax.plot(res["times"], res["saved"], linewidth=2, label=label)

    ax.set_title("People saved over time (Base + Top 5 configs)")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Evacuated agents")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(results_dir, "top5_plus_base_saved_over_time.png"), dpi=150)

def main():
    random.seed(42)
    np.random.seed(42)

    # Base config
    original_cfg = ("D", "H")

    candidate_nodes = START_NODES
    all_cfgs = [(a, b) for a, b in itertools.permutations(candidate_nodes, 2)]

    # evaluar todas las configuraciones
    sampled_cfgs = all_cfgs

    results = []
    for idx, cfg in enumerate(sampled_cfgs, start=1):
        print(f"Evaluating config {idx}/{len(sampled_cfgs)}: S1->{cfg[0]}, S2->{cfg[1]}")
        res = evaluate_configuration_with_q_learning(
            cfg[0],
            cfg[1],
            n_episodes=200,
        )
        results.append(res)

    # mejores configuraciones
    best_res = max(results, key=lambda r: r["score"])

    # meter original para comparar
    original_res = next((r for r in results if r["config"] == original_cfg), None)
    if original_res is None:
        original_res = evaluate_configuration_with_q_learning(
            original_cfg[0],
            original_cfg[1],
            n_episodes=200,
        )

    plot_best_vs_original(original_res, best_res)

    # Base y top 5 configs
    ranked = sorted(results, key=lambda r: r["score"], reverse=True)
    top5_ex_base = [r for r in ranked if r["config"] != original_cfg][:5] # mejores sin la base
    selected_6 = [original_res] + top5_ex_base

    # trayectorias
    plot_config_trajectory(original_res, rank_label="Base")
    for k, res in enumerate(top5_ex_base, start=1):
        plot_config_trajectory(res, rank_label=f"Top {k}")

    # saved de todas
    plot_saved_over_time_selected(selected_6)

if __name__ == "__main__":
    main()

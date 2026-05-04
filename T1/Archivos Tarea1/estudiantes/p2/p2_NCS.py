import os

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta as beta_dist

from channel_selector import ChannelSelector


def thompson_sampling_simulation(n_steps=2500, n_channels=6, seed=0, ts_comm=0.05):
    # selector de canales
    channel_selector = ChannelSelector(n_channels=n_channels, seed=seed)

    alpha = np.ones(n_channels) # alpha inicial = 1 por canal
    beta_params = np.ones(n_channels) # beta tb

    chosen_channels = np.zeros(n_steps, dtype=int) # canal de cada paso
    # diccionario de veces elegidas por canal
    chosen_channels_dict = {i: [] for i in range(n_channels)}

    rewards = np.zeros(n_steps) 
    cumulative_reward = np.zeros(n_steps)
    cumulative_regret = np.zeros(n_steps)
    estimated_means = np.zeros((n_steps, n_channels))
    true_probs = channel_selector.probs.copy()
    best_prob = float(np.max(true_probs))
    rng = np.random.default_rng(seed)

    for k in range(n_steps): 
        # clase MAB Thompson Sampling
        # beta sampling
        samples = rng.beta(alpha, beta_params)

        # elegimos el canal con mayor valor
        channel = int(np.argmax(samples)) 
        chosen_channels[k] = channel
        chosen_channels_dict[channel].append(k)

        # apply action y reward
        success = channel_selector.send_u(np.zeros((2, 1)), channel=channel)
        reward = 1.0 if success else 0.0

        rewards[k] = reward
        cumulative_reward[k] = rewards[:k + 1].sum()
        cumulative_regret[k] = (cumulative_regret[k - 1] if k > 0 else 0.0) + (best_prob - true_probs[channel])

        # actualizar distribuciones
        if reward > 0:
            alpha[channel] += 1.0
        else:
            beta_params[channel] += 1.0

        estimated_means[k, :] = alpha / (alpha + beta_params)

    return {
        "ts": ts_comm,
        "true_probs": true_probs,
        "best_prob": best_prob,
        "chosen_channels": chosen_channels,
        "rewards": rewards,
        "cumulative_reward": cumulative_reward,
        "cumulative_regret": cumulative_regret,
        "estimated_means": estimated_means,
        "alpha": alpha,
        "beta": beta_params,
        "chosen_channels_dict": chosen_channels_dict,
    }


def epsilon_greedy_simulation(true_probs, n_steps=2500, epsilon=0.1, seed=1, ts_comm=0.05):
    # igual pero sin alpha beta y distribuciones
    n_channels = len(true_probs)
    channel_selector = ChannelSelector(n_channels=n_channels, seed=seed)
    channel_selector.probs = np.array(true_probs, dtype=float)

    counts = np.zeros(n_channels, dtype=int)
    means = np.zeros(n_channels)

    chosen_channels = np.zeros(n_steps, dtype=int)
    chosen_channels_dict = {i: [] for i in range(n_channels)}
    rewards = np.zeros(n_steps)
    cumulative_reward = np.zeros(n_steps)
    cumulative_regret = np.zeros(n_steps)
    best_prob = float(np.max(true_probs))
    rng = np.random.default_rng(seed)

    for k in range(n_steps):
        explore = rng.random() < epsilon
        if explore or np.all(counts == 0):
            channel = int(rng.integers(0, n_channels))
        else:
            channel = int(np.argmax(means))

        chosen_channels[k] = channel
        chosen_channels_dict[channel].append(k)

        success = channel_selector.send_u(np.zeros((2, 1)), channel=channel)
        reward = 1.0 if success else 0.0

        rewards[k] = reward
        cumulative_reward[k] = rewards[:k + 1].sum()
        cumulative_regret[k] = (cumulative_regret[k - 1] if k > 0 else 0.0) + (best_prob - true_probs[channel])

        counts[channel] += 1
        means[channel] += (reward - means[channel]) / counts[channel]

    return {
        "ts": ts_comm,
        "true_probs": np.array(true_probs, dtype=float),
        "best_prob": best_prob,
        "chosen_channels": chosen_channels,
        "rewards": rewards,
        "cumulative_reward": cumulative_reward,
        "cumulative_regret": cumulative_regret,
        "means": means,
        "counts": counts,
        "epsilon": epsilon,
        "chosen_channels_dict": chosen_channels_dict,
    }

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(base_dir, "results")
    os.makedirs(results_dir, exist_ok=True)

    # params
    n_steps=2500
    n_channels=6
    seed=0

    # sampling
    ts_comm=0.05

    # epsilon greedy
    epsilon=0.1

    ## THOMPSON SAMPLING
    comm = thompson_sampling_simulation(n_steps=n_steps, n_channels=n_channels, seed=seed, ts_comm=ts_comm)
    t_comm = np.arange(len(comm["rewards"])) * comm["ts"]
    best_channel = int(np.argmax(comm["true_probs"]))

    fig2, axes2 = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    axes2[0].plot(t_comm, comm["cumulative_reward"], lw=1.4)
    axes2[0].set_ylabel("cumulative reward")
    axes2[0].grid(True, alpha=0.3)
    axes2[0].set_title("Thompson Sampling for wireless channel selection")

    axes2[1].step(t_comm, comm["chosen_channels"], where="post", lw=1.0)
    axes2[1].axhline(best_channel, color="k", ls=":", lw=1.0, label=f"best channel = {best_channel}")
    axes2[1].set_ylabel("selected channel")
    axes2[1].set_xlabel(f"time [s] (Ts={comm['ts']})")
    axes2[1].grid(True, alpha=0.3)
    axes2[1].legend()

    fig2.tight_layout()
    fig2_path = os.path.join(results_dir, "thompson_sampling_channel_reward.png")
    fig2.savefig(fig2_path, dpi=150)

    x_beta = np.linspace(0.0, 1.0, 400)
    fig3, ax3 = plt.subplots(1, 1, figsize=(13, 7))
    colors = plt.cm.tab10(np.linspace(0, 1, len(comm["alpha"])))
    for i in range(len(comm["alpha"])):
        a_i = comm["alpha"][i]
        b_i = comm["beta"][i]
        y_beta = beta_dist.pdf(x_beta, a_i, b_i)
        mean_i = a_i / (a_i + b_i)
        y_mean = beta_dist.pdf(mean_i, a_i, b_i)
        n_selected = len(comm["chosen_channels_dict"][i])

        ax3.plot(x_beta, y_beta, lw=1.8, color=colors[i], label=f"Ch {i}")
        ax3.axvline(comm["true_probs"][i], color=colors[i], ls=":", lw=1.0, alpha=0.8)
        ax3.scatter([mean_i], [y_mean], color=colors[i], s=22, zorder=3)
        ax3.text(mean_i, y_mean, f" n={n_selected}", color=colors[i], fontsize=9, va="bottom")

    ax3.set_title("Posterior Beta distributions (all channels)")
    ax3.set_xlabel("p")
    ax3.set_ylabel("pdf")
    ax3.grid(True, alpha=0.3)
    ax3.legend(ncol=3, fontsize=9)
    fig3.tight_layout()
    fig3_path = os.path.join(results_dir, "thompson_beta_posteriors.png")
    fig3.savefig(fig3_path, dpi=150)

    ## EPLSILON GREEDY
    eps_comm = epsilon_greedy_simulation(comm["true_probs"], n_steps=n_steps, epsilon=epsilon, seed=1, ts_comm=ts_comm)

    fig_eps, axes_eps = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    axes_eps[0].plot(t_comm, eps_comm["cumulative_reward"], lw=1.4)
    axes_eps[0].set_ylabel("cumulative reward")
    axes_eps[0].grid(True, alpha=0.3)
    axes_eps[0].set_title(f"Epsilon-greedy for wireless channel selection (eps={epsilon})")

    axes_eps[1].step(t_comm, eps_comm["chosen_channels"], where="post", lw=1.0)
    axes_eps[1].axhline(best_channel, color="k", ls=":", lw=1.0, label=f"best channel = {best_channel}")
    axes_eps[1].set_ylabel("selected channel")
    axes_eps[1].set_xlabel(f"time [s] (Ts={comm['ts']})")
    axes_eps[1].grid(True, alpha=0.3)
    axes_eps[1].legend()

    fig_eps.tight_layout()
    fig_eps_path = os.path.join(results_dir, "epsilon_greedy_channel_reward.png")
    fig_eps.savefig(fig_eps_path, dpi=150)

    fig4, ax4 = plt.subplots(1, 1, figsize=(12, 6))
    ax4.plot(t_comm, comm["cumulative_regret"], label="Thompson Sampling", lw=1.6)
    ax4.plot(t_comm, eps_comm["cumulative_regret"], label=f"epsilon-greedy (eps={epsilon})", lw=1.6)
    ax4.set_xlabel(f"time [s] (Ts={comm['ts']})")
    ax4.set_ylabel("cumulative regret")
    ax4.grid(True, alpha=0.3)
    ax4.legend()
    ax4.set_title("Cumulative regret comparison")
    fig4.tight_layout()
    fig4_path = os.path.join(results_dir, "regret_thompson_vs_epsilon_greedy.png")
    fig4.savefig(fig4_path, dpi=150)

    print("=== THOMPSON SAMPLING / CHANNEL SELECTION ===")
    print(f"True channel reliabilities: {comm['true_probs']}")
    print(f"Best true channel: {best_channel}")
    print(f"Final cumulative regret: {comm['cumulative_regret'][-1]:.4f}")
    print(f"Final cumulative reward: {comm['cumulative_reward'][-1]:.2f}")

    print("=== EPSILON-GREEDY COMPARISON ===")
    print(f"Final cumulative regret (epsilon-greedy): {eps_comm['cumulative_regret'][-1]:.4f}")
    print(f"Final cumulative reward (epsilon-greedy): {eps_comm['cumulative_reward'][-1]:.2f}")

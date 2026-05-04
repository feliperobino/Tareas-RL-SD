
# La llave principal (m,a, n) es el momento del día. La llave secundaria (1,2,3) es el valor de la variable en el tiempo t
# El diccionario interior son los valores en t+1 con su respectiva probabilidad
# 'm': {1: {1: 0.7, 2: 0.2, 3: 0.1} -> En este caso: Si es de mañana y el valor en t es 1, hay un 70% de probabilidad de permanecer en 1, 20% de transicionar a 2 y 10% de transicionar a 3
import numpy as np
import matplotlib.pyplot as plt
import os

L_trans_probs = {
    'm': {
        1: {1: 0.7, 2: 0.2, 3: 0.1},
        2: {1: 0.2, 2: 0.6, 3: 0.2},
        3: {1: 0.1, 2: 0.3, 3: 0.6},
    },
    'a': {
        1: {1: 0.6, 2: 0.3, 3: 0.1},
        2: {1: 0.2, 2: 0.5, 3: 0.3},
        3: {1: 0.1, 2: 0.3, 3: 0.6},    
    },
    'n': {
        1: {1: 0.6, 2: 0.3, 3: 0.1},
        2: {1: 0.2, 2: 0.5, 3: 0.3},
        3: {1: 0.1, 2: 0.2, 3: 0.7},
    }
}

G_trans_probs = {
    'm': {
        0: {0: 0.6, 1: 0.3, 2: 0.1, 3: 0.0},
        1: {0: 0.2, 1: 0.5, 2: 0.3, 3: 0.0},
        2: {0: 0.1, 1: 0.3, 2: 0.6, 3: 0.0},
        3: {0: 0.1, 1: 0.2, 2: 0.5, 3: 0.2},
    },
    'a': {
        0: {0: 0.4, 1: 0.3, 2: 0.2, 3: 0.1},
        1: {0: 0.2, 1: 0.4, 2: 0.3, 3: 0.1},
        2: {0: 0.1, 1: 0.2, 2: 0.4, 3: 0.3},
        3: {0: 0.05, 1: 0.15, 2: 0.3, 3: 0.5},
    },
    'n': {
        0: {0: 1.0, 1: 0.0, 2: 0.0, 3: 0.0},
        1: {0: 1.0, 1: 0.0, 2: 0.0, 3: 0.0},
        2: {0: 1.0, 1: 0.0, 2: 0.0, 3: 0.0},
        3: {0: 1.0, 1: 0.0, 2: 0.0, 3: 0.0},
    }
}

# POLICY ITERATION clase 3
def policy_evaluation(pi, P, gamma=1.0, theta=1e-10, max_iterations=10000):
    V = np.zeros(len(P), dtype=np.float64)
    inner_iterations = 0

    for _ in range(max_iterations):
        inner_iterations += 1
        prev_V = V.copy()
        V_new = np.zeros(len(P), dtype=np.float64)

        for s in range(len(P)):
            for prob, next_state, reward, done in P[s][pi[s]]:
                continuation = 0.0 if done else gamma * V[next_state]
                V_new[s] += prob * (reward + continuation)

        V = V_new

        if np.max(np.abs(V - prev_V)) < theta:
            break

    return V, inner_iterations

def policy_improvement(V, P, gamma=1.0):
    Q = np.zeros((len(P), len(P[0])), dtype=np.float64)
    for s in range(len(P)):
        for a in range(len(P[s])):
            for prob, next_state, reward, done in P[s][a]:
                Q[s][a] += prob * (reward + gamma * V[next_state] * (not done))
    new_pi = {s:a for s, a in enumerate(np.argmax(Q, axis=1))}
    return new_pi

def policy_iteration(P, gamma=1.0, theta=1e-10, max_outer_iterations=1000, max_eval_iterations=10000):
    random_actions = np.random.choice(tuple(P[0].keys()), len(P))
    pi = {s: a for s, a in enumerate(random_actions)}
    outer_iterations = 0
    inner_iterations_total = 0
    inner_iterations_per_outer = []

    for _ in range(max_outer_iterations):
        outer_iterations += 1
        old_pi = {s:pi[s] for s in range(len(P))}
        V, inner_iters = policy_evaluation(pi, P, gamma, theta, max_eval_iterations)
        inner_iterations_total += inner_iters
        inner_iterations_per_outer.append(inner_iters)
        pi = policy_improvement(V, P, gamma)
        if old_pi == {s:pi[s] for s in range(len(P))}:
            break
    return V, pi, outer_iterations, inner_iterations_total, inner_iterations_per_outer

# VALUE ITERATION
def value_iteration(P, gamma=0.98, theta=1e-10, max_iterations=10000):
    V = np.zeros(len(P), dtype=np.float64)
    iterations = 0

    for _ in range(max_iterations):
        iterations += 1
        Q = np.zeros((len(P), len(P[0])), dtype=np.float64)

        for s in range(len(P)):
            for a in range(len(P[s])):
                for prob, next_state, reward, done in P[s][a]:
                    continuation = 0.0 if done else gamma * V[next_state]
                    Q[s][a] += prob * (reward + continuation)

        V_new = np.max(Q, axis=1)
        if np.max(np.abs(V_new - V)) < theta:
            V = V_new
            break
        V = V_new

    pi = {s: a for s, a in enumerate(np.argmax(Q, axis=1))}
    return V, pi, iterations

#########################
# MDP
# bateria
def next_battery(B, action):
    return max(0, min(5, B + action))


def apply_control(B, L, G, action):
    if action < 0:
        discharge = min(-action, B)
        effective_action = -discharge
    elif action > 0:
        solar_surplus = max(0, G - L)
        charge = min(action, 5 - B, solar_surplus)
        effective_action = charge
    else:
        effective_action = 0

    B_next = next_battery(B, effective_action)

    discharge = max(0, -effective_action)
    charge = max(0, effective_action)
    net_supply_for_load = G + discharge - charge

    diesel = max(0, L - net_supply_for_load)
    spill = max(0, net_supply_for_load - L)
    invalid_magnitude = abs(action - effective_action)

    return B_next, effective_action, diesel, spill, invalid_magnitude

# día
def nex_d(D):
    if D == 'm':
        return 'a'
    elif D == 'a':
        return 'n'
    else:
        return 'm'

# defs espacios de acciones y estados
L_states = [1, 2, 3] # demanda
G_states = [0, 1, 2, 3] # generacion solar
D_states = ['m', 'a', 'n'] # momento del día
actions = [-2, -1, 0, 1] # [-2 descargar batería fuerte, -1 descargar batería leve, 0 mantener estado, 1 cargar batería leve]
B_states = [0, 1, 2, 3, 4, 5] # carga bat

T = 300 # horizonte

# fun de reward
def reward(D, L, G, B, action):
    _, effective_action, diesel, spill, invalid_magnitude = apply_control(B, L, G, action)

    # Objetivos del enunciado: minimizar diesel, evitar déficit y no abusar de batería.
    c_diesel = 12.0 * diesel
    c_battery = 1.0 * abs(effective_action) # proporcional al uso de batería
    c_invalid = 6.0 * invalid_magnitude
    c_night_reserve = 6.0 if (D == 'n' and B <= 1 and diesel > 0) else 0.0 # reserva de noche para prever deficit de gen solar
    c_spill = 0.5 * spill # desperdiciar excedente

    return -(c_diesel + c_battery + c_invalid + c_night_reserve + c_spill)
#########################
# estados
states = []
states_to_index = {}

idx = 0
for B in B_states:
    for D in D_states:
        for L in L_states:
            for G in G_states:
                s = (B, D, L, G)
                states.append(s)
                states_to_index[s] = idx
                idx += 1

#########
# P[s][a] = [(prob, next_state, reward, done), ...]
P = {s_idx: {a_idx: [] for a_idx in range(len(actions))} for s_idx in range(len(states))}

for s_idx, (B, D, L, G) in enumerate(states):
    
    for a_idx, a in enumerate(actions):
        B_next, _, _, _, _ = apply_control(B, L, G, a)
        D_next = nex_d(D)

        # P(X_{t+1} | X_t, D_{t+1}).
        for L_next, prob_L in L_trans_probs[D_next][L].items():
            for G_next, prob_G in G_trans_probs[D_next][G].items():

                prob = prob_L * prob_G

                s_next = (B_next, D_next, L_next, G_next)
                s_next_idx = states_to_index[s_next]

                r = reward(D, L, G, B, a)

                done = False # no hay estado terminal

                P[s_idx][a_idx].append((prob, s_next_idx, r, done))

def _sample_from_probs(prob_dict, rng):
    values = list(prob_dict.keys())
    probs = list(prob_dict.values())
    return rng.choice(values, p=probs)


def simulate_policy_trajectory(pi, T, initial_state, seed=123):
    rng = np.random.default_rng(seed)

    B_hist = np.zeros(T, dtype=np.int32)
    L_hist = np.zeros(T, dtype=np.int32)
    G_hist = np.zeros(T, dtype=np.int32)
    a_hist = np.zeros(T, dtype=np.int32)
    r_hist = np.zeros(T, dtype=np.float64)
    diesel_hist = np.zeros(T, dtype=np.float64)

    state = initial_state

    for t in range(T):
        B, D, L, G = state
        s_idx = states_to_index[state]

        if isinstance(pi, list):
            # Política dependiente del tiempo (ej. value iteration finito).
            a_idx = pi[min(t, len(pi) - 1)][s_idx]
        else:
            # Política estacionaria (ej. policy iteration).
            a_idx = pi[s_idx]

        a = actions[a_idx]
        B_next, a_effective, diesel, _, _ = apply_control(B, L, G, a)

        B_hist[t] = B
        L_hist[t] = L
        G_hist[t] = G
        a_hist[t] = a_effective
        r_hist[t] = reward(D, L, G, B, a)
        diesel_hist[t] = diesel

        D_next = nex_d(D)
        # P(X_{t+1} | X_t, D_{t+1}).
        L_next = _sample_from_probs(L_trans_probs[D_next][L], rng)
        G_next = _sample_from_probs(G_trans_probs[D_next][G], rng)
        state = (B_next, D_next, L_next, G_next)

    R_cum = np.cumsum(r_hist)
    R_avg = R_cum / (np.arange(T) + 1)
    return B_hist, L_hist, G_hist, a_hist, R_cum, R_avg, r_hist, diesel_hist


def plot_policy_trajectory(title, B_hist, L_hist, G_hist, a_hist, R_cum, R_avg, save_path=None):
    t = np.arange(len(B_hist))
    fig, axes = plt.subplots(5, 1, figsize=(12, 12), sharex=True)

    axes[0].plot(t, B_hist, color='tab:blue')
    axes[0].set_ylabel('B_t')
    axes[0].set_title(title)
    axes[0].grid(alpha=0.3)

    axes[1].plot(t, L_hist, color='tab:green')
    axes[1].set_ylabel('L_t')
    axes[1].grid(alpha=0.3)

    axes[2].plot(t, G_hist, color='tab:orange')
    axes[2].set_ylabel('G_t')
    axes[2].grid(alpha=0.3)

    axes[3].step(t, a_hist, where='post', color='tab:red')
    axes[3].set_ylabel('a_t')
    axes[3].grid(alpha=0.3)

    axes[4].plot(t, R_cum, color='tab:purple', label='Reward acumulado')
    axes[4].plot(t, R_avg, color='tab:brown', linestyle='--', label='Reward promedio')
    axes[4].set_ylabel('Reward')
    axes[4].set_xlabel('Tiempo t')
    axes[4].legend()
    axes[4].grid(alpha=0.3)

    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close(fig)


def build_action_grid_by_battery_and_surplus(pi, V, D_fixed, t_policy=0):
    surplus_values = np.arange(min(G_states) - max(L_states), max(G_states) - min(L_states) + 1)
    grid = np.full((len(surplus_values), len(B_states)), np.nan)
    value_grid = np.full((len(surplus_values), len(B_states)), np.nan)

    for b_idx, B in enumerate(B_states):
        for s_idx, surplus in enumerate(surplus_values):
            action_votes = []
            state_values = []

            for L in L_states:
                for G in G_states:
                    if G - L != surplus:
                        continue

                    state = (B, D_fixed, L, G)
                    state_idx = states_to_index[state]

                    if isinstance(pi, list):
                        a_idx = pi[min(t_policy, len(pi) - 1)][state_idx]
                    else:
                        a_idx = pi[state_idx]

                    action_votes.append(int(a_idx))
                    state_values.append(V[state_idx])

            if action_votes:
                counts = np.bincount(action_votes, minlength=len(actions))
                grid[s_idx, b_idx] = float(np.argmax(counts))
                value_grid[s_idx, b_idx] = np.mean(state_values)

    return grid, value_grid, surplus_values


def plot_policy_action_heatmaps_by_day(pi_left, pi_right, V_left, V_right, title_left, title_right, t_right=0, save_dir=None):
    cmap = plt.get_cmap('RdYlBu', len(actions))

    for D_fixed in D_states:
        grid_left, value_grid_left, surplus_values = build_action_grid_by_battery_and_surplus(
            pi_left,
            V_left,
            D_fixed=D_fixed,
            t_policy=0
        )
        grid_right, value_grid_right, _ = build_action_grid_by_battery_and_surplus(
            pi_right,
            V_right,
            D_fixed=D_fixed,
            t_policy=t_right
        )

        fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True, constrained_layout=True)
        for ax, grid, value_grid, title in [(axes[0], grid_left, value_grid_left, title_left), (axes[1], grid_right, value_grid_right, title_right)]:
            im = ax.imshow(
                grid,
                origin='lower',
                aspect='auto',
                cmap=cmap,
                vmin=-0.5,
                vmax=len(actions) - 0.5,
                extent=[B_states[0] - 0.5, B_states[-1] + 0.5, surplus_values[0] - 0.5, surplus_values[-1] + 0.5]
            )

            ax.set_title(f"{title} | D_t={D_fixed}")
            ax.set_xlabel('Estado de batería B_t')
            ax.set_xticks(B_states)
            ax.set_yticks(surplus_values)
            ax.grid(alpha=0.2)

            for yi, surplus in enumerate(surplus_values):
                for xi, B in enumerate(B_states):
                    if np.isnan(grid[yi, xi]):
                        continue
                    action_idx = int(grid[yi, xi])
                    value = value_grid[yi, xi]
                    text_str = f"a={actions[action_idx]}\nv={value:.1f}"
                    ax.text(B, surplus, text_str, ha='center', va='center', color='black', fontsize=7)

        axes[0].set_ylabel('Excedente energético (G_t - L_t)')

        cbar = fig.colorbar(im, ax=axes, fraction=0.046, pad=0.04)
        cbar.set_label('Acción óptima')
        cbar.set_ticks(np.arange(len(actions)))
        cbar.set_ticklabels([str(a) for a in actions])

        if save_dir is not None:
            plt.savefig(os.path.join(save_dir, f'policy_comparison_heatmap_D_{D_fixed}.png'), dpi=200, bbox_inches='tight')
        plt.close(fig)


IMG_DIR = os.path.join(os.path.dirname(__file__), 'img')
os.makedirs(IMG_DIR, exist_ok=True)

GAMMA = 0.98
THETA = 1e-8
MAX_ITER = 10000

# Ejecutar policy/value iteration para el mismo problema estacionario descontado
V_pi, pi_pi, pi_outer, pi_inner_total, pi_inner_by_outer = policy_iteration(
    P,
    gamma=GAMMA,
    theta=THETA,
    max_outer_iterations=MAX_ITER,
    max_eval_iterations=MAX_ITER
)
V_vi, pi_vi, iters_vi = value_iteration(P, gamma=GAMMA, theta=THETA, max_iterations=MAX_ITER)

print("Policy Iteration:")
print("Valor de los estados:", V_pi)
print("Iteraciones externas:", pi_outer)
print("Iteraciones internas totales (evaluación de política):", pi_inner_total)
print("Iteraciones internas por ciclo externo:", pi_inner_by_outer)
print("Iteraciones totales PI (internas + externas):", pi_outer + pi_inner_total)

print("\nValue Iteration:")
print("Valor de los estados:", V_vi)
print("Iteraciones:", iters_vi)

# Estado inicial
initial_state = (3, 'm', 2, 1)

# Gráfico Policy Iteration
B_pi, L_pi, G_pi, a_pi, Rcum_pi, Ravg_pi, r_pi, diesel_pi = simulate_policy_trajectory(pi_pi, T, initial_state, seed=123)
plot_policy_trajectory(
    'Policy Iteration: B_t, L_t, G_t, a_t y rewards',
    B_pi,
    L_pi,
    G_pi,
    a_pi,
    Rcum_pi,
    Ravg_pi,
    save_path=os.path.join(IMG_DIR, 'policy_iteration_trajectory.png')
)

# Gráfico Value Iteration
B_vi, L_vi, G_vi, a_vi, Rcum_vi, Ravg_vi, r_vi, diesel_vi = simulate_policy_trajectory(pi_vi, T, initial_state, seed=123)
plot_policy_trajectory(
    'Value Iteration: B_t, L_t, G_t, a_t y rewards',
    B_vi,
    L_vi,
    G_vi,
    a_vi,
    Rcum_vi,
    Ravg_vi,
    save_path=os.path.join(IMG_DIR, 'value_iteration_trajectory.png')
)

print('\nResumen trayectoria (Policy Iteration):')
print(f'Reward promedio: {np.mean(r_pi):.3f}')
print(f'Diesel total usado: {np.sum(diesel_pi):.3f}')
print(f'Uso total batería (|a_t|): {np.sum(np.abs(a_pi)):.3f}')

print('\nResumen trayectoria (Value Iteration):')
print(f'Reward promedio: {np.mean(r_vi):.3f}')
print(f'Diesel total usado: {np.sum(diesel_vi):.3f}')
print(f'Uso total batería (|a_t|): {np.sum(np.abs(a_vi)):.3f}')

# Políticas de VI y PI separadas por momento del día (D_t = m, a, n)
plot_policy_action_heatmaps_by_day(
    pi_pi,
    pi_vi,
    V_pi,
    V_vi,
    'Policy Iteration (estacionaria)',
    'Value Iteration (estacionaria)',
    t_right=0,
    save_dir=IMG_DIR
)

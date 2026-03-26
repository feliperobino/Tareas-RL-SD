from evacuation import EvacuationEnv

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


env = EvacuationEnv(n_agents=50, s0=["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]*5,
                        edges=edges, node_positions=positions)
print(env.get_neighbors('A'))

n_episodes = 1000
for ep in range(n_episodes):
    states = env.reset()
    valid_actions = env.get_valid_actions_for_current_state()
    step_count = 0
    max_env_steps = 60 # 5 second steps
    while step_count < max_env_steps and not all(env.done_list):
        actions = [0] * env.n_agents
        next_states, rewards, arrived_to_node, times_to_arrival, global_time, done_list, just_arrived_to_shelter, arrival_event_time = env.step(actions)
        print(arrival_event_time)

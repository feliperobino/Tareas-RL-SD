import networkx as nx
import matplotlib.pyplot as plt
import math
import numpy as np


class EvacuationEnv:
    def __init__(self, n_agents=5, s0=['A', 'B', 'C', 'D', 'E']*5, edges=[],
                 node_positions={}):
        self.n_agents = n_agents
        self.s0 = s0
        self.edges = edges
        self.node_positions = node_positions
        self.graph = self.build_graph(self.edges, self.node_positions)
        self.plot_graph(self.graph)
        self.max_degree = int(max(dict(self.graph.degree()).values()))
        self.action_space = [i for i in range(self.max_degree)]
        self.last_state = [self.get_full_state(s0[i]) for i in range(n_agents)]
        self.current_state = [self.get_full_state(s0[i]) for i in range(n_agents)]
        self.arrived_to_node = [True]*n_agents
        self.done_list = [False]*n_agents
        self.time_to_arrivals = [0]*n_agents
        self.global_time = 0

    def build_graph(self, edges, positions):
        G = nx.Graph()

        for n, pos in positions.items():
            G.add_node(n, pos=pos)

        for u, v, length, capacity, occupancy in edges:
            G.add_edge(u, v, length=length, capacity=capacity, occupancy=occupancy)

        return G


    def plot_graph(self, G, safe_nodes=None):
        pos = nx.get_node_attributes(G, "pos")
        if not pos:
            pos = nx.spring_layout(G, seed=0)

        node_colors = []
        for n in G.nodes():
            if safe_nodes and n in safe_nodes:
                node_colors.append("lightgreen")
            else:
                node_colors.append("lightblue")

        plt.figure(figsize=(10, 5))
        nx.draw(
            G,
            pos,
            with_labels=True,
            node_color=node_colors,
            node_size=900,
            font_size=10,
        )

        edge_labels = {
            (u, v): f"L={d['length']}, C={d['capacity']}"
            for u, v, d in G.edges(data=True)
        }
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)
        plt.title("12-node evacuation graph")
        plt.axis("off")
        plt.show()

    def get_valid_actions_for_current_state(self):
        valid_actions = []
        for i in range(self.n_agents):
            node = self.current_state[i][0]
            neighbors = self.get_neighbors(node)
            valid_actions.append([j for j in range(len(neighbors))])
        return valid_actions


    def get_full_state(self, node):
        neighs_nodes = self.get_neighbors(node)
        densities = [self.get_density(node, neighs_nodes[i]) for i in range(len(neighs_nodes))]
        state = [node] + densities
        if len(densities) < self.max_degree:
            state = state + [-1]*(self.max_degree - len(densities))
        return tuple(state)

    def get_neighbors(self, node):
        neighs = list(self.graph[node].keys())
        neighs.sort()
        return neighs

    def get_edge(self, node1, node2):
        return self.graph[node1][node2]

    def get_density(self, node1, node2):
        capacity = self.graph[node1][node2]['capacity']
        occupancy = self.graph[node1][node2]['occupancy']
        density = occupancy/capacity

        if density < 0.33:
            return 0
        elif density >= 0.33 and density < 0.66:
            return 1
        elif density >= 0.66 and density < 1:
            return 2
        else:
            return 3


    def get_velocity(self, node1, node2):
        density = self.get_density(node1, node2)

        if density == 0:
            return 1
        elif density == 1:
            return 0.5
        elif density == 2:
            return 0.3
        else:
            return 0.1

    def get_travel_time(self, node1, node2):
        if node1 == node2:
            return 0

        velocity = self.get_velocity(node1, node2)
        length = self.graph[node1][node2]['length']

        travel_time = math.ceil(length/velocity)
        return travel_time

    def transition(self, state, action):
        current_node = state[0]
        neighbors = self.get_neighbors(current_node)
        if action >= len(neighbors) or action < 0:
            return current_node
        else:
            next_node = neighbors[action]
            return next_node

    def update_occupancy(self, node1, node2, increase=1):
        if node1 == node2:
            return
        current_occupancy = self.graph[node1][node2]['occupancy']
        capacity = self.graph[node1][node2]['capacity']
        occupancy = max(current_occupancy + increase, 0)
        self.graph[node1][node2]['occupancy'] = occupancy


    def step_dep(self, actions):
        # See who has arrived
        for i in range(len(actions)):
            if not self.arrived_to_node[i]:
                self.time_to_arrivals[i] -= 1
                if self.time_to_arrivals[i] <= 0:
                    self.time_to_arrivals[i] = 0
                    self.arrived_to_node[i] = True

        # First we make the transitions
        for i in range(len(actions)):
            if self.arrived_to_node[i]: # If this agent has arrived to destination
                last_node = self.last_state[i][0]
                current_node = self.current_state[i][0]
                if current_node[0] == 'S':
                    next_node = current_node
                else:
                    next_node = self.transition(self.current_state[i], actions[i])

                # Update occupancies
                self.update_occupancy(last_node, current_node, increase=-1)
                self.update_occupancy(current_node, next_node, increase=1)
            else: # It hasn't arrived to the destination, do nothing
                pass

        # Given the new positions, we calculate the rewards
        rewards_list = []
        next_nodes = []
        for i in range(len(actions)):
            if self.arrived_to_node[i] and not self.done_list[i]: # If this agent has arrived to destination
                current_node = self.current_state[i][0]
                next_node = self.transition(self.current_state[i], actions[i])
                travel_time = self.get_travel_time(current_node, next_node)
                self.arrived_to_node[i] = False
                self.time_to_arrivals[i] = travel_time

                if current_node[0] == 'S':
                    reward = 0
                    self.arrived_to_node[i] = True
                    self.time_to_arrivals[i] = 0
                    self.done_list[i] = True
                elif next_node[0] == 'S':
                    reward = -travel_time + 1000
                else:
                    reward = -travel_time

                rewards_list.append(reward)
                next_nodes.append(next_node)

            else:
                rewards_list.append(None)
                next_nodes.append(self.current_state[i][0])

        self.last_state = self.current_state.copy()
        self.current_state = [self.get_full_state(next_node) for next_node in next_nodes]

        self.global_time += 5

        return (self.current_state, rewards_list, self.arrived_to_node,
                self.time_to_arrivals, self.global_time, self.done_list)

    def step(self, actions):
        n = len(actions)

        # Who was already ready Before this step started?
        ready_before = self.arrived_to_node.copy()

        for i in range(n):
            if (not ready_before[i]) and (not self.done_list[i]):
                self.time_to_arrivals[i] -= 1
                if self.time_to_arrivals[i] <= 0:
                    self.time_to_arrivals[i] = 0
                    self.arrived_to_node[i] = True

        # Agents allowed to act now are only those that were
        # already ready before this call started.
        can_act_now = [ready_before[i] and (not self.done_list[i]) for i in range(n)]

        for i in range(n):
            if not can_act_now[i]:
                continue

            last_node = self.last_state[i][0]
            current_node = self.current_state[i][0]

            if current_node[0] == 'S':
                next_node = current_node
            else:
                next_node = self.transition(self.current_state[i], actions[i])

            # Remove occupancy from previous traversed edge, if any
            self.update_occupancy(last_node, current_node, increase=-1)

            # Add occupancy to new chosen edge, if any
            self.update_occupancy(current_node, next_node, increase=1)


        rewards_list = []
        next_nodes = []

        for i in range(n):
            if can_act_now[i]:
                current_node = self.current_state[i][0]

                if current_node[0] == 'S':
                    # Agent is physically at shelter and now becomes done
                    reward = 0
                    next_node = current_node
                    self.arrived_to_node[i] = True
                    self.time_to_arrivals[i] = 0
                    self.done_list[i] = True
                else:
                    next_node = self.transition(self.current_state[i], actions[i])
                    travel_time = self.get_travel_time(current_node, next_node)

                    self.arrived_to_node[i] = False
                    self.time_to_arrivals[i] = travel_time

                    if next_node[0] == 'S':
                        reward = -travel_time + 1000
                    else:
                        reward = -travel_time

                rewards_list.append(reward)
                next_nodes.append(next_node)

            else:
                rewards_list.append(None)
                next_nodes.append(self.current_state[i][0])

        self.last_state = self.current_state.copy()
        self.current_state = [self.get_full_state(node) for node in next_nodes]


        self.global_time += 5

        just_arrived = [
            (not ready_before[i]) and self.arrived_to_node[i]
            for i in range(self.n_agents)
        ]

        just_arrived_to_shelter = [
            just_arrived[i] and self.current_state[i][0][0] == 'S'
            for i in range(self.n_agents)
        ]

        arrival_event_time = [
            self.global_time if just_arrived_to_shelter[i] else None
            for i in range(self.n_agents)
        ]

        return (
            self.current_state,
            rewards_list,
            self.arrived_to_node.copy(),
            self.time_to_arrivals.copy(),
            self.global_time,
            self.done_list.copy(),
            just_arrived_to_shelter,
            arrival_event_time,
        )

    def reset(self, s0=None):
        # Reset initial positions
        if s0 is not None:
            self.s0 = list(s0[:self.n_agents])

        # Reset all edge occupancies
        for u, v in self.graph.edges():
            self.graph[u][v]['occupancy'] = 0

        # Reset states
        self.last_state = [self.get_full_state(self.s0[i]) for i in range(self.n_agents)]
        self.current_state = [self.get_full_state(self.s0[i]) for i in range(self.n_agents)]

        # Reset agent status
        self.arrived_to_node = [True] * self.n_agents
        self.done_list = [False] * self.n_agents
        self.time_to_arrivals = [0] * self.n_agents

        for i in range(self.n_agents):
            node = self.current_state[i][0]
            if node[0] == 'S':
                self.done_list[i] = True

        # Reset global time
        self.global_time = 0

        return self.current_state



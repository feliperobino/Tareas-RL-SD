import torch
import pickle
import numpy as np
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from scripts.expert_controller import WineExpertController
from scripts.fermentation_env import WineFermentationEnv
from scripts.quality_functions import maceration_extraction, get_wine_quality_score
import time

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Actor(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, act_low: np.ndarray, act_high: np.ndarray, hidden: int = 256):
        super().__init__()
        self.register_buffer("act_low", torch.as_tensor(act_low, dtype=torch.float32))
        self.register_buffer("act_high", torch.as_tensor(act_high, dtype=torch.float32))

        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden, act_dim), nn.Tanh(),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        x = self.net(obs)
        x = 0.5 * (x + 1.0)
        return self.act_low + x * (self.act_high - self.act_low)



class DAGGER:
    def __init__(self, T_lims=[8, 38], N_lims=[0, 50]):
        self.T_lims = T_lims
        self.N_lims = N_lims
        self.act_lims = [np.array([self.T_lims[0], self.N_lims[0]]), np.array([self.T_lims[1], self.N_lims[1]])]


    def get_normalization_params(self, trajectories):
        all_states = []
        for traj in trajectories:
            states = traj['trajectory']
            all_states.append(states)
        all_states = np.concatenate(all_states, axis=0)
        min_val = all_states.min(axis=0)
        min_val = np.array([max(elem*0.9, 0) for elem in min_val])
        max_val = all_states.max(axis=0)
        max_val = np.array([elem*1.1 for elem in max_val])
        return min_val, max_val

    def preprocess_data(self, trajectories, ratio=0.8):
        all_states = []
        all_actions = []
        for traj in trajectories:
            all_states.append(traj['trajectory'])
            all_actions.append(traj['actions'])

        all_states = np.concatenate(all_states, axis=0)
        all_actions = np.concatenate(all_actions, axis=0)

        states_scaled = (all_states - self.state_lims[0]) / (self.state_lims[1] - self.state_lims[0])
        actions_scaled = (all_actions - self.act_lims[0]) / (self.act_lims[1] - self.act_lims[0])


        indices = np.array([i for i in range(states_scaled.shape[0])])
        np.random.shuffle(indices)

        states_scaled = states_scaled[indices]
        actions_scaled = actions_scaled[indices]

        train_states = states_scaled[: int(ratio*len(indices))]
        train_actions = actions_scaled[: int(ratio*len(indices))]
        val_states = states_scaled[int(ratio*len(indices)):]
        val_actions = actions_scaled[int(ratio*len(indices)):]
        return (train_states, train_actions), (val_states, val_actions)


    def sup_controller(self, trajectories, epochs=500, batch_size=64):
        self.state_lims = self.get_normalization_params(trajectories)
        (train_states, train_actions), (val_states, val_actions) = self.preprocess_data(trajectories)

        actor = Actor(obs_dim=len(self.state_lims[0]), act_dim=2,
                           act_low=self.act_lims[0], act_high=self.act_lims[1]).to(device)

        optimizer = optim.Adam(actor.parameters(), lr=1e-3)
        criterion = nn.MSELoss()
        patience = 25
        best_eval = 10000
        no_better = 0


        train_loss_sup = []
        val_loss_sup = []
        for epoch in range(epochs):
            # Training
            actor.train()
            batch = 0
            train_loss = []
            while batch < len(train_states):
                X = torch.from_numpy(train_states[batch: batch + batch_size]).to(device).float()
                Y = torch.from_numpy(train_actions[batch: batch + batch_size]).to(device).float()

                optimizer.zero_grad()
                Y_pred = actor(X)
                Y_pred_norm = (Y_pred - actor.act_low)/(actor.act_high - actor.act_low)
                loss = criterion(Y_pred_norm, Y)
                loss.backward()
                optimizer.step()
                train_loss.append(loss.item())
                batch += batch_size
            train_loss = sum(train_loss) / len(train_loss)
            train_loss_sup.append(train_loss)

            # Validation
            val_loss = []
            with torch.no_grad():
                actor.eval()
                batch = 0
                while batch < len(val_states):
                    X = torch.from_numpy(val_states[batch: batch + batch_size]).to(device).float()
                    Y = torch.from_numpy(val_actions[batch: batch + batch_size]).to(device).float()
                    Y_pred = actor(X)
                    Y_pred_norm = (Y_pred - actor.act_low) / (actor.act_high - actor.act_low)
                    loss = criterion(Y_pred_norm, Y)
                    val_loss.append(loss.item())
                    batch += batch_size
            # Metrics
            val_loss = sum(val_loss) / len(val_loss)
            val_loss_sup.append(val_loss)
            if val_loss < best_eval:
                best_eval = val_loss
                no_better = 0
                print('Best Model Epoch {}|{} -> Validation Loss: {}'.format(epoch, epochs, val_loss))
                torch.save(actor.state_dict(), 'best_model.pt')
            else:
                no_better += 1
                if no_better >= patience:
                    print('Early stopping')
                    actor.load_state_dict(torch.load('best_model.pt'))
                    return actor

        actor.load_state_dict(torch.load('best_model.pt'))
        return actor

    def get_action(self, actor, observation):
        with torch.no_grad():
            observation = (observation.reshape(1, -1) - self.state_lims[0])/(self.state_lims[1] - self.state_lims[0])
            action = actor(torch.from_numpy(observation).to(device))
            return action.cpu().detach().numpy()[0]

    def linear_schedule(self, step: int, total_steps: int, start: float, end: float) -> float:
        frac = max(0.0, 1.0 - step / float(total_steps))
        return end + (start - end) * frac

    def run(self, expert_trajectories, n_trajs=100, epochs=100): #COMPLETAR
        actor = self.sup_controller(expert_trajectories, epochs=epochs)



if __name__ == "__main__":
    # Training Phase
    with open('data/expert_trajectories.pkl', 'rb') as f:
        expert_trajectories = pickle.load(f)

    dagger = DAGGER()
    actor = dagger.run(expert_trajectories, n_trajs=50)



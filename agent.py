import torch
import numpy as np
from model import ActorCritic
import torch.optim as optim
import torch.nn as nn

#1. Look at drone + goal
#2. Ask AI: “what should I do?”
#3. Add some randomness
#4. Output action to environment

class Agent:

    def __init__(self,
                 obs_dim=9,
                 action_dim=4,
                 lr=0.01,
                 gamma=0.99,
                 std=0.2):
        self.model = ActorCritic(obs_dim, action_dim)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

        # =================================================
        # Hyperparameters -- control knobs for how agent learns and behaves
        # =================================================

        # gamma - How much the agent cares about future rewards vs immediate rewards.
        #   0.0  → only cares about now
        #   0.99 → cares strongly about future
        #   1.0  → cares equally forever
        self.gamma = gamma
        # std - The exploration noise (standard deviation of action sampling).
        #   std = 0.05 - very precise, low exploration
        #   std = 0.5  - very chaotic, much exploration
        self.std = std

        # 4 action dimensions for vx, vy, vz and yaw_rate
        self.act_dim = action_dim

        # # =================================================
        # Memory buffers
        # # =================================================
        self.reset_memory()

    # =====================================================
    # CLEAR MEMORY
    # =====================================================
    def reset_memory(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.dones = []

    # =====================================================
    # STORE EXPERIENCE
    # =====================================================
    def store(self, obs, action, reward, value, done):
        self.states.append(obs)
        self.actions.append(action)
        self.rewards.append(reward)
        self.values.append(value)
        self.dones.append(done)

    # =====================================================
    # DISCOUNTED RETURNS
    # =====================================================
    def compute_returns(self):
        returns = []
        G = 0.0

        for reward, done in zip(reversed(self.rewards), reversed(self.dones)):
            if done:
                G = 0.0

            G = reward + self.gamma * G
            returns.insert(0, G)

        return np.array(returns, dtype=np.float32)

    # =====================================================
    # LEARN AFTER EPISODE
    # =====================================================
    def update(self):
        if len(self.states) == 0:
            return

        # -----------------------------
        # Prepare tensors
        # -----------------------------
        states = torch.tensor(np.array(self.states), dtype=torch.float32)
        actions = torch.tensor(np.array(self.actions), dtype=torch.float32)
        returns = torch.tensor(self.compute_returns()),

        returns = returns[0].float()
        values = torch.tensor(self.values, dtype=torch.float32)

        # normalize returns
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        # advantage = better than expected?
        advantages = returns - values

        # -----------------------------
        # Forward pass
        # -----------------------------
        mean, value_pred = self.model(states)

        dist = torch.distributions.Normal(mean, self.std)
        log_probs = dist.log_prob(actions).sum(dim=1)

        # -----------------------------
        # Losses
        # -----------------------------
        actor_loss = -(log_probs * advantages.detach()).mean()

        critic_loss = nn.functional.mse_loss(
            value_pred.squeeze(),
            returns
        )

        entropy = dist.entropy().mean()

        loss = actor_loss + 0.5 * critic_loss - 0.001 * entropy

        # -----------------------------
        # Backprop
        # -----------------------------
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # -----------------------------
        # Reset memory
        # -----------------------------
        self.reset_memory()

        print(
            f"Update | Loss: {loss.item():.3f} "
            f"| Actor: {actor_loss.item():.3f} "
            f"| Critic: {critic_loss.item():.3f}"
        )

    # =====================================================
    # SAVE MODEL
    # =====================================================
    def save(self, path="drone_model.pth"):
        torch.save(self.model.state_dict(), path)

    # =====================================================
    # LOAD MODEL
    # =====================================================
    def load(self, path="drone_model.pth"):
        self.model.load_state_dict(torch.load(path))
        self.model.eval()

    def act(self, obs):
        # =================================================
        # TAKE THE ENVIRONMENT STATE
        # =================================================
        obs = torch.tensor(obs, dtype=torch.float32)


        # =================================================
        # ASK NEURAL NETWORK WHAT TO DO
        # =================================================
        with torch.no_grad():
            mean, value = self.model(obs)
        # =================================================
        # ADD RANDOMNESS SO IT DOESN'T DO THE SAME ALL THE TIME
        # value means how good the situation is
        # =================================================

        dist = torch.distributions.Normal(mean, self.std)
        action = dist.sample()

        # keep in valid range
        action = torch.clamp(action, -1.0, 1.0)

        return action.numpy(), value


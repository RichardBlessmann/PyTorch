import torch
import numpy as np
from model import ActorCritic

#1. Look at drone + goal
#2. Ask AI: “what should I do?”
#3. Add some randomness
#4. Output action to environment

class Agent:

    def __init__(self):
        self.model = ActorCritic()
        self.std = 0.1

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

        return action.numpy(), value
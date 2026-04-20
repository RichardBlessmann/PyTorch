import time
import numpy as np

from gym_pybullet_drones.envs.HoverAviary import HoverAviary
from gym_pybullet_drones.utils.enums import DroneModel, Physics

# Create environment
env = HoverAviary(
    drone_model=DroneModel.CF2X,
    physics=Physics.PYB,
    gui=True
)

obs, info = env.reset()

print("Simulation started")

for i in range(2400):
    action = np.array([[14000, 14000, 14000, 14000]])
    obs, reward, terminated, truncated, info = env.step(action)
    time.sleep(1/240)

env.close()
print("Finished")
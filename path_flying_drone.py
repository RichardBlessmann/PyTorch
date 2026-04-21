import time
import numpy as np

from gym_pybullet_drones.envs.VelocityAviary import VelocityAviary
from gym_pybullet_drones.utils.enums import DroneModel, Physics

import udp_sender as udp
import pybullet as p

import build_obstacle_course as obst
from agent import Agent


#initial_xyzs  = spawn point
#initial_rpys  = facing direction
#obstacles     = world layout
#num_drones    = agents
#physics       = realism

#1. create env
#2. env.reset()
#3. spawn obstacles (NOW they stick)
#4. run loop





start = np.array([0, 0, 1.0])
goal = np.array([5.0, 0.0, 1.0])
start_rpy = [0.0, 0.0, 1.57]

agent = Agent()

# Create environmentW
env = VelocityAviary(
    drone_model=DroneModel.CF2X,
    num_drones=1,
    gui=True,
    obstacles=True,
    initial_xyzs = np.array([start]), #1 meter high
    initial_rpys = np.array([start_rpy]) #roll, pitch, yaw in radian
)
#Time to initialize

time.sleep(1)
# 2. RESET (IMPORTANT!)
obs, info = env.reset()

# =================================================
# add random objects to fly through! later! first just fly through
# =================================================

#obst.spawn_curved_tower_course(start, goal, client=env.CLIENT, difficulty=1)

#visual markers /start/end
p.loadURDF("sphere2.urdf", start, globalScaling=0.3)
p.loadURDF("sphere2.urdf", goal, globalScaling=0.3)


state = env._getDroneStateVector(0)
pos = state[0:3]


print("Simulation started")


#Run simulation
for i in range(4 * 2400):

    state = env._getDroneStateVector(0)
    obs = state[0:9]  # adjust if needed

    action, value = agent.act(obs)

    # scale action properly
    vx, vy, vz, yaw_rate = action[0]

    action_env = np.array([[vx * 2, vy * 2, vz * 1, yaw_rate]])

    obs, reward, terminated, truncated, info = env.step(action_env)


env.close()
print("Finished")